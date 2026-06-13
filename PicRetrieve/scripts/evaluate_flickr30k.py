#!/usr/bin/env python
"""Flickr30k 标准 text-to-image 检索评测。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from app.embedder import ClipEmbedder, l2_normalize
from app.index_store import IndexStore


def metadata_record(item: dict[str, Any]) -> dict[str, Any]:
    """! @brief 提取 bootstrap 写入的 metadata.jsonl 记录。"""

    metadata = item.get("metadata") or {}
    return metadata.get("metadata_jsonl") or {}


def benchmark_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """! @brief 读取 Flickr30k 评测字段。"""

    record = metadata_record(item)
    metadata = record.get("metadata") or {}
    if metadata.get("benchmark") != "flickr30k":
        return {}
    return metadata


def collect_queries(
    store: IndexStore,
    ids: np.ndarray,
    limit_queries: int,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """! @brief 从索引条目中收集 caption 查询和对应图片向量行号。"""

    items = {int(item["id"]): item for item in store.list_items()}
    texts: list[str] = []
    positive_rows: list[int] = []
    images: list[dict[str, Any]] = []

    for row_idx, item_id in enumerate(ids):
        item = items.get(int(item_id))
        if item is None:
            continue
        metadata = benchmark_metadata(item)
        captions = metadata.get("captions") or []
        if not captions:
            continue
        images.append(
            {
                "item_id": int(item_id),
                "row": row_idx,
                "rel_path": item["rel_path"],
                "image_key": metadata.get("image_key", ""),
                "caption_count": len(captions),
            }
        )
        for caption in captions:
            if limit_queries and len(texts) >= limit_queries:
                return texts, positive_rows, images
            texts.append(str(caption))
            positive_rows.append(row_idx)
    return texts, positive_rows, images


def top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """! @brief 取单条查询分数最高的若干图片行号。"""

    if limit >= scores.shape[0]:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    return candidates[np.argsort(-scores[candidates])]


def evaluate_batches(
    embedder: ClipEmbedder,
    image_embeddings: np.ndarray,
    texts: list[str],
    positive_rows: list[int],
    batch_size: int,
    top_ks: list[int],
) -> dict[str, Any]:
    """! @brief 分 batch 编码 caption 并计算 Recall@K、MRR 和 rank。"""

    max_k = max(top_ks)
    hits = {k: 0 for k in top_ks}
    reciprocal_ranks: list[float] = []
    ranks: list[int] = []
    examples: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="flickr30k eval", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        batch_positive_rows = positive_rows[start : start + batch_size]
        text_embeddings = embedder.encode_texts(batch_texts, batch_size=batch_size)
        scores = text_embeddings @ image_embeddings.T

        for local_idx, positive_row in enumerate(batch_positive_rows):
            row_scores = scores[local_idx]
            positive_score = row_scores[positive_row]
            rank = int(np.count_nonzero(row_scores > positive_score) + 1)
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            for k in top_ks:
                if rank <= k:
                    hits[k] += 1
            if len(examples) < 5:
                top_rows = top_indices(row_scores, max_k).tolist()
                examples.append(
                    {
                        "query": batch_texts[local_idx],
                        "positive_row": int(positive_row),
                        "rank": rank,
                        "top_rows": [int(row) for row in top_rows],
                    }
                )

    total = max(len(texts), 1)
    metrics: dict[str, Any] = {f"recall@{k}": hits[k] / total for k in top_ks}
    metrics["mrr"] = sum(reciprocal_ranks) / total
    metrics["median_rank"] = statistics.median(ranks) if ranks else 0
    metrics["mean_rank"] = sum(ranks) / total if ranks else 0
    return {"metrics": metrics, "examples": examples}


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建 Flickr30k 评测参数解析器。"""

    parser = argparse.ArgumentParser(prog="evaluate_flickr30k.py")
    parser.add_argument("--data-dir", type=Path, default=Path("data/flickr30k_index"))
    parser.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    args = build_parser().parse_args()
    store = IndexStore(args.data_dir)
    image_embeddings, ids = store.load_embeddings()
    if image_embeddings.size == 0:
        raise SystemExit("empty index; run app.cli index on data/flickr30k_test/images first")
    image_embeddings = l2_normalize(image_embeddings)
    top_ks = sorted({k for k in args.top_k if k > 0})
    if not top_ks:
        raise SystemExit("--top-k must contain at least one positive integer")

    texts, positive_rows, images = collect_queries(store, ids, args.limit_queries)
    if not texts:
        raise SystemExit("no Flickr30k benchmark captions found in this index")

    embedder = ClipEmbedder(args.model_name, args.device)
    result = evaluate_batches(
        embedder=embedder,
        image_embeddings=image_embeddings,
        texts=texts,
        positive_rows=positive_rows,
        batch_size=args.batch_size,
        top_ks=top_ks,
    )
    summary = {
        "dataset": "clip-benchmark/wds_flickr30k",
        "task": "text-to-image-retrieval",
        "metric_note": "Pure CLIP image/text embedding retrieval; metadata/FTS is not used.",
        "data_dir": str(args.data_dir),
        "model_name": args.model_name,
        "images": len(images),
        "queries": len(texts),
        "top_k": top_ks,
        **result,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
