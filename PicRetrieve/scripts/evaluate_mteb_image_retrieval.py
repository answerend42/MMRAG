#!/usr/bin/env python
"""MTEB 图搜图检索 benchmark 评测。"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from app.embedder import ClipEmbedder, l2_normalize
from scripts.bootstrap_data import download_hf_dataset_file


def image_from_cell(value: Any) -> Image.Image | None:
    """! @brief 从 parquet image 单元格读取 PIL 图片。"""

    if not isinstance(value, dict):
        return None
    image_bytes = value.get("bytes")
    if not image_bytes:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def encode_parquet_images(
    path: Path,
    embedder: ClipEmbedder,
    batch_size: int,
    limit: int,
) -> tuple[list[str], np.ndarray]:
    """! @brief 编码 MTEB parquet 中的 image 行。"""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch_images: list[Image.Image] = []
    total = min(limit, parquet.metadata.num_rows) if limit else parquet.metadata.num_rows

    def flush() -> None:
        if not batch_images:
            return
        embeddings.append(embedder.encode_images(batch_images, batch_size=batch_size))
        ids.extend(batch_ids)
        batch_ids.clear()
        batch_images.clear()

    with tqdm(total=total, desc=path.name, unit="img") as progress:
        for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "image"]):
            for row in record_batch.to_pylist():
                if limit and len(ids) + len(batch_ids) >= limit:
                    break
                image = image_from_cell(row.get("image"))
                if image is None:
                    continue
                batch_ids.append(str(row["id"]))
                batch_images.append(image)
                progress.update(1)
                if len(batch_images) >= batch_size:
                    flush()
            if limit and len(ids) + len(batch_ids) >= limit:
                break
        flush()
    if not embeddings:
        return ids, np.empty((0, 0), dtype=np.float32)
    return ids, l2_normalize(np.vstack(embeddings).astype(np.float32))


def load_qrels(path: Path, query_ids: set[str], corpus_ids: set[str]) -> dict[str, set[str]]:
    """! @brief 读取 qrels，并过滤到实际编码成功的 query/corpus。"""

    import pyarrow.parquet as pq

    qrels: dict[str, set[str]] = defaultdict(set)
    table = pq.read_table(path, columns=["query-id", "corpus-id", "score"])
    for row in table.to_pylist():
        if int(row.get("score") or 0) <= 0:
            continue
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        if query_id in query_ids and corpus_id in corpus_ids:
            qrels[query_id].add(corpus_id)
    return qrels


def top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """! @brief 取单条查询分数最高的若干 corpus 行号。"""

    if limit >= scores.shape[0]:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    return candidates[np.argsort(-scores[candidates])]


def average_precision_at_k(top_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """! @brief 计算单条查询的 AP@K。"""

    hits = 0
    precision_sum = 0.0
    for rank, corpus_id in enumerate(top_ids[:k], start=1):
        if corpus_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / max(min(len(relevant_ids), k), 1)


def reciprocal_rank(top_ids: list[str], relevant_ids: set[str]) -> float:
    """! @brief 计算首个相关结果的倒数排名。"""

    for rank, corpus_id in enumerate(top_ids, start=1):
        if corpus_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_embeddings: np.ndarray,
    qrels: dict[str, set[str]],
    top_ks: list[int],
) -> dict[str, Any]:
    """! @brief 按 qrels 计算图搜图检索指标。"""

    max_k = max(top_ks)
    corpus_row_by_id = {corpus_id: row_idx for row_idx, corpus_id in enumerate(corpus_ids)}
    sums: dict[str, float] = defaultdict(float)
    details: list[dict[str, Any]] = []
    evaluated = 0

    for start in tqdm(range(query_embeddings.shape[0]), desc="mteb eval", unit="query"):
        query_id = query_ids[start]
        relevant_ids = qrels.get(query_id)
        if not relevant_ids:
            continue
        scores = query_embeddings[start] @ corpus_embeddings.T
        self_id = query_id.replace("q", "d", 1)
        self_row = corpus_row_by_id.get(self_id)
        if self_row is not None and self_id not in relevant_ids:
            scores[self_row] = -np.inf
        top_rows = top_indices(scores, max_k)
        top_ids = [corpus_ids[int(row)] for row in top_rows]
        evaluated += 1
        for k in top_ks:
            hits = sum(1 for corpus_id in top_ids[:k] if corpus_id in relevant_ids)
            sums[f"hit@{k}"] += float(hits > 0)
            sums[f"precision@{k}"] += hits / k
            sums[f"recall@{k}"] += hits / len(relevant_ids)
            sums[f"map@{k}"] += average_precision_at_k(top_ids, relevant_ids, k)
        sums["mrr"] += reciprocal_rank(top_ids, relevant_ids)
        if len(details) < 5:
            details.append(
                {
                    "query_id": query_id,
                    "relevant_total": len(relevant_ids),
                    "top_ids": top_ids,
                    "hits_in_top": [corpus_id for corpus_id in top_ids if corpus_id in relevant_ids],
                }
            )

    metrics = {key: value / max(evaluated, 1) for key, value in sorted(sums.items())}
    return {"evaluated_queries": evaluated, "metrics": metrics, "examples": details}


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建 MTEB 图搜图 benchmark 参数解析器。"""

    parser = argparse.ArgumentParser(prog="evaluate_mteb_image_retrieval.py")
    parser.add_argument("--repo-id", default="mteb/cub200_retrieval")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--limit-corpus", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    args = build_parser().parse_args()
    top_ks = sorted({k for k in args.top_k if k > 0})
    if not top_ks:
        raise SystemExit("--top-k must contain at least one positive integer")

    query_path = download_hf_dataset_file(args.repo_id, "query-00000-of-00001.parquet", args.cache_dir)
    corpus_path = download_hf_dataset_file(args.repo_id, "corpus-00000-of-00001.parquet", args.cache_dir)
    qrels_path = download_hf_dataset_file(args.repo_id, "qrels-00000-of-00001.parquet", args.cache_dir)

    embedder = ClipEmbedder(args.model_name, args.device)
    corpus_ids, corpus_embeddings = encode_parquet_images(
        corpus_path,
        embedder,
        batch_size=args.batch_size,
        limit=args.limit_corpus,
    )
    query_ids, query_embeddings = encode_parquet_images(
        query_path,
        embedder,
        batch_size=args.batch_size,
        limit=args.limit_queries,
    )
    if corpus_embeddings.size == 0 or query_embeddings.size == 0:
        raise SystemExit("empty query or corpus embeddings")
    qrels = load_qrels(qrels_path, set(query_ids), set(corpus_ids))
    result = evaluate_rankings(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        qrels=qrels,
        top_ks=top_ks,
    )
    summary = {
        "dataset": args.repo_id,
        "task": "image-to-image-retrieval",
        "metric_note": "Uses official query/corpus/qrels parquet files; metadata/FTS is not used.",
        "model_name": args.model_name,
        "queries": len(query_ids),
        "corpus": len(corpus_ids),
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
