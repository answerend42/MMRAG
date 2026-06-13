#!/usr/bin/env python
"""Flickr30k 近重复图搜图检索评测。"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

from app.embedder import ClipEmbedder, l2_normalize
from app.index_store import IndexStore


def metadata_record(item: dict[str, Any]) -> dict[str, Any]:
    """! @brief 提取 bootstrap 写入的 metadata.jsonl 记录。"""

    metadata = item.get("metadata") or {}
    return metadata.get("metadata_jsonl") or {}


def is_flickr30k_item(item: dict[str, Any]) -> bool:
    """! @brief 判断索引条目是否来自 Flickr30k benchmark。"""

    record = metadata_record(item)
    payload = record.get("metadata") or {}
    return payload.get("benchmark") == "flickr30k"


def collect_items(store: IndexStore, ids: np.ndarray, limit_images: int) -> list[dict[str, Any]]:
    """! @brief 按向量行号收集可评测的 Flickr30k 图片条目。"""

    by_id = {int(item["id"]): item for item in store.list_items()}
    rows: list[dict[str, Any]] = []
    for row_idx, item_id in enumerate(ids):
        item = by_id.get(int(item_id))
        if item is None or not is_flickr30k_item(item):
            continue
        path = Path(item["path"])
        if not path.exists():
            continue
        rows.append(
            {
                "item_id": int(item_id),
                "row": row_idx,
                "path": path,
                "rel_path": item["rel_path"],
            }
        )
        if limit_images and len(rows) >= limit_images:
            break
    return rows


def center_crop(image: Image.Image, ratio: float) -> Image.Image:
    """! @brief 按比例执行中心裁剪。"""

    width, height = image.size
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    """! @brief 通过 JPEG 压缩再解码生成查询图。"""

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def transform_image(image: Image.Image, variant: str) -> Image.Image:
    """! @brief 生成确定性的图搜图查询变体。"""

    image = image.convert("RGB")
    if variant == "jpeg_q50":
        return jpeg_roundtrip(image, 50)
    if variant == "center_crop_80":
        return center_crop(image, 0.80).resize(image.size)
    if variant == "downscale_50":
        small_size = (max(1, image.width // 2), max(1, image.height // 2))
        return image.resize(small_size).resize(image.size)
    if variant == "brightness_90":
        return ImageEnhance.Brightness(image).enhance(0.90)
    raise ValueError(f"unknown variant: {variant}")


def top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """! @brief 取单条查询分数最高的若干图片行号。"""

    if limit >= scores.shape[0]:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    return candidates[np.argsort(-scores[candidates])]


def update_metrics(
    row_scores: np.ndarray,
    positive_row: int,
    top_ks: list[int],
    counters: dict[str, Any],
) -> int:
    """! @brief 根据单条查询分数更新 Recall/MRR 计数器。"""

    positive_score = row_scores[positive_row]
    rank = int(np.count_nonzero(row_scores > positive_score) + 1)
    counters["ranks"].append(rank)
    counters["reciprocal_ranks"].append(1.0 / rank)
    for k in top_ks:
        if rank <= k:
            counters["hits"][k] += 1
    return rank


def summarize_counters(counters: dict[str, Any], top_ks: list[int]) -> dict[str, float]:
    """! @brief 把命中计数器转换成指标字典。"""

    total = max(len(counters["ranks"]), 1)
    metrics = {f"recall@{k}": counters["hits"][k] / total for k in top_ks}
    metrics["mrr"] = sum(counters["reciprocal_ranks"]) / total
    metrics["median_rank"] = float(statistics.median(counters["ranks"])) if counters["ranks"] else 0.0
    metrics["mean_rank"] = sum(counters["ranks"]) / total if counters["ranks"] else 0.0
    return metrics


def new_counters(top_ks: list[int]) -> dict[str, Any]:
    """! @brief 创建指标累加器。"""

    return {
        "hits": {k: 0 for k in top_ks},
        "ranks": [],
        "reciprocal_ranks": [],
    }


def evaluate(
    embedder: ClipEmbedder,
    image_embeddings: np.ndarray,
    items: list[dict[str, Any]],
    variants: list[str],
    batch_size: int,
    top_ks: list[int],
) -> dict[str, Any]:
    """! @brief 对 Flickr30k 图片变体执行 image-to-image 检索评测。"""

    max_k = max(top_ks)
    overall = new_counters(top_ks)
    by_variant = defaultdict(lambda: new_counters(top_ks))
    examples: list[dict[str, Any]] = []
    batch_images: list[Image.Image] = []
    batch_meta: list[dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        query_embeddings = embedder.encode_images(batch_images, batch_size=batch_size)
        scores = query_embeddings @ image_embeddings.T
        for local_idx, meta in enumerate(batch_meta):
            row_scores = scores[local_idx]
            rank = update_metrics(row_scores, meta["positive_row"], top_ks, overall)
            update_metrics(row_scores, meta["positive_row"], top_ks, by_variant[meta["variant"]])
            if len(examples) < 5:
                examples.append(
                    {
                        "query": meta["rel_path"],
                        "variant": meta["variant"],
                        "positive_row": int(meta["positive_row"]),
                        "rank": rank,
                        "top_rows": [int(row) for row in top_indices(row_scores, max_k).tolist()],
                    }
                )
        batch_images.clear()
        batch_meta.clear()

    total = len(items) * len(variants)
    with tqdm(total=total, desc="image benchmark", unit="query") as progress:
        for item in items:
            with Image.open(item["path"]) as base_image:
                base_image = base_image.convert("RGB")
                for variant in variants:
                    batch_images.append(transform_image(base_image, variant))
                    batch_meta.append(
                        {
                            "rel_path": item["rel_path"],
                            "positive_row": int(item["row"]),
                            "variant": variant,
                        }
                    )
                    progress.update(1)
                    if len(batch_images) >= batch_size:
                        flush_batch()
        flush_batch()

    return {
        "metrics": summarize_counters(overall, top_ks),
        "by_variant": {
            variant: summarize_counters(counters, top_ks)
            for variant, counters in sorted(by_variant.items())
        },
        "examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建图搜图 benchmark 参数解析器。"""

    parser = argparse.ArgumentParser(prog="evaluate_image_benchmark.py")
    parser.add_argument("--data-dir", type=Path, default=Path("data/flickr30k_index"))
    parser.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["jpeg_q50", "center_crop_80", "downscale_50", "brightness_90"],
    )
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

    items = collect_items(store, ids, args.limit_images)
    if not items:
        raise SystemExit("no Flickr30k benchmark images found in this index")

    embedder = ClipEmbedder(args.model_name, args.device)
    result = evaluate(
        embedder=embedder,
        image_embeddings=image_embeddings,
        items=items,
        variants=args.variants,
        batch_size=args.batch_size,
        top_ks=top_ks,
    )
    summary = {
        "dataset": "clip-benchmark/wds_flickr30k",
        "task": "image-to-image-near-duplicate-retrieval",
        "metric_note": "Queries are deterministic image transformations; metadata/FTS is not used.",
        "data_dir": str(args.data_dir),
        "model_name": args.model_name,
        "images": len(items),
        "queries": len(items) * len(args.variants),
        "variants": args.variants,
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
