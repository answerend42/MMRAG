#!/usr/bin/env python
"""PicRetrieve 最小冒烟评测。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from app.embedder import ClipEmbedder
from app.index_store import IndexStore
from app.retrieval import Retriever

DEFAULT_QUERIES = [
    {"query": "AMD Ryzen CPU", "must_category": "cpu", "must_brand": "AMD"},
    {"query": "NVIDIA RTX graphics card", "must_category": "video-card"},
    {"query": "2TB SSD internal hard drive", "must_category": "internal-hard-drive"},
    {"query": "scanned email document", "must_category": "email"},
    {"query": "receipt with subtotal price", "must_category": "receipt"},
]


def ensure_queries(path: Path) -> None:
    """! @brief 查询文件缺失时写入默认弱监督 query。"""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in DEFAULT_QUERIES:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> list[dict[str, Any]]:
    """! @brief 读取 JSONL 查询文件。"""

    ensure_queries(path)
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def metadata_record(result: dict[str, Any]) -> dict[str, Any]:
    """! @brief 从检索结果里提取 bootstrap 写入的元信息记录。"""

    metadata = result.get("metadata") or {}
    return metadata.get("metadata_jsonl") or metadata.get("sidecar") or {}


def item_category(item: dict[str, Any]) -> str:
    """! @brief 提取结果或 item 的类别字段。"""

    metadata = item.get("metadata") or {}
    record = metadata_record(item) if "score" in item else metadata.get("metadata_jsonl", {})
    return str(record.get("category") or metadata.get("folder") or item.get("folder") or "")


def hit_by_field(results: list[dict[str, Any]], field: str, expected: str | None) -> bool:
    """! @brief 判断 top-k 中是否存在目标字段命中。"""

    if not expected:
        return True
    expected_lower = expected.lower()
    for result in results:
        record = metadata_record(result)
        value = str(record.get(field) or "")
        if value.lower() == expected_lower or expected_lower in value.lower():
            return True
    return False


def evaluate_text_queries(
    retriever: Retriever,
    queries: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """! @brief 评测文本到图片的 category/brand hit@k。"""

    details: list[dict[str, Any]] = []
    category_hits = 0
    brand_total = 0
    brand_hits = 0
    for query in queries:
        results = retriever.search_by_text(query["query"], top_k=top_k)
        category_hit = hit_by_field(results, "category", query.get("must_category"))
        brand_hit = hit_by_field(results, "brand", query.get("must_brand"))
        category_hits += int(category_hit)
        if query.get("must_brand"):
            brand_total += 1
            brand_hits += int(brand_hit)
        details.append(
            {
                "query": query["query"],
                "category_hit": category_hit,
                "brand_hit": brand_hit if query.get("must_brand") else None,
                "top": [result["rel_path"] for result in results[:3]],
            }
        )
    return {
        "category_hit@k": category_hits / max(len(queries), 1),
        "brand_hit@k": brand_hits / max(brand_total, 1),
        "details": details,
    }


def evaluate_image_queries(
    retriever: Retriever,
    store: IndexStore,
    probes_per_category: int,
) -> dict[str, Any]:
    """! @brief 从索引中抽图片做 image-to-image 同类命中冒烟评测。"""

    if probes_per_category <= 0:
        return {"same_category_hit@5": None, "same_category_hit@10": None, "details": []}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in store.list_items():
        category = item_category(item)
        if category:
            by_category[category].append(item)

    details: list[dict[str, Any]] = []
    hit5 = 0
    hit10 = 0
    total = 0
    for category, items in by_category.items():
        for item in items[:probes_per_category]:
            path = Path(item["path"])
            if not path.exists():
                continue
            with Image.open(path) as image:
                results = retriever.search_by_image(image.convert("RGB"), top_k=10)
            result_categories = [item_category(result) for result in results]
            same5 = category in result_categories[:5]
            same10 = category in result_categories[:10]
            hit5 += int(same5)
            hit10 += int(same10)
            total += 1
            details.append(
                {
                    "query": item["rel_path"],
                    "category": category,
                    "same_category_hit@5": same5,
                    "same_category_hit@10": same10,
                }
            )
    return {
        "same_category_hit@5": hit5 / max(total, 1),
        "same_category_hit@10": hit10 / max(total, 1),
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建冒烟评测参数解析器。"""

    parser = argparse.ArgumentParser(prog="evaluate_smoke.py")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/smoke_queries.jsonl"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--image-probes-per-category", type=int, default=3)
    parser.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    args = build_parser().parse_args()
    store = IndexStore(args.data_dir)
    embedder = ClipEmbedder(args.model_name, args.device)
    retriever = Retriever(store, embedder)
    queries = load_queries(args.queries)
    summary = {
        "text": evaluate_text_queries(retriever, queries, args.top_k),
        "image": evaluate_image_queries(retriever, store, args.image_probes_per_category),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
