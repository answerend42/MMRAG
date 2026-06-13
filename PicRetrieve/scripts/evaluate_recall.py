#!/usr/bin/env python
"""PicRetrieve 弱监督召回率评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.embedder import ClipEmbedder
from app.index_store import IndexStore
from app.retrieval import Retriever

DEFAULT_QUERIES = [
    {
        "query": "2080",
        "must_category": "video-card",
        "must_text_all": ["2080"],
        "note": "短数字型号，检验 metadata 精确召回和融合。",
    },
    {
        "query": "RTX 2080",
        "must_category": "video-card",
        "must_text_all": ["RTX", "2080"],
    },
    {
        "query": "GeForce RTX 2080",
        "must_category": "video-card",
        "must_text_all": ["GeForce", "RTX", "2080"],
    },
    {
        "query": "Ryzen 7",
        "must_category": "cpu",
        "must_text_all": ["Ryzen", "7"],
    },
    {
        "query": "DDR5 4800",
        "must_category": "memory",
        "must_text_all": ["DDR5", "4800"],
    },
    {
        "query": "2 TB SSD",
        "must_category": "internal-hard-drive",
        "must_text_all": ["2 TB", "SSD"],
    },
    {
        "query": "invoice document",
        "must_category": "invoice",
        "must_text_all": ["invoice"],
    },
]


def ensure_queries(path: Path) -> None:
    """! @brief 查询文件缺失时写入默认型号/属性召回评测集。"""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in DEFAULT_QUERIES:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> list[dict[str, Any]]:
    """! @brief 读取 JSONL 评测查询。"""

    ensure_queries(path)
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def metadata_record(item: dict[str, Any]) -> dict[str, Any]:
    """! @brief 提取 bootstrap 写入的元信息记录。"""

    metadata = item.get("metadata") or {}
    return metadata.get("metadata_jsonl") or metadata.get("sidecar") or {}


def searchable_text(item: dict[str, Any]) -> str:
    """! @brief 生成用于判定弱监督相关性的文本。"""

    record = metadata_record(item)
    pieces = [
        item.get("rel_path", ""),
        item.get("metadata_text", ""),
        record.get("title", ""),
        record.get("brand", ""),
        record.get("category", ""),
        record.get("text", ""),
        json.dumps(record.get("metadata") or {}, ensure_ascii=False, default=str),
    ]
    return " ".join(str(piece) for piece in pieces if piece).lower()


def item_category(item: dict[str, Any]) -> str:
    """! @brief 获取 item 的归一化类别名。"""

    record = metadata_record(item)
    return str(record.get("category") or item.get("folder") or "").lower()


def is_relevant(item: dict[str, Any], query: dict[str, Any]) -> bool:
    """! @brief 按 JSONL 中的弱监督条件判断 item 是否相关。"""

    expected_category = str(query.get("must_category") or "").lower()
    if expected_category and item_category(item) != expected_category:
        return False

    text = searchable_text(item)
    for token in query.get("must_text_all") or []:
        if str(token).lower() not in text:
            return False

    any_tokens = [str(token).lower() for token in query.get("must_text_any") or []]
    return not any_tokens or any(token in text for token in any_tokens)


def reciprocal_rank(result_ids: list[int], relevant_ids: set[int]) -> float:
    """! @brief 计算首个相关结果的倒数排名。"""

    for rank, item_id in enumerate(result_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(
    retriever: Retriever,
    all_items: list[dict[str, Any]],
    query: dict[str, Any],
    top_k: int,
    candidate_k: int,
) -> dict[str, Any]:
    """! @brief 对单条查询计算 hit/precision/recall/MRR。"""

    relevant_ids = {int(item["id"]) for item in all_items if is_relevant(item, query)}
    results = retriever.search_by_text(
        query["query"],
        top_k=top_k,
        candidate_k=candidate_k,
        profile=query.get("profile", "general"),
        metadata_query=query.get("metadata_query"),
    )
    result_ids = [int(result["id"]) for result in results]
    hit_ids = [item_id for item_id in result_ids if item_id in relevant_ids]
    top = []
    for result in results[: min(top_k, 10)]:
        record = metadata_record(result)
        top.append(
            {
                "id": result["id"],
                "rel_path": result["rel_path"],
                "score": result["score"],
                "metadata_score": result["metadata_score"],
                "title": record.get("title", ""),
                "category": record.get("category", ""),
            }
        )
    return {
        "query": query["query"],
        "note": query.get("note", ""),
        "relevant_total": len(relevant_ids),
        "hit@1": bool(result_ids[:1] and result_ids[0] in relevant_ids),
        "hit@5": bool(set(result_ids[:5]) & relevant_ids),
        "hit@10": bool(set(result_ids[:10]) & relevant_ids),
        f"precision@{top_k}": len(hit_ids) / max(len(result_ids), 1),
        f"recall@{top_k}": len(hit_ids) / max(len(relevant_ids), 1),
        "mrr": reciprocal_rank(result_ids, relevant_ids),
        "top": top,
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    """! @brief 对详情行的数值字段求平均。"""

    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建召回评测参数解析器。"""

    parser = argparse.ArgumentParser(prog="evaluate_recall.py")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/recall_queries.jsonl"))
    parser.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=200)
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    args = build_parser().parse_args()
    store = IndexStore(args.data_dir)
    embedder = ClipEmbedder(args.model_name, args.device)
    retriever = Retriever(store, embedder)
    all_items = store.list_items()
    details = [
        evaluate_query(retriever, all_items, query, args.top_k, args.candidate_k)
        for query in load_queries(args.queries)
    ]
    evaluated = [row for row in details if row["relevant_total"] > 0]
    recall_key = f"recall@{args.top_k}"
    precision_key = f"precision@{args.top_k}"
    summary = {
        "queries": len(details),
        "evaluated_queries": len(evaluated),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "hit@1": average(evaluated, "hit@1"),
        "hit@5": average(evaluated, "hit@5"),
        "hit@10": average(evaluated, "hit@10"),
        precision_key: average(evaluated, precision_key),
        recall_key: average(evaluated, recall_key),
        "mrr": average(evaluated, "mrr"),
        "details": details,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
