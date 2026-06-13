"""PicRetrieve LLM verifier and retry planner."""

from __future__ import annotations

import json
import logging
from typing import Any

from mrag.models.evidence import Confidence, Modality
from mrag.models.retrieval import RetrievalPlan, RetrievalTask
from mrag.state import MMRAGState
from mrag.utils.json_utils import parse_json_object

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 MMRAG 的图片召回结果 Verifier。

你不能直接看图片，只能根据 PicRetrieve 返回的图片路径、分数、尺寸和 metadata 判断：
1. 当前召回是否足够可信，可以展示给用户；
2. 是否需要用更好的 query 再召回一次。

只返回紧凑 JSON，不要 markdown，不要解释文本。schema:
{
  "is_sufficient": true,
  "reason": "简短中文原因",
  "improved_queries": ["query for retry"],
  "confidence": 0.0
}

判断规则：
- 如果 top score >= 0.50，且路径或 metadata 没有明显冲突，一般认为足够。
- 如果结果分数低、query 太抽象、或路径明显偏离用户目标，给出 1-2 条更具体的英文 retry query。
- 不要为了完美主义反复重试；当前目标是快速原型展示。
"""


def picretrieve_verifier_node(state: MMRAGState, llm=None) -> dict[str, Any]:
    """Verify reranked image evidence with an LLM-assisted policy."""

    cards = state.reranked_cards
    if not cards:
        verification = {
            "mode": "fallback",
            "is_sufficient": False,
            "reason": "没有召回结果。",
            "improved_queries": [state.query],
            "confidence": 0.0,
        }
        return {
            "is_confident": False,
            "confidence_reason": verification["reason"],
            "agent_verification": verification,
            "trace": state.trace + ["picretrieve_verifier: no evidence"],
        }

    verification = _verify_with_llm(state, llm)
    is_sufficient = bool(verification.get("is_sufficient"))
    reason = str(verification.get("reason") or "")
    trace_item = (
        f"picretrieve_verifier: {verification.get('mode', 'fallback')} "
        f"{'pass' if is_sufficient else 'retry'}"
    )

    return {
        "is_confident": is_sufficient,
        "confidence_reason": reason,
        "agent_verification": verification,
        "trace": state.trace + [trace_item],
    }


def picretrieve_reflection_node(state: MMRAGState) -> dict[str, Any]:
    """Turn verifier feedback into a concrete retry RetrievalPlan."""

    retry_count = state.retry_count + 1
    verification = state.agent_verification or {}
    improved = _clean_queries(verification.get("improved_queries"), fallback=state.query, limit=2)

    tasks = [
        RetrievalTask(
            modality=Modality.IMAGE, subquery=query, top_k=max(1, min(int(state.top_k or 10), 50))
        )
        for query in improved
    ]
    plan = RetrievalPlan(
        query=state.query,
        tasks=tasks,
        budget=len(tasks),
        explanation=str(verification.get("reason") or "LLM verifier requested a retry"),
    )

    trace_item = f"picretrieve_reflection: retry #{retry_count} {len(tasks)} queries"
    return {
        "retrieval_plan": plan,
        "retry_count": retry_count,
        "retry_feedback": plan.explanation,
        "trace": state.trace + [trace_item],
    }


def _verify_with_llm(state: MMRAGState, llm) -> dict[str, Any]:
    if llm is None:
        return _fallback_verification(state, mode="fallback")

    try:
        response = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _verification_user_message(state)},
            ],
            temperature=0,
            max_tokens=700,
        )
        data = parse_json_object(response)
        is_sufficient = bool(data.get("is_sufficient"))
        improved_raw = data.get("improved_queries")
        return {
            "mode": "llm",
            "is_sufficient": is_sufficient,
            "reason": str(data.get("reason") or ""),
            "improved_queries": []
            if is_sufficient and not improved_raw
            else _clean_queries(improved_raw, fallback=state.query, limit=2),
            "confidence": _float_between(data.get("confidence"), default=0.7),
        }
    except Exception as exc:
        logger.warning("PicRetrieve verifier LLM failed: %s", exc)
        fallback = _fallback_verification(state, mode="fallback")
        fallback["reason"] = f"{fallback['reason']} LLM 验证失败：{exc}"
        return fallback


def _fallback_verification(state: MMRAGState, mode: str) -> dict[str, Any]:
    cards = state.reranked_cards
    top = cards[0] if cards else None
    high = sum(1 for card in cards if card.confidence == Confidence.HIGH)
    medium = sum(1 for card in cards if card.confidence == Confidence.MEDIUM)
    top_score = top.score if top else 0.0
    sufficient = top_score >= 0.5 or high >= 1 or medium >= 2
    reason = f"基于 PicRetrieve 分数降级判断：top={top_score:.3f}, high={high}, medium={medium}。"
    return {
        "mode": mode,
        "is_sufficient": sufficient,
        "reason": reason,
        "improved_queries": [state.query],
        "confidence": 0.55 if sufficient else 0.35,
    }


def _verification_user_message(state: MMRAGState) -> str:
    query_plan = state.query_plan
    planned_queries = query_plan.sub_queries if query_plan else [state.query]
    rows = []
    for card in state.reranked_cards[:8]:
        metadata = card.metadata or {}
        rows.append(
            {
                "id": card.source_id,
                "path": card.locator,
                "score": round(card.score, 4),
                "confidence": card.confidence.value,
                "visual_score": metadata.get("visual_score"),
                "metadata_score": metadata.get("metadata_score"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
            }
        )
    payload = {
        "original_query": state.query,
        "planner_queries": planned_queries,
        "retry_count": state.retry_count,
        "max_retries": state.max_retries,
        "top_results": rows,
    }
    return json.dumps(payload, ensure_ascii=False)


def _clean_queries(values: Any, fallback: str, limit: int) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = [fallback]
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        cleaned.append(text[:200])
        seen.add(key)
        if len(cleaned) >= limit:
            break
    return cleaned or [fallback]


def _float_between(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
