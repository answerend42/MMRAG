"""Reranker — 候选合并与重排。"""

from __future__ import annotations

import logging

from mrag.state import MMRAGState

logger = logging.getLogger(__name__)


def reranker_node(state: MMRAGState) -> dict:
    """去重并按分数降序排列。

    Args:
        state: 含 evidence_cards。

    Returns:
        更新 reranked_cards。
    """
    cards = state.evidence_cards
    if not cards:
        return {"reranked_cards": [], "trace": state.trace + ["reranker: empty"]}

    # 去重：相同 source_id + locator 保留最高分
    seen: dict[tuple[str, str], int] = {}
    deduped: list = []
    for c in cards:
        key = (c.source_id, c.locator)
        if key in seen:
            idx = seen[key]
            if c.score > deduped[idx].score:
                deduped[idx] = c
        else:
            seen[key] = len(deduped)
            deduped.append(c)

    deduped.sort(key=lambda c: -c.score)
    result = deduped[:30]

    by_modality = {}
    for c in result:
        by_modality[c.modality.value] = by_modality.get(c.modality.value, 0) + 1

    logger.info("Reranker: %d→%d cards %s", len(cards), len(result), by_modality)
    return {"reranked_cards": result, "trace": state.trace + [f"reranker: {len(result)} cards"]}
