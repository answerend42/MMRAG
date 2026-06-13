"""Evidence Verifier — 证据验证节点。

CLIP 余弦相似度在 [0, 1] 范围（已归一化），
通常 >0.5 为相关，>0.6 为强相关。
"""

from __future__ import annotations

import logging

from mrag.models.evidence import Confidence
from mrag.state import MMRAGState

logger = logging.getLogger(__name__)


def evidence_verifier_node(state: MMRAGState) -> dict:
    """验证检索到的证据是否足够支撑答案生成。

    Args:
        state: 含 reranked_cards。

    Returns:
        更新 is_confident 和 confidence_reason。
    """
    cards = state.reranked_cards
    if not cards:
        return {
            "is_confident": False,
            "confidence_reason": "无检索结果",
            "trace": state.trace + ["verifier: no evidence"],
        }

    high = sum(1 for c in cards if c.confidence == Confidence.HIGH)
    medium = sum(1 for c in cards if c.confidence == Confidence.MEDIUM)
    low = sum(1 for c in cards if c.confidence == Confidence.LOW)
    avg_score = sum(c.score for c in cards) / len(cards)

    # CLIP 分数 > 0.5 通常表示语义相关
    relevant = sum(1 for c in cards if c.score > 0.5)

    if high >= 1 or (medium >= 2 and avg_score > 0.45):
        is_confident = True
        reason = f"证据可信: high={high} medium={medium} avg={avg_score:.3f}"
    elif relevant >= 1:
        is_confident = True
        reason = f"有相关结果: {relevant}/{len(cards)} 条相关"
    else:
        is_confident = False
        reason = f"置信不足: high={high} medium={medium} low={low} avg={avg_score:.3f}"

    logger.info("Verifier: %s — %s", "✅" if is_confident else "⚠️", reason)
    return {
        "is_confident": is_confident,
        "confidence_reason": reason,
        "trace": state.trace + [f"verifier: {'pass' if is_confident else 'retry'}"],
    }
