"""Reflection — 反思与重试节点。

职责：
1. 分析为什么证据置信度不足。
2. 生成改进的检索计划。
3. 控制重试次数，防止无限循环。

策略（按重试次数递增）：
- 第1次：尝试增加缺失的模态。
- 第2次：扩大 top_k 召回更多候选。
- 第3次：使用不同子查询改写。
- 超过上限时终止并走生成。
"""

from __future__ import annotations

import logging

from mrag.models.evidence import Modality
from mrag.models.retrieval import RetrievalPlan, RetrievalTask
from mrag.state import MMRAGState

logger = logging.getLogger(__name__)


def reflection_node(state: MMRAGState) -> dict[str, any]:
    """LangGraph 节点：反思与重试规划。

    Args:
        state: 当前图状态。

    Returns:
        更新 state 的 dict，包含更新的 retrieval_plan 和 retry_count。
    """
    retry_count = state.retry_count + 1
    max_retries = state.max_retries

    if retry_count > max_retries:
        logger.info("Max retries (%d) reached, stopping", max_retries)
        return {
            "retry_count": retry_count,
            "retry_feedback": "已达最大重试次数，终止检索",
            "trace": state.trace + [f"reflection: max retries ({max_retries}) reached"],
        }

    cards = state.evidence_cards
    existing_modalities = set(c.modality for c in cards)

    plan = _build_retry_plan(state.query, existing_modalities, retry_count, max_retries)

    return {
        "retrieval_plan": plan,
        "retry_count": retry_count,
        "retry_feedback": plan.explanation,
        "trace": state.trace + [f"reflection: retry #{retry_count} — {plan.explanation[:60]}"],
    }


def _build_retry_plan(
    query: str, existing_modalities: set[Modality], retry_count: int, max_retries: int
) -> RetrievalPlan:
    """根据重试次数生成不同的检索策略。"""
    if retry_count == 1:
        # 策略1：尝试缺失的模态
        tasks: list[RetrievalTask] = []
        feedback: list[str] = []
        if Modality.TEXT not in existing_modalities:
            tasks.append(RetrievalTask(modality=Modality.TEXT, subquery=query, top_k=15))
            feedback.append("增加文本检索")
        if Modality.IMAGE not in existing_modalities:
            tasks.append(RetrievalTask(modality=Modality.IMAGE, subquery=query, top_k=10))
            feedback.append("增加图片检索")
        if not tasks:
            # 所有模态都有了，扩大召回
            tasks.append(RetrievalTask(modality=Modality.TEXT, subquery=query, top_k=30))
            tasks.append(RetrievalTask(modality=Modality.IMAGE, subquery=query, top_k=20))
            feedback.append("扩大召回数量")
        plan = RetrievalPlan(
            query=query, tasks=tasks, budget=len(tasks), explanation="; ".join(feedback)
        )
        return plan

    elif retry_count == 2:
        # 策略2：加大 top_k 并改写查询
        expanded = f"{query} 详细描述说明"
        return RetrievalPlan(
            query=query,
            tasks=[
                RetrievalTask(modality=Modality.TEXT, subquery=expanded, top_k=30),
                RetrievalTask(modality=Modality.IMAGE, subquery=expanded, top_k=20),
            ],
            budget=2,
            explanation="扩大召回并改写查询",
        )

    else:
        # 策略3：用更宽松的查询再次检索
        simpler = query.split("？")[0].split("?")[0] if any(c in query for c in "？?") else query
        return RetrievalPlan(
            query=query,
            tasks=[
                RetrievalTask(modality=Modality.TEXT, subquery=simpler, top_k=20),
                RetrievalTask(modality=Modality.IMAGE, subquery=query, top_k=15),
            ],
            budget=2,
            explanation="使用简化查询重试",
        )
