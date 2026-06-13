"""Retrievers Node — 注册表驱动的多模态召回执行。

不再有 if/elif 分支。每种模态的召回器独立注册，
节点遍历 retrieval_plan.tasks 查表调用。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, cast

from mrag.models.evidence import EvidenceCard, Modality
from mrag.models.retrieval import RetrievalTask
from mrag.retrievers.base import Retriever
from mrag.state import MMRAGState

logger = logging.getLogger(__name__)


class ImageDataRetriever(Protocol):
    def search_by_image_data(
        self, image_data: dict[str, Any], top_k: int = 10, metadata_query: str | None = None
    ) -> list[EvidenceCard]: ...


def _run_task(task: RetrievalTask, retriever: Retriever) -> list[EvidenceCard]:
    if task.query_image is None:
        return retriever.search(task.subquery, top_k=task.top_k)

    # Image-to-image search is an optional extension on image retrievers; keeping
    # it duck-typed lets future retrievers adopt the same capability incrementally.
    if hasattr(retriever, "search_by_image_data"):
        image_retriever = cast(ImageDataRetriever, retriever)
        return image_retriever.search_by_image_data(
            task.query_image,
            top_k=task.top_k,
            metadata_query=task.metadata_filter.get("metadata_query"),
        )
    return retriever.search(task.subquery, top_k=task.top_k)


def retrievers_node(state: MMRAGState, registry: dict[Modality, Retriever] | None = None) -> dict:
    """遍历检索计划中的任务，用注册表分发到对应召回器。

    Args:
        state: 图状态，需包含 retrieval_plan.tasks。
        registry: 模态→召回器映射表。为 None 时从 state 获取
                  （兼容旧版闭包注入）。

    Returns:
        更新 evidence_cards。
    """
    plan = state.retrieval_plan
    if plan is None or not plan.tasks:
        return {"evidence_cards": [], "trace": state.trace + ["retrievers: no tasks"]}

    reg = registry or getattr(state, "_registry", None)
    if reg is None:
        logger.warning("检索器注册表为空，无法执行检索")
        return {"evidence_cards": [], "trace": state.trace + ["retrievers: no registry"]}

    all_cards: list[EvidenceCard] = []

    for task in plan.tasks:
        retriever = reg.get(task.modality)
        if retriever is None:
            logger.debug("模态 %s 无对应召回器，跳过", task.modality.value)
            continue
        if not retriever.is_ready:
            logger.debug("召回器 %s 未就绪，跳过", type(retriever).__name__)
            continue

        cards = _run_task(task, retriever)
        all_cards.extend(cards)
        logger.debug("%s: %d 条结果", type(retriever).__name__, len(cards))

    logger.info("检索总计: %d 条证据", len(all_cards))
    return {
        "evidence_cards": all_cards,
        "trace": state.trace + [f"retrievers: {len(all_cards)} cards from {len(plan.tasks)} tasks"],
    }
