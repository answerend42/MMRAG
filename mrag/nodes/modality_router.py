"""QueryPlan -> RetrievalPlan router for the current PicRetrieve prototype."""

from __future__ import annotations

import logging
from typing import Any

from mrag.models.evidence import Modality
from mrag.models.retrieval import QueryPlan, RetrievalPlan, RetrievalTask
from mrag.state import MMRAGState

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
MAX_TOP_K = 50
MAX_INPUT_IMAGES = 3
TEXT_RETRIEVAL_MODES = {"text_only", "hybrid_text_image"}

MODALITY_TOP_K: dict[Modality, int] = {
    Modality.TEXT: 10,
    Modality.IMAGE: 10,
    Modality.PAGE: 5,
    Modality.DOCUMENT: 5,
    Modality.AUDIO: 5,
    Modality.VIDEO: 5,
    Modality.TABLE: 5,
    Modality.SQL: 3,
    Modality.KG: 5,
}


def _bounded_top_k(value: int | None) -> int:
    return max(1, min(int(value or DEFAULT_TOP_K), MAX_TOP_K))


def _allow_text_tasks(input_images: list[dict[str, Any]], retrieval_mode: str) -> bool:
    # Image reference prompts such as "按这张图找" should not silently create
    # parallel text retrieval; text is treated as instruction or metadata only.
    return not input_images or retrieval_mode in TEXT_RETRIEVAL_MODES


def _query_texts(state: MMRAGState, plan: QueryPlan, allow_text_tasks: bool) -> list[str]:
    if plan.sub_queries:
        return plan.sub_queries
    query = state.query.strip()
    return [query] if query and allow_text_tasks else []


def _route_modalities(
    query_plan: QueryPlan, queries: list[str], default_top_k: int | None = None
) -> list[RetrievalTask]:
    """根据 query plan 生成检索任务。"""
    tasks: list[RetrievalTask] = []

    modalities = query_plan.required_modalities or [Modality.TEXT, Modality.IMAGE]

    for modality in modalities:
        top_k = default_top_k or MODALITY_TOP_K.get(modality, DEFAULT_TOP_K)
        for sub_query in queries:
            tasks.append(RetrievalTask(modality=modality, subquery=sub_query, top_k=top_k))

    return tasks


def _image_tasks(state: MMRAGState, top_k: int) -> list[RetrievalTask]:
    metadata_query = state.query.strip()
    tasks: list[RetrievalTask] = []
    for index, image in enumerate((state.input_images or [])[:MAX_INPUT_IMAGES], start=1):
        tasks.append(
            RetrievalTask(
                modality=Modality.IMAGE,
                subquery=metadata_query or f"uploaded image {index}",
                top_k=top_k,
                metadata_filter={"metadata_query": metadata_query} if metadata_query else {},
                query_image=image,
            )
        )
    return tasks


def modality_router_node(state: MMRAGState) -> dict[str, Any]:
    """LangGraph 节点：模态路由。

    将 QueryPlan 转换为具体的 RetrievalPlan。

    Args:
        state: 当前图状态，需包含 query_plan。

    Returns:
        更新 state 的 dict，包含 retrieval_plan。
    """
    plan = state.query_plan
    if plan is None:
        fallback_plan = RetrievalPlan(
            query=state.query,
            tasks=[
                RetrievalTask(modality=Modality.TEXT, subquery=state.query, top_k=10),
                RetrievalTask(modality=Modality.IMAGE, subquery=state.query, top_k=10),
            ],
            explanation="No query plan available, using default text+image retrieval",
        )
        return {
            "retrieval_plan": fallback_plan,
            "trace": state.trace + ["modality_router: fallback to text+image"],
        }

    if not plan.need_retrieval:
        noop_plan = RetrievalPlan(
            query=state.query,
            tasks=[],
            budget=0,
            explanation="Query planner determined no retrieval needed",
        )
        return {
            "retrieval_plan": noop_plan,
            "trace": state.trace + ["modality_router: no retrieval needed"],
        }

    requested_top_k = _bounded_top_k(getattr(state, "top_k", DEFAULT_TOP_K))
    input_images = state.input_images or []
    retrieval_mode = (state.agent_plan or {}).get("retrieval_mode", "")
    allow_text_tasks = _allow_text_tasks(input_images, retrieval_mode)
    queries = _query_texts(state, plan, allow_text_tasks)
    tasks = (
        _route_modalities(plan, queries, default_top_k=requested_top_k)
        if queries and allow_text_tasks
        else []
    )
    tasks.extend(_image_tasks(state, requested_top_k))

    budget = len(tasks)

    modality_count = len(set(t.modality for t in tasks))
    retrieval_plan = RetrievalPlan(
        query=state.query,
        tasks=tasks,
        budget=budget,
        explanation=f"Routing to {len(tasks)} tasks across {modality_count} modalities",
    )

    logger.info(
        "Modality router: %d tasks, modalities=%s", len(tasks), [t.modality.value for t in tasks]
    )

    return {
        "retrieval_plan": retrieval_plan,
        "trace": state.trace + [f"modality_router: {len(tasks)} tasks"],
    }
