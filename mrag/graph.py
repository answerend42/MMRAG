"""MMRAG LangGraph 状态图。

接收一个召回器注册表 (dict[Modality, Retriever])，
每种模态的召回器各自独立，注册表驱动分发。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from mrag.models.evidence import Modality
from mrag.nodes.answer_generator import answer_generator_node
from mrag.nodes.evidence_verifier import evidence_verifier_node
from mrag.nodes.modality_router import modality_router_node
from mrag.nodes.picretrieve_answer import picretrieve_answer_node
from mrag.nodes.picretrieve_planner import picretrieve_planner_node
from mrag.nodes.picretrieve_verifier import picretrieve_reflection_node, picretrieve_verifier_node
from mrag.nodes.query_planner import query_planner_node
from mrag.nodes.reflection import reflection_node
from mrag.nodes.reranker import reranker_node
from mrag.nodes.retrievers import retrievers_node
from mrag.retrievers.base import Retriever
from mrag.state import MMRAGState


def _need_retrieval(state: MMRAGState) -> Literal["modality_router", "answer_generator"]:
    if state.query_plan and not state.query_plan.need_retrieval:
        return "answer_generator"
    return "modality_router"


def _need_picretrieve_retrieval(
    state: MMRAGState,
) -> Literal["modality_router", "picretrieve_answer"]:
    if state.query_plan and not state.query_plan.need_retrieval:
        return "picretrieve_answer"
    return "modality_router"


def _check_confidence(state: MMRAGState) -> Literal["answer_generator", "reflection"]:
    return "reflection" if not state.is_confident else "answer_generator"


def _check_retry(state: MMRAGState) -> Literal["modality_router", "answer_generator"]:
    return "modality_router" if state.retry_count < state.max_retries else "answer_generator"


def _check_picretrieve_retry(
    state: MMRAGState,
) -> Literal["picretrieve_reflection", "picretrieve_answer"]:
    if not state.is_confident and state.retry_count < state.max_retries:
        return "picretrieve_reflection"
    return "picretrieve_answer"


def build_graph(registry: dict[Modality, Retriever] | None = None, llm=None) -> StateGraph:
    """构建 MMRAG 状态图。

    Args:
        registry: 模态→召回器映射。例如 {Modality.TEXT: TextRetriever(...)}。
        llm: LLMClient 实例（用于规划与生成）。

    Returns:
        编译后的图。
    """
    reg = registry or {}

    workflow = StateGraph(MMRAGState)

    workflow.add_node("query_planner", lambda s: query_planner_node(s, llm=llm))
    workflow.add_node("modality_router", modality_router_node)
    workflow.add_node("retrievers", lambda s: retrievers_node(s, registry=reg))
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("evidence_verifier", evidence_verifier_node)
    workflow.add_node("answer_generator", lambda s: answer_generator_node(s, llm=llm))
    workflow.add_node("reflection", reflection_node)

    workflow.set_entry_point("query_planner")

    workflow.add_conditional_edges(
        "query_planner",
        _need_retrieval,
        {"modality_router": "modality_router", "answer_generator": "answer_generator"},
    )
    workflow.add_edge("modality_router", "retrievers")
    workflow.add_edge("retrievers", "reranker")
    workflow.add_edge("reranker", "evidence_verifier")

    workflow.add_conditional_edges(
        "evidence_verifier",
        _check_confidence,
        {"answer_generator": "answer_generator", "reflection": "reflection"},
    )
    workflow.add_conditional_edges(
        "reflection",
        _check_retry,
        {"modality_router": "modality_router", "answer_generator": "answer_generator"},
    )
    workflow.add_edge("answer_generator", END)

    return workflow.compile()


def build_picretrieve_graph(
    registry: dict[Modality, Retriever] | None = None, llm=None
) -> StateGraph:
    """构建最小 PicRetrieve Agent 图。

    该图只启用图片召回模块，但沿用 QueryPlan、RetrievalPlan 和
    EvidenceCard 这些通用契约，便于后续逐个增加新模块。
    """
    reg = registry or {}

    workflow = StateGraph(MMRAGState)

    # This is the production path for the current prototype. The generic
    # build_graph above remains available for CLI experiments with extra modules.
    workflow.add_node("picretrieve_planner", lambda s: picretrieve_planner_node(s, llm=llm))
    workflow.add_node("modality_router", modality_router_node)
    workflow.add_node("picretrieve", lambda s: retrievers_node(s, registry=reg))
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("picretrieve_verifier", lambda s: picretrieve_verifier_node(s, llm=llm))
    workflow.add_node("picretrieve_reflection", picretrieve_reflection_node)
    workflow.add_node("picretrieve_answer", lambda s: picretrieve_answer_node(s, llm=llm))

    workflow.set_entry_point("picretrieve_planner")
    workflow.add_conditional_edges(
        "picretrieve_planner",
        _need_picretrieve_retrieval,
        {"modality_router": "modality_router", "picretrieve_answer": "picretrieve_answer"},
    )
    workflow.add_edge("modality_router", "picretrieve")
    workflow.add_edge("picretrieve", "reranker")
    workflow.add_edge("reranker", "picretrieve_verifier")
    workflow.add_conditional_edges(
        "picretrieve_verifier",
        _check_picretrieve_retry,
        {
            "picretrieve_reflection": "picretrieve_reflection",
            "picretrieve_answer": "picretrieve_answer",
        },
    )
    workflow.add_edge("picretrieve_reflection", "picretrieve")
    workflow.add_edge("picretrieve_answer", END)

    return workflow.compile()
