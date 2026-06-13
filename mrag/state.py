"""LangGraph 状态定义。

保存整个多模态 RAG 流程的生命周期状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph import add_messages

from mrag.models.evidence import EvidenceCard
from mrag.models.retrieval import QueryPlan, RetrievalPlan


@dataclass
class MMRAGState:
    """多模态 RAG 图的完整状态。"""

    # ── 输入 ──
    query: str = ""
    """用户原始查询。"""

    input_images: list[dict[str, Any]] = field(default_factory=list)
    """用户上传的查询图片，支持与文本 query 一起进入多模态 planner。"""

    top_k: int = 10
    """单个召回模块的默认返回数量。"""

    active_modules: list[str] = field(default_factory=lambda: ["picretrieve"])
    """当前原型启用的模块 ID，后续可扩展到 text/table/kg 等。"""

    llm_model: str = ""
    """当前 Agent 使用的语言模型名称。"""

    # ── 规划 ──
    query_plan: QueryPlan | None = None
    """查询规划器输出。"""

    agent_plan: dict[str, Any] = field(default_factory=dict)
    """LLM Planner 的可展示中间结果。"""

    retrieval_plan: RetrievalPlan | None = None
    """模态路由器输出。"""

    # ── 检索结果 ──
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    """所有检索器产出的证据卡片。"""

    # ── 重排 ──
    reranked_cards: list[EvidenceCard] = field(default_factory=list)
    """重排后的最终证据列表。"""

    # ── 验证 ──
    is_confident: bool = False
    """验证器是否认为证据足够且一致。"""

    confidence_reason: str = ""
    """验证器输出（置信/不置信的原因）。"""

    agent_verification: dict[str, Any] = field(default_factory=dict)
    """LLM Verifier 的可展示中间结果。"""

    # ── 生成 ──
    answer: str = ""
    """最终生成的回答。"""

    # ── 重试 ──
    retry_count: int = 0
    """当前重试次数。"""

    max_retries: int = 3
    """最大重试次数。"""

    retry_feedback: str = ""
    """重试反馈（反思节点的输出）。"""

    # ── 消息日志（可选，用于 LLM 对话） ──
    messages: Annotated[list[dict[str, Any]], add_messages] = field(default_factory=list)
    """消息历史，langgraph add_messages reducer 自动追加。"""

    # ── 调试信息 ──
    errors: list[str] = field(default_factory=list)
    """流程中遇到的错误。"""

    trace: list[str] = field(default_factory=list)
    """节点执行轨迹，用于调试。"""
