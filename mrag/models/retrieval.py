"""检索计划和查询规划模型。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from mrag.models.evidence import Modality


class IntentType(StrEnum):
    """用户查询意图类型。"""

    FACTUAL = "factual"
    """事实型问答。"""

    COMPARISON = "comparison"
    """比较型。"""

    EXPLANATION = "explanation"
    """解释型。"""

    SUMMARIZATION = "summarization"
    """总结型。"""

    VISUAL_LOCALIZATION = "visual_localization"
    """视觉定位。"""

    MULTI_HOP = "multi_hop"
    """多跳推理。"""

    STRUCTURED_QUERY = "structured_query"
    """结构化查询（SQL/KG）。"""

    GENERATION_ONLY = "generation_only"
    """仅生成，不需要检索。"""


class RetrievalTask(BaseModel):
    """单个检索任务。"""

    modality: Modality
    """目标模态。"""

    subquery: str
    """子查询文本。"""

    top_k: int = 10
    """召回数量。"""

    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    """元信息过滤条件。"""

    query_image: dict[str, Any] | None = None
    """可选的查询图片，通常包含 data_url、mime_type、name 等字段。"""


class QueryPlan(BaseModel):
    """查询规划器的输出。"""

    intent: IntentType
    """识别出的意图。"""

    need_retrieval: bool = True
    """是否需要外部检索。"""

    sub_queries: list[str] = Field(default_factory=list)
    """拆解后的子查询列表。"""

    required_modalities: list[Modality] = Field(default_factory=list)
    """需要的模态。"""

    is_multi_hop: bool = False
    """是否多跳问题。"""

    max_hops: int = 1
    """最大跳数。"""

    confidence: float = 1.0
    """规划置信度。"""


class RetrievalPlan(BaseModel):
    """模态路由器输出——具体的检索计划。"""

    query: str
    """原始查询。"""

    tasks: list[RetrievalTask] = Field(default_factory=list)
    """要执行的检索任务列表。"""

    budget: int = 3
    """本轮检索预算（最多调用多少个检索器）。"""

    explanation: str = ""
    """路由决策说明。"""
