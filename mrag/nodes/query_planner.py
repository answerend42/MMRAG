"""Query Planner — 查询规划节点。

职责：
1. 分析用户查询意图（事实型、比较型、多跳、仅生成等）。
2. 判断是否需要外部检索。
3. 拆解子查询。
4. 确定所需模态。

输出统一为 QueryPlan，供 Modality Router 使用。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mrag.models.evidence import Modality
from mrag.models.retrieval import IntentType, QueryPlan
from mrag.state import MMRAGState
from mrag.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# 意图识别提示模板
PLANNER_SYSTEM_PROMPT = (
    "你是一个多模态 RAG 系统的查询规划器。你的任务是分析用户查询"
    "并生成结构化的检索计划。\n"
    "\n"
    "你必须输出 JSON 格式，包含以下字段：\n"
    '- "intent": 查询意图，可选值："factual", "comparison", "explanation", '
    '"summarization", "visual_localization", "multi_hop", '
    '"structured_query", "generation_only"\n'
    '- "need_retrieval": 布尔值，是否需要外部检索\n'
    '- "sub_queries": 字符串列表，拆解后的子查询'
    "（如果不需要检索则为空列表）\n"
    '- "required_modalities": 字符串列表，需要的模态，可选值：'
    '"text", "image", "page", "document", "audio", '
    '"video", "table", "sql", "kg"\n'
    '- "is_multi_hop": 布尔值，是否多跳问题\n'
    '- "max_hops": 整数，最大跳数\n'
    '- "confidence": 0-1 浮点数，你对这个规划的置信度\n'
    "\n"
    "规则：\n"
    "- 如果查询是关于视觉内容（图片、图表、截图等），"
    '必须包含 "image" 模态。\n'
    "- 如果是纯知识问答不需要外部证据，need_retrieval 设为 false。\n"
    "- 复杂问题拆解为 2-4 个子查询。\n"
    "- 默认 max_hops 为 1，多跳问题设为 2-3。"
)


def create_planner_system_prompt() -> str:
    """生成 query planner 的系统提示。"""
    return PLANNER_SYSTEM_PROMPT


def create_planner_user_message(query: str) -> list[dict[str, Any]]:
    """构造 planner 的用户消息。"""
    return [
        {"role": "system", "content": create_planner_system_prompt()},
        {"role": "user", "content": f"请分析以下查询并输出规划 JSON：\n\n{query}"},
    ]


def query_planner_node(state: MMRAGState, llm: LLMClient | None = None) -> dict[str, Any]:
    """LangGraph 节点：查询规划。

    使用 LLM 分析查询意图并生成 QueryPlan。

    Args:
        state: 当前图状态。
        llm: LLM 客户端，默认从环境变量创建。

    Returns:
        更新 state 的 dict。
    """
    query = state.query
    if not query.strip():
        return {
            "query_plan": QueryPlan(
                intent=IntentType.GENERATION_ONLY, need_retrieval=False, confidence=1.0
            ),
            "trace": state.trace + ["query_planner: empty query, skip retrieval"],
        }

    if llm is None:
        llm = LLMClient()

    messages = create_planner_user_message(query)
    try:
        response = llm.chat(messages)
        # 解析 JSON
        data = _parse_llm_json(response)
        plan = QueryPlan(
            intent=IntentType(data.get("intent", "factual")),
            need_retrieval=data.get("need_retrieval", True),
            sub_queries=data.get("sub_queries", []),
            required_modalities=[Modality(m) for m in data.get("required_modalities", []) if m],
            is_multi_hop=data.get("is_multi_hop", False),
            max_hops=data.get("max_hops", 1),
            confidence=data.get("confidence", 0.8),
        )
        logger.info(
            "Query plan: intent=%s need_retrieval=%s sub_queries=%d",
            plan.intent.value,
            plan.need_retrieval,
            len(plan.sub_queries),
        )
    except Exception as exc:
        logger.warning("Query planner LLM failed, using default plan: %s", exc)
        # 降级：默认走文本检索
        plan = QueryPlan(
            intent=IntentType.FACTUAL,
            need_retrieval=True,
            sub_queries=[query],
            required_modalities=[Modality.TEXT, Modality.IMAGE],
            confidence=0.5,
        )

    return {"query_plan": plan, "trace": state.trace + [f"query_planner: {plan.intent.value}"]}


def _parse_llm_json(response: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON。兼容 markdown 代码块包裹。"""
    # 去掉 ```json ... ``` 包裹
    text = response.strip()
    if text.startswith("```"):
        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)
