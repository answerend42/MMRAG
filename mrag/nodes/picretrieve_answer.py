"""PicRetrieve LLM answer node."""

from __future__ import annotations

import logging

from mrag.state import MMRAGState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 MMRAG 的图片召回 Agent。

你负责把 LangGraph 的规划、召回和验证结果总结给用户。
注意：你不能直接看图片，只能基于 PicRetrieve 的路径、分数、metadata 和 Verifier 结论表述。
不要臆测品种、人物、地点或细节；除非路径或 metadata 明确出现，否则只说“候选图片”。
用中文，简洁但要让用户看出 Agent 做了哪些步骤。
"""


def picretrieve_answer_node(state: MMRAGState, llm=None) -> dict:
    """Generate the final answer, using the local LLM when available."""

    cards = state.reranked_cards
    query = state.query.strip() or "上传图片"
    if not cards:
        return {
            "answer": f"没有为「{query}」召回到图片结果。",
            "trace": state.trace + ["picretrieve_answer: no result"],
        }

    if llm is not None:
        try:
            answer = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _answer_user_message(state)},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            return {
                "answer": answer.strip(),
                "trace": state.trace + ["picretrieve_answer: llm summary"],
            }
        except Exception as exc:
            logger.warning("PicRetrieve answer LLM failed: %s", exc)

    top = cards[0]
    answer = (
        f"已为「{query}」召回 {len(cards)} 张候选图片。"
        f"最高相关度为 {top.score:.3f}，结果来自 PicRetrieve 图片索引。"
    )
    return {"answer": answer, "trace": state.trace + ["picretrieve_answer: summary"]}


def _answer_user_message(state: MMRAGState) -> str:
    plan = state.agent_plan or {}
    verification = state.agent_verification or {}
    rows = []
    for card in state.reranked_cards[:5]:
        metadata = card.metadata or {}
        rows.append(
            {
                "id": card.source_id,
                "path": card.locator,
                "score": round(card.score, 4),
                "visual_score": metadata.get("visual_score"),
                "metadata_score": metadata.get("metadata_score"),
            }
        )
    return (
        f"用户查询：{state.query or '（未提供文字，使用上传图片）'}\n"
        f"上传图片数量：{len(state.input_images or [])}\n"
        f"LLM Planner：{plan}\n"
        f"LLM Verifier：{verification}\n"
        f"重试次数：{state.retry_count}\n"
        f"候选图片总数：{len(state.reranked_cards)}\n"
        f"Top results sample：{rows}\n"
        "请输出 2-3 句完整中文总结：说明改写了什么、召回质量如何、用户可以看哪些结果。"
        "不要把 Top results sample 当成全部数量。不要补充样例里没有明确给出的视觉细节。"
    )
