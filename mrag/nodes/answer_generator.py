"""Answer Generator — 答案生成节点。

基于检索到的证据，调用 LLM 生成带引用的回答。
如果 LLM 不可用，给出清晰提示而非原始证据倾倒。
"""

from __future__ import annotations

import logging

from mrag.models.evidence import EvidenceCard, Modality
from mrag.state import MMRAGState
from mrag.utils.llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个严格基于证据的问答助手。

规则：
1. 只使用下方"证据"中的信息回答问题。
2. 每个关键结论后用 [来源: <source_id>] 标注。
3. 证据不足以回答时，明确说"证据不足以回答"。
4. 不要添加证据中没有的信息。
5. 用中文回答。"""


def answer_generator_node(state: MMRAGState, llm: LLMClient | None = None) -> dict:
    """基于证据生成回答。

    Args:
        state: 当前图状态，含 reranked_cards 和 query。
        llm: LLM 客户端。为 None 时尝试从环境创建。

    Returns:
        更新 answer。
    """
    query = state.query
    cards = state.reranked_cards

    if not cards:
        return {
            "answer": "未检索到相关证据，无法回答该问题。",
            "trace": state.trace + ["answer_generator: no evidence"],
        }

    if llm is None:
        try:
            llm = LLMClient()
        except Exception:
            llm = None

    evidence_text = _format_evidence(cards)

    if llm is not None:
        try:
            answer = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"## 问题\n{query}\n\n## 证据\n{evidence_text}\n\n"
                        f"请基于证据回答问题，并标注来源。",
                    },
                ],
                temperature=0.1,
            )
            return {
                "answer": answer,
                "trace": state.trace + [f"answer_generator: {len(answer)} chars (LLM)"],
            }
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            # 降级为简单的结构化输出
            pass

    # LLM 不可用时的降级：简洁格式
    top = cards[:5]
    lines = [f"关于「{query}」检索到以下信息："]
    for c in top:
        icon = "📄" if c.modality == Modality.TEXT else "🖼️"
        snippet = c.content_ref[:120].replace("\n", " ")
        score_str = f" (相关度: {c.score:.2f})" if c.score > 0 else ""
        lines.append(f"  {icon} [{c.source_id}] {snippet}{score_str}")
    lines.append("\n⚠️ 提示：设置 MRAG_LLM_API_KEY 后系统会自动生成结构化答案。")

    return {
        "answer": "\n".join(lines),
        "trace": state.trace + [f"answer_generator: {len(cards)} cards (fallback)"],
    }


def _format_evidence(cards: list[EvidenceCard], max_cards: int = 10) -> str:
    """格式化证据供 LLM 使用。"""
    parts = []
    for i, c in enumerate(cards[:max_cards]):
        parts.append(
            f"[{i + 1}] 模态: {c.modality.value} | ID: {c.source_id} | 位置: {c.locator}\n"
            f"    内容: {c.content_ref[:300]}"
        )
    return "\n\n".join(parts)
