"""PicRetrieve-first LLM planner node."""

from __future__ import annotations

import logging
from typing import Any

from mrag.models.evidence import Modality
from mrag.models.retrieval import IntentType, QueryPlan
from mrag.state import MMRAGState
from mrag.utils.json_utils import parse_json_object

logger = logging.getLogger(__name__)

TEXT_RETRIEVAL_MODES = {"text_only", "hybrid_text_image"}
IMAGE_RETRIEVAL_MODES = {"image_direct", "image_guided", "hybrid_text_image"}
MAX_INPUT_IMAGES = 3
IMAGE_REFERENCE_MARKERS = [
    "按图",
    "按照图片",
    "按照对应的图片",
    "根据图片",
    "对应的图片",
    "这张图",
    "这张图片",
    "以图",
    "相似图片",
    "类似这张",
    "similar to this",
    "like this",
    "this image",
    "reference image",
]


SYSTEM_PROMPT = """你是 MMRAG 的图片召回 Agent Planner。

当前系统只有一个真实模块：PicRetrieve 图片召回，能力是用 CLIP 文本到图片检索。
你的任务是把用户输入改写成更适合图片召回的 1-3 个视觉检索 query。
用户输入可能包含文字，也可能包含上传图片，或二者同时存在。

只返回紧凑 JSON，不要 markdown，不要解释文本。schema:
{
  "intent": "visual_localization",
  "retrieval_mode": "text_only | image_direct | image_guided | hybrid_text_image",
  "rewritten_queries": ["query 1", "query 2"],
  "confidence": 0.0
}

改写要求：
- 保留用户真正要找的视觉对象、颜色、场景、动作、文本线索。
- 如果用户上传了图片，结合图片内容和文字要求生成可搜索的英文视觉描述。
- 如果只有图片没有文字，描述图片中最主要、最可检索的对象/场景。
- 如果文字只是“按这张图找 / 查找对应图片 / similar to this”等指令，不要强行并行文本检索；
  retrieval_mode 应为 image_direct 或 image_guided，rewritten_queries 可为空。
- 只有当文字本身是独立检索目标，且确实需要额外文本搜图时，才使用 hybrid_text_image。
- 对英文图片索引，优先生成英文 query；中文输入可翻译成英文视觉描述。
- 不要生成抽象空话，比如 high quality、beautiful、relevant image。
- 最多 3 条，避免互相重复。
"""

PLANNER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "picretrieve_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": ["visual_localization"]},
                "retrieval_mode": {
                    "type": "string",
                    "enum": [
                        "text_only",
                        "image_direct",
                        "image_guided",
                        "hybrid_text_image",
                    ],
                },
                "rewritten_queries": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 160},
                    "maxItems": 3,
                },
                "confidence": {"type": "number", "minimum": 0.5, "maximum": 1},
            },
            "required": [
                "intent",
                "retrieval_mode",
                "rewritten_queries",
                "confidence",
            ],
        },
    },
}

PLANNER_JSON_OBJECT_FORMAT = {"type": "json_object"}


def picretrieve_planner_node(state: MMRAGState, llm=None) -> dict:
    """Create an image-retrieval QueryPlan, using LLM query rewriting when available."""

    query = state.query.strip()
    input_images = state.input_images or []
    if not query and not input_images:
        plan = QueryPlan(
            intent=IntentType.GENERATION_ONLY,
            need_retrieval=False,
            sub_queries=[],
            required_modalities=[],
            confidence=1.0,
        )
        return {
            "query_plan": plan,
            "agent_plan": {"mode": "empty", "rewritten_queries": []},
            "trace": state.trace + ["picretrieve_planner: empty query"],
        }

    fallback_query = query or "uploaded reference image"
    llm_plan = _plan_with_llm(query, input_images, llm)
    retrieval_mode = _resolve_retrieval_mode(query, input_images, llm_plan)
    raw_queries = llm_plan.get("rewritten_queries") or (
        [] if input_images and not query else [fallback_query]
    )
    suggested_queries = _clean_queries(raw_queries, fallback=fallback_query) if raw_queries else []
    rewritten_queries = (
        suggested_queries if retrieval_mode in TEXT_RETRIEVAL_MODES else []
    )

    plan = QueryPlan(
        intent=IntentType.VISUAL_LOCALIZATION,
        need_retrieval=True,
        sub_queries=rewritten_queries,
        required_modalities=[Modality.IMAGE],
        is_multi_hop=False,
        max_hops=1,
        confidence=_planner_confidence(llm_plan.get("confidence"), default=0.75),
    )

    trace_item = (
        f"picretrieve_planner: {llm_plan.get('mode', 'fallback')} {len(rewritten_queries)} queries"
    )
    return {
        "query_plan": plan,
        "agent_plan": {
            "mode": llm_plan.get("mode", "fallback"),
            "intent": plan.intent.value,
            "rewritten_queries": rewritten_queries,
            "suggested_queries": suggested_queries,
            "rationale": _planner_rationale(
                mode=llm_plan.get("mode", "fallback"),
                retrieval_mode=retrieval_mode,
                input_images=len(input_images),
                rewritten_queries=rewritten_queries,
                suggested_queries=suggested_queries,
            ),
            "confidence": plan.confidence,
            "input_images": len(input_images),
            "retrieval_mode": retrieval_mode,
        },
        "trace": state.trace + [trace_item],
    }


def _plan_with_llm(query: str, input_images: list[dict[str, Any]], llm) -> dict[str, Any]:
    fallback_query = query or "uploaded reference image"
    if llm is None:
        return {
            "mode": "fallback",
            "rewritten_queries": [] if input_images and not query else [fallback_query],
            "confidence": 0.55,
            "retrieval_mode": "image_guided" if input_images else "text_only",
        }

    try:
        user_content = _planner_user_content(query, input_images)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = _chat_planner_json(llm, messages)
        data = parse_json_object(response)
        return {
            "mode": "llm",
            "rewritten_queries": data.get("rewritten_queries") or [fallback_query],
            "confidence": _float_between(data.get("confidence"), default=0.75),
            "retrieval_mode": str(data.get("retrieval_mode") or ""),
        }
    except Exception as exc:
        logger.warning("PicRetrieve planner LLM failed: %s", exc)
        return {
            "mode": "fallback",
            "rewritten_queries": [] if input_images and not query else [fallback_query],
            "planner_error": str(exc),
            "confidence": 0.5,
            "retrieval_mode": "image_guided" if input_images else "text_only",
        }


def _chat_planner_json(llm, messages: list[dict[str, Any]]) -> str:
    try:
        return llm.chat(
            messages,
            temperature=0,
            max_tokens=450,
            response_format=PLANNER_RESPONSE_FORMAT,
        )
    except Exception as exc:
        if not _is_response_format_unsupported(exc):
            raise
        logger.warning("Planner json_schema output mode unsupported, retrying json_object: %s", exc)
        return llm.chat(
            messages,
            temperature=0,
            max_tokens=450,
            response_format=PLANNER_JSON_OBJECT_FORMAT,
        )


def _is_response_format_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = [
        "response_format",
        "json_schema",
        "unsupported",
        "not supported",
        "invalid request",
        "400",
    ]
    return ("response_format" in message or "json_schema" in message) and any(
        marker in message for marker in markers
    )


def _planner_user_content(query: str, input_images: list[dict[str, Any]]) -> Any:
    text = (
        f"用户文字查询：{query or '（未提供文字，只上传了图片）'}\n"
        f"上传图片数量：{len(input_images)}\n"
        "请把文字需求和图片内容合并，输出适合 PicRetrieve/CLIP 文本搜图的 query。"
    )
    if not input_images:
        return text

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in input_images[:MAX_INPUT_IMAGES]:
        data_url = str(image.get("data_url") or "").strip()
        if not data_url:
            continue
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def _resolve_retrieval_mode(
    query: str, input_images: list[dict[str, Any]], llm_plan: dict[str, Any]
) -> str:
    if not input_images:
        return "text_only"
    if not query.strip():
        return "image_direct"

    # Reference-image instructions are product semantics, so they override the
    # model's tendency to invent parallel text queries.
    if _looks_like_image_reference_instruction(query):
        return "image_direct"

    mode = str(llm_plan.get("retrieval_mode") or "").strip()
    if mode in IMAGE_RETRIEVAL_MODES:
        return mode

    return "image_guided"


def _looks_like_image_reference_instruction(query: str) -> bool:
    text = query.strip().lower()
    return any(marker in text for marker in IMAGE_REFERENCE_MARKERS)


def _clean_queries(values: Any, fallback: str) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = [fallback]

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        cleaned.append(text[:200])
        seen.add(key)
        if len(cleaned) >= 3:
            break
    return cleaned or [fallback]


def _planner_confidence(value: Any, default: float) -> float:
    number = _float_between(value, default=default)
    return max(0.5, number)


def _planner_rationale(
    mode: str,
    retrieval_mode: str,
    input_images: int,
    rewritten_queries: list[str],
    suggested_queries: list[str],
) -> str:
    if mode == "fallback":
        return "LLM 结构化规划失败，使用规则路由。"
    if retrieval_mode == "image_direct":
        return "图片作为主要查询，直接以图搜图。"
    if retrieval_mode == "image_guided":
        return "图片直接检索，文本作为约束信息。"
    if retrieval_mode == "hybrid_text_image":
        return f"图文混合，执行以图搜图和 {len(rewritten_queries)} 条文本搜图。"
    if input_images and suggested_queries:
        return "模型已看图并生成视觉检索建议。"
    return f"生成 {len(rewritten_queries)} 条文本检索 query。"


def _float_between(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
