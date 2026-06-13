"""MMRAG PicRetrieve Agent web prototype.

Run:
  uv run uvicorn mrag.web:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from mrag.graph import build_picretrieve_graph
from mrag.models.evidence import EvidenceCard, Modality
from mrag.retrievers.base import Retriever
from mrag.state import MMRAGState

logger = logging.getLogger("mrag.web")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PICRETRIEVE_ROOT = PROJECT_ROOT / "PicRetrieve"
if PICRETRIEVE_ROOT.exists() and str(PICRETRIEVE_ROOT) not in sys.path:
    sys.path.insert(0, str(PICRETRIEVE_ROOT))

GRAPH_NODES = [
    {"id": "picretrieve_planner", "label": "Plan", "kind": "planner"},
    {"id": "modality_router", "label": "Route", "kind": "router"},
    {"id": "picretrieve", "label": "PicRetrieve", "kind": "module"},
    {"id": "reranker", "label": "Rank", "kind": "ranker"},
    {"id": "picretrieve_verifier", "label": "Verify", "kind": "llm"},
    {"id": "picretrieve_reflection", "label": "Retry", "kind": "loop"},
    {"id": "picretrieve_answer", "label": "Answer", "kind": "response"},
]

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LLM_MODEL = "google/gemma-4-26b-a4b"


@dataclass
class Runtime:
    """Lazy-loaded prototype runtime."""

    graph: Any | None = None
    registry: dict[Modality, Retriever] = field(default_factory=dict)
    store: Any | None = None
    image_root: Path | None = None
    data_dir: Path | None = None
    model_dir: Path | None = None
    llm: Any | None = None
    llm_ready: bool = False
    llm_model: str = ""
    llm_base_url: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.registry.get(Modality.IMAGE))


_runtime: Runtime | None = None

app = FastAPI(title="MMRAG PicRetrieve Agent", version="0.1.0")


def _resolve_path(*parts: str) -> Path:
    """Resolve a path relative to cwd or project root."""

    raw = Path(*parts)
    if raw.exists():
        return raw.resolve()
    for base in (PROJECT_ROOT, Path.cwd(), Path.cwd().parent):
        candidate = base / raw
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _resolve_image_root(data_dir: Path) -> Path:
    marker = data_dir / "image_root.txt"
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    return _resolve_path("PicRetrieve/data/corpus/images")


def _init_llm(runtime: Runtime) -> None:
    runtime.llm_base_url = os.getenv("MRAG_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    runtime.llm_model = os.getenv("MRAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    api_key = os.getenv("MRAG_LLM_API_KEY", "lm-studio")

    try:
        with urlopen(f"{runtime.llm_base_url.rstrip('/')}/models", timeout=2) as response:
            payload = response.read().decode("utf-8")
        runtime.llm_ready = runtime.llm_model in payload
    except (OSError, URLError) as exc:
        runtime.errors.append(f"LM Studio unavailable: {exc}")
        runtime.llm_ready = False

    if runtime.llm_ready:
        from mrag.utils.llm import LLMClient

        runtime.llm = LLMClient(
            model=runtime.llm_model, api_key=api_key, base_url=runtime.llm_base_url, temperature=0
        )


def _get_runtime() -> Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime

    runtime = Runtime()
    data_dir = _resolve_path("PicRetrieve/data")
    model_dir = _resolve_path("PicRetrieve/data/models/openai_clip-vit-base-patch32")
    runtime.data_dir = data_dir
    runtime.model_dir = model_dir
    _init_llm(runtime)

    try:
        from app.embedder import ClipEmbedder
        from app.index_store import IndexStore
    except Exception as exc:
        runtime.errors.append(f"PicRetrieve import failed: {exc}")
        runtime.graph = build_picretrieve_graph(registry=runtime.registry, llm=runtime.llm)
        _runtime = runtime
        return runtime

    if not data_dir.exists():
        runtime.errors.append(f"PicRetrieve data dir missing: {data_dir}")
    if not model_dir.exists():
        runtime.errors.append(f"CLIP model dir missing: {model_dir}")

    if not runtime.errors:
        try:
            embedder = ClipEmbedder(model_name=str(model_dir))
            store = IndexStore(data_dir)
            store.create_tables()

            from mrag.retrievers import ImageRetriever

            image_retriever = ImageRetriever(embedder, store)
            runtime.registry[Modality.IMAGE] = image_retriever
            runtime.store = store
            runtime.image_root = _resolve_image_root(data_dir)
            logger.info("PicRetrieve ready: %d images", store.count_items())
        except Exception as exc:
            logger.exception("Failed to initialize PicRetrieve runtime")
            runtime.errors.append(f"PicRetrieve runtime failed: {exc}")

    runtime.graph = build_picretrieve_graph(registry=runtime.registry, llm=runtime.llm)
    _runtime = runtime
    return runtime


def _state_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _serialize_card(card: EvidenceCard) -> dict[str, Any]:
    rel_path = card.locator or card.raw_reference
    meta = card.metadata or {}
    nested_meta = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    return {
        "id": card.source_id,
        "modality": card.modality.value,
        "title": Path(rel_path).name or card.source_id,
        "folder": str(Path(rel_path).parent) if rel_path else "",
        "locator": rel_path,
        "content_ref": card.content_ref,
        "score": round(card.score, 4),
        "confidence": card.confidence.value,
        "image_url": f"/api/images/{card.source_id}" if card.modality == Modality.IMAGE else "",
        "width": meta.get("width"),
        "height": meta.get("height"),
        "visual_score": meta.get("visual_score"),
        "metadata_score": meta.get("metadata_score"),
        "metadata": nested_meta,
    }


def _module_status(runtime: Runtime) -> list[dict[str, Any]]:
    image_retriever = runtime.registry.get(Modality.IMAGE)
    items = getattr(image_retriever, "_item_count", 0) if image_retriever else 0
    return [
        {
            "id": "lm-studio",
            "name": "LM Studio",
            "modality": "llm",
            "ready": runtime.llm_ready,
            "metric": runtime.llm_model or "not configured",
            "role": "规划 / 改写 / 验证",
        },
        {
            "id": "picretrieve",
            "name": "PicRetrieve",
            "modality": "image",
            "ready": bool(image_retriever and image_retriever.is_ready),
            "items": items,
            "metric": f"{items:,} 张",
            "role": "图片召回",
        },
    ]


def _build_agent_state(body: AskBody, runtime: Runtime) -> MMRAGState:
    return MMRAGState(
        query=body.query.strip(),
        input_images=[image.model_dump() for image in body.input_images],
        top_k=body.top_k,
        active_modules=body.modules,
        llm_model=runtime.llm_model if runtime.llm_ready else "",
        max_retries=body.max_retries,
    )


def _result_payload(
    result: Any,
    body: AskBody,
    runtime: Runtime,
    latency_ms: int,
    stage_timings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cards = _state_get(result, "reranked_cards", []) or _state_get(result, "evidence_cards", [])
    return {
        "query": _state_get(result, "query", body.query),
        "answer": _state_get(result, "answer", ""),
        "confident": _state_get(result, "is_confident", False),
        "reason": _state_get(result, "confidence_reason", ""),
        "agent_plan": _state_get(result, "agent_plan", {}),
        "agent_verification": _state_get(result, "agent_verification", {}),
        "retry_count": _state_get(result, "retry_count", 0),
        "llm_model": runtime.llm_model if runtime.llm_ready else "",
        "input_images": len(body.input_images),
        "trace": _state_get(result, "trace", []),
        "graph_nodes": GRAPH_NODES,
        "stage_timings": stage_timings or [],
        "evidence": len(cards),
        "evidence_cards": [_serialize_card(card) for card in cards[: body.top_k]],
        "latency_ms": latency_ms,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=_json_default)
    return f"event: {event}\ndata: {payload}\n\n"


def _merge_update(accumulated: dict[str, Any], update: dict[str, Any]) -> None:
    accumulated.update(update)


def _next_node(node: str, update: dict[str, Any], accumulated: dict[str, Any]) -> str | None:
    if node == "picretrieve_planner":
        query_plan = update.get("query_plan") or accumulated.get("query_plan")
        if query_plan is not None and not getattr(query_plan, "need_retrieval", True):
            return "picretrieve_answer"
        return "modality_router"
    if node == "modality_router":
        return "picretrieve"
    if node == "picretrieve":
        return "reranker"
    if node == "reranker":
        return "picretrieve_verifier"
    if node == "picretrieve_verifier":
        confident = bool(update.get("is_confident"))
        retry_count = int(accumulated.get("retry_count") or 0)
        max_retries = int(accumulated.get("max_retries") or 0)
        return (
            "picretrieve_answer"
            if confident or retry_count >= max_retries
            else "picretrieve_reflection"
        )
    if node == "picretrieve_reflection":
        return "picretrieve"
    return None


def _stage_summary(node: str, update: dict[str, Any]) -> dict[str, Any]:
    if node == "picretrieve_planner":
        plan = update.get("agent_plan", {})
        queries = plan.get("rewritten_queries", [])
        retrieval_mode = plan.get("retrieval_mode", "")
        direct_image = retrieval_mode in {"image_direct", "image_guided"}
        return {
            "title": "LLM Planner 完成",
            "detail": (
                "图片输入将直接用于 PicRetrieve 以图搜图。"
                if direct_image
                else f"调用本地模型生成 {len(queries)} 条视觉检索 query。"
            ),
            "items": queries or plan.get("suggested_queries", []),
            "meta": {
                "mode": plan.get("mode", ""),
                "retrieval_mode": retrieval_mode,
                "confidence": plan.get("confidence", 0),
                "rationale": plan.get("rationale", ""),
            },
        }

    if node == "modality_router":
        retrieval_plan = update.get("retrieval_plan")
        tasks = getattr(retrieval_plan, "tasks", []) if retrieval_plan is not None else []
        task_rows = [
            (
                f"{task.modality.value}: 直接使用上传图片 · top_k={task.top_k}"
                if task.query_image is not None
                else f"{task.modality.value}: {task.subquery} · top_k={task.top_k}"
            )
            for task in tasks
        ]
        return {
            "title": "Router 完成",
            "detail": f"把 planner 结果转换成 {len(tasks)} 个 PicRetrieve 检索任务。",
            "items": task_rows,
            "meta": {
                "budget": getattr(retrieval_plan, "budget", 0) if retrieval_plan else 0,
                "explanation": getattr(retrieval_plan, "explanation", "") if retrieval_plan else "",
            },
        }

    if node == "picretrieve":
        cards = update.get("evidence_cards", [])
        top = sorted(cards, key=lambda card: card.score, reverse=True)[:3]
        return {
            "title": "PicRetrieve 召回完成",
            "detail": f"执行图片召回，返回 {len(cards)} 张候选图片。",
            "items": [f"{card.source_id} · {card.score:.3f} · {card.locator}" for card in top],
            "meta": {"top_score": round(top[0].score, 4) if top else 0},
            "preview_cards": [_serialize_card(card) for card in top],
        }

    if node == "reranker":
        cards = update.get("reranked_cards", [])
        top = cards[:3]
        return {
            "title": "Rank 完成",
            "detail": f"去重并按相关度排序，保留 {len(cards)} 张候选图片。",
            "items": [f"{card.source_id} · {card.score:.3f} · {card.locator}" for card in top],
            "meta": {"top_score": round(top[0].score, 4) if top else 0},
        }

    if node == "picretrieve_verifier":
        verification = update.get("agent_verification", {})
        passed = bool(update.get("is_confident"))
        improved = verification.get("improved_queries", [])
        return {
            "title": "LLM Verifier 完成",
            "detail": verification.get("reason", "") or update.get("confidence_reason", ""),
            "items": improved,
            "meta": {
                "mode": verification.get("mode", ""),
                "passed": passed,
                "confidence": verification.get("confidence", 0),
            },
        }

    if node == "picretrieve_reflection":
        retrieval_plan = update.get("retrieval_plan")
        tasks = getattr(retrieval_plan, "tasks", []) if retrieval_plan is not None else []
        return {
            "title": "Retry 计划完成",
            "detail": update.get("retry_feedback", "") or "根据 verifier 反馈生成新的检索任务。",
            "items": [task.subquery for task in tasks],
            "meta": {"retry_count": update.get("retry_count", 0)},
        }

    if node == "picretrieve_answer":
        answer = update.get("answer", "")
        return {
            "title": "Answer 完成",
            "detail": "本地模型生成最终总结。",
            "items": [answer] if answer else [],
            "meta": {"chars": len(answer)},
        }

    return {"title": f"{node} 完成", "detail": "", "items": [], "meta": {}}


@app.on_event("startup")
async def _startup() -> None:
    _get_runtime()


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMRAG PicRetrieve Agent</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Afacad:wght@500;600;700&family=Noto+Sans+SC:wght@400;500;700;800&display=swap");
    :root {
      --paper: oklch(97.8% 0.012 96);
      --surface: oklch(99% 0.007 96);
      --field: oklch(94.5% 0.014 96);
      --ink: oklch(24% 0.024 78);
      --muted: oklch(48% 0.027 78);
      --line: oklch(84% 0.018 85);
      --accent: oklch(57% 0.17 36);
      --accent-ink: oklch(98% 0.01 60);
      --teal: oklch(55% 0.105 178);
      --ok: oklch(57% 0.12 145);
      --warn: oklch(62% 0.14 67);
      --space-xs: 4px;
      --space-sm: 8px;
      --space-md: 12px;
      --space-lg: 16px;
      --space-xl: 24px;
      --space-2xl: 32px;
      --space-3xl: 48px;
      color-scheme: light;
      font-family: "Noto Sans SC", "Afacad", ui-sans-serif, system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
    }
    button, textarea, input { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
    }
    .side {
      min-width: 0;
      display: grid;
      align-content: start;
      gap: var(--space-xl);
      padding: clamp(20px, 3vw, 40px);
      border-right: 1px solid var(--line);
      background: color-mix(in oklch, var(--surface), var(--paper) 35%);
    }
    .brand { display: grid; gap: var(--space-md); }
    h1 {
      margin: 0;
      max-width: 10ch;
      font-family: "Afacad", "Noto Sans SC", sans-serif;
      font-size: 2.35rem;
      line-height: .95;
      letter-spacing: 0;
    }
    .kicker {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      max-width: 36ch;
    }
    .panel {
      display: grid;
      gap: var(--space-lg);
      padding-top: var(--space-xl);
      border-top: 1px solid var(--line);
    }
    label {
      display: grid;
      gap: var(--space-sm);
      color: var(--muted);
      font-size: .86rem;
      font-weight: 700;
    }
    textarea {
      width: 100%;
      min-height: 124px;
      resize: vertical;
      padding: 12px 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      outline: none;
      line-height: 1.55;
    }
    textarea:focus, input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent), transparent 76%);
    }
    .composer {
      display: grid;
      gap: var(--space-sm);
      padding: var(--space-sm);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .composer:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent), transparent 76%);
    }
    .composer textarea {
      min-height: 112px;
      padding: 6px 4px;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }
    .composer textarea:focus {
      border: 0;
      box-shadow: none;
    }
    .composer-actions {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: .78rem;
    }
    .hidden-file { display: none; }
    .image-preview {
      display: none;
      grid-template-columns: 64px 1fr auto;
      gap: var(--space-md);
      align-items: center;
      padding: var(--space-xs);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--field);
    }
    .image-preview.show { display: grid; }
    .image-preview img {
      width: 64px;
      height: 64px;
      border-radius: 6px;
      object-fit: cover;
      background: var(--field);
    }
    .image-preview-text {
      min-width: 0;
      color: var(--muted);
      font-size: .82rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .ghost {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--field);
      color: var(--muted);
      cursor: pointer;
      padding: 0 10px;
      font-weight: 700;
    }
    .range-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: var(--space-md);
      align-items: center;
    }
    input[type="range"] {
      accent-color: var(--accent);
      width: 100%;
    }
    .count {
      min-width: 3ch;
      text-align: right;
      color: var(--ink);
      font-weight: 800;
    }
    .primary {
      min-height: 46px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: var(--accent-ink);
      font-weight: 800;
      cursor: pointer;
    }
    .primary:disabled { opacity: .58; cursor: progress; }
    .module-list { display: grid; gap: var(--space-sm); }
    .module-row {
      display: grid;
      grid-template-columns: 12px 1fr auto;
      align-items: center;
      gap: var(--space-md);
      padding: 10px 0;
      border-bottom: 1px solid color-mix(in oklch, var(--line), transparent 35%);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.ready { background: var(--ok); }
    .module-name { font-weight: 800; }
    .module-role { color: var(--muted); font-size: .82rem; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      font-size: .78rem;
      color: var(--muted);
      background: var(--surface);
      white-space: nowrap;
    }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: var(--space-xl);
      padding: clamp(20px, 3vw, 40px);
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: var(--space-xl);
      padding-bottom: var(--space-lg);
      border-bottom: 1px solid var(--line);
    }
    h2 {
      margin: 0;
      font-size: 1.25rem;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .statusline {
      color: var(--muted);
      font-size: .9rem;
      text-align: right;
    }
    .graph {
      display: grid;
      grid-template-columns: repeat(7, minmax(88px, 1fr));
      gap: var(--space-sm);
      overflow-x: auto;
    }
    .node {
      min-height: 58px;
      display: grid;
      align-content: center;
      gap: 3px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 9px 10px;
    }
    .node.active {
      border-color: color-mix(in oklch, var(--teal), var(--line) 20%);
      background: color-mix(in oklch, var(--teal), var(--surface) 88%);
    }
    .node.running {
      border-color: var(--accent);
      background: color-mix(in oklch, var(--accent), var(--surface) 88%);
    }
    .node.done {
      border-color: color-mix(in oklch, var(--ok), var(--line) 20%);
      background: color-mix(in oklch, var(--ok), var(--surface) 90%);
    }
    .node.error {
      border-color: var(--warn);
      background: color-mix(in oklch, var(--warn), var(--surface) 86%);
    }
    .node-label { font-weight: 800; }
    .node-kind { color: var(--muted); font-size: .75rem; }
    .node-time {
      color: var(--accent);
      font-size: .72rem;
      font-weight: 800;
      min-height: 1em;
    }
    .pipeline {
      display: none;
      gap: var(--space-md);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: var(--space-lg);
    }
    .pipeline.show { display: grid; }
    .pipeline-head {
      display: flex;
      justify-content: space-between;
      gap: var(--space-lg);
      align-items: baseline;
    }
    .pipeline-head h3 {
      margin: 0;
      font-size: .98rem;
      letter-spacing: 0;
    }
    .pipeline-status {
      color: var(--muted);
      font-size: .82rem;
      text-align: right;
    }
    .event-log {
      display: grid;
      gap: var(--space-sm);
    }
    .event-row {
      display: grid;
      gap: var(--space-xs);
      padding: var(--space-md);
      border: 1px solid color-mix(in oklch, var(--line), transparent 30%);
      border-radius: 8px;
      background: color-mix(in oklch, var(--field), var(--surface) 55%);
    }
    .event-row.running {
      background: color-mix(in oklch, var(--accent), var(--surface) 92%);
    }
    .event-row.error {
      background: color-mix(in oklch, var(--warn), var(--surface) 88%);
    }
    .event-title {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: var(--space-sm);
      font-weight: 800;
    }
    .event-detail {
      color: var(--muted);
      line-height: 1.5;
      font-size: .86rem;
    }
    .event-items {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
    }
    .summary {
      display: none;
      gap: var(--space-md);
      align-items: start;
      justify-content: space-between;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: var(--space-lg);
    }
    .summary.show { display: grid; }
    .answer { line-height: 1.65; }
    .insights {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: var(--space-lg);
      padding-top: var(--space-md);
      border-top: 1px solid var(--line);
    }
    .insight {
      display: grid;
      align-content: start;
      gap: var(--space-sm);
      min-width: 0;
    }
    .insight h3 {
      margin: 0;
      font-size: .9rem;
      letter-spacing: 0;
    }
    .insight-body {
      color: var(--muted);
      font-size: .86rem;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }
    .query-list {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
      margin-top: var(--space-sm);
    }
    .trace {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
    }
    .trace span {
      border-radius: 999px;
      background: var(--field);
      padding: 4px 9px;
      color: var(--muted);
      font-size: .78rem;
    }
    .results {
      min-width: 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: var(--space-lg);
      align-content: start;
    }
    .card {
      min-width: 0;
      display: grid;
      gap: var(--space-md);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: var(--space-md);
    }
    .thumb-wrap {
      aspect-ratio: 4 / 3;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 6px;
      background: var(--field);
    }
    .thumb {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .card-body { display: grid; gap: var(--space-sm); min-width: 0; }
    .title {
      font-weight: 800;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
      color: var(--muted);
      font-size: .78rem;
    }
    .score { color: var(--accent); font-weight: 800; }
    .empty {
      display: grid;
      place-items: center;
      min-height: 320px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      text-align: center;
      padding: var(--space-xl);
    }
    .empty.hidden { display: none; }
    @media (max-width: 920px) {
      .app { grid-template-columns: 1fr; }
      .side { border-right: 0; border-bottom: 1px solid var(--line); }
      h1 { max-width: none; }
      .graph { grid-template-columns: repeat(7, minmax(110px, 1fr)); }
    }
    @media (max-width: 560px) {
      .topbar { display: grid; align-items: start; }
      .statusline { text-align: left; }
      .results { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="side">
      <section class="brand">
        <h1>MMRAG Agent</h1>
        <p class="kicker">LangGraph 驱动的图片召回原型，当前模块为 PicRetrieve。</p>
      </section>

      <form class="panel" id="searchForm">
        <label>
          查询
          <div class="composer">
            <div class="image-preview" id="imagePreview">
              <img id="imagePreviewThumb" alt="查询图片预览">
              <div class="image-preview-text" id="imagePreviewText"></div>
              <button class="ghost" id="clearImageButton" type="button">移除</button>
            </div>
            <textarea id="queryInput" spellcheck="false" placeholder="输入文字，或添加图片后直接运行">a black dog running in the grass</textarea>
            <div class="composer-actions">
              <button class="ghost" id="attachImageButton" type="button">添加图片</button>
              <input class="hidden-file" id="imageInput" type="file" accept="image/*">
              <span>支持纯文本、纯图片、图文指令</span>
            </div>
          </div>
        </label>
        <label>
          返回数量
          <div class="range-row">
            <input id="topKInput" type="range" min="3" max="20" value="8">
            <span class="count" id="topKValue">8</span>
          </div>
        </label>
        <button class="primary" id="runButton" type="submit">运行 Agent</button>
      </form>

      <section class="panel">
        <div class="module-list" id="moduleList"></div>
      </section>
    </aside>

    <main class="main">
      <header class="topbar">
        <h2>PicRetrieve 图片召回</h2>
        <div class="statusline" id="statusLine">初始化中</div>
      </header>

      <section class="graph" id="graphNodes"></section>

      <section class="pipeline" id="pipelinePanel">
        <div class="pipeline-head">
          <h3>Pipeline Trace</h3>
          <div class="pipeline-status" id="pipelineStatus">等待运行</div>
        </div>
        <div class="event-log" id="eventLog"></div>
      </section>

      <section class="summary" id="summaryBox">
        <div class="answer" id="answerText"></div>
        <div class="insights" id="insightsBox">
          <section class="insight">
            <h3>LLM Planner</h3>
            <div class="insight-body" id="planText"></div>
          </section>
          <section class="insight">
            <h3>LLM Verifier</h3>
            <div class="insight-body" id="verifyText"></div>
          </section>
        </div>
        <div class="trace" id="traceList"></div>
      </section>

      <section class="empty" id="emptyState">输入查询后运行 Agent</section>
      <section class="results" id="resultsGrid"></section>
    </main>
  </div>

  <script>
    const nodes = __GRAPH_NODES__;
    const $ = (id) => document.getElementById(id);
    const state = {
      running: false,
      nodeStatus: {},
      nodeTimings: {},
      imagePayload: null,
      eventCount: 0
    };

    function escapeHtml(value) {
      const div = document.createElement("div");
      div.textContent = value == null ? "" : String(value);
      return div.innerHTML;
    }

    function renderGraph(activeTrace = []) {
      const aliases = { retrievers: "picretrieve", verifier: "picretrieve_verifier" };
      const active = new Set(activeTrace.map((item) => {
        const key = String(item).split(":")[0];
        return aliases[key] || key;
      }));
      $("graphNodes").innerHTML = nodes.map((node) => `
        <div class="node ${state.nodeStatus[node.id] || (active.has(node.id) ? "active" : "")}">
          <div class="node-label">${escapeHtml(node.label)}</div>
          <div class="node-kind">${escapeHtml(state.nodeStatus[node.id] || node.kind)}</div>
          <div class="node-time">${escapeHtml(formatNodeTiming(node.id))}</div>
        </div>
      `).join("");
    }

    function resetPipeline() {
      state.nodeStatus = {};
      state.nodeTimings = {};
      state.eventCount = 0;
      nodes.forEach((node) => { state.nodeStatus[node.id] = "pending"; });
      $("eventLog").innerHTML = "";
      $("pipelineStatus").textContent = "准备执行";
      $("pipelinePanel").classList.add("show");
      renderGraph();
    }

    function formatMs(value) {
      if (value === "" || value === null || value === undefined) return "";
      const ms = Number(value);
      if (!Number.isFinite(ms)) return "";
      if (ms <= 0) return "0 ms";
      if (ms < 1000) return `${Math.round(ms)} ms`;
      return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
    }

    function formatNodeTiming(nodeId) {
      const timing = state.nodeTimings[nodeId];
      if (!timing) return "";
      const duration = formatMs(timing.total_ms);
      if (!duration) return "";
      return timing.runs > 1 ? `${timing.runs}x · ${duration}` : duration;
    }

    function appendPipelineEvent(payload) {
      state.eventCount += 1;
      const items = Array.isArray(payload.items) ? payload.items : [];
      const meta = payload.meta || {};
      const durationText = payload.status === "running"
        ? "计时中"
        : formatMs(payload.duration_ms);
      const elapsedText = payload.elapsed_ms ? `总计 ${formatMs(payload.elapsed_ms)}` : "";
      const metaText = Object.entries(meta)
        .filter(([, value]) => value !== "" && value !== null && value !== undefined)
        .map(([key, value]) => `${key}: ${typeof value === "number" ? Number(value).toFixed(2) : value}`)
        .join(" · ");
      const itemHtml = items.length
        ? `<div class="event-items">${items.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("")}</div>`
        : "";
      $("eventLog").insertAdjacentHTML("beforeend", `
        <article class="event-row ${escapeHtml(payload.status || "")}">
          <div class="event-title">
            <span>${state.eventCount}. ${escapeHtml(payload.title || payload.node || "Stage")}</span>
            <span class="pill">${escapeHtml([payload.status || "", durationText, elapsedText].filter(Boolean).join(" · "))}</span>
          </div>
          <div class="event-detail">${escapeHtml(payload.detail || "")}</div>
          ${metaText ? `<div class="event-detail">${escapeHtml(metaText)}</div>` : ""}
          ${itemHtml}
        </article>
      `);
      $("pipelineStatus").textContent = payload.status === "running"
        ? `${payload.node} 运行中`
        : `${payload.node} ${payload.status}`;
    }

    function handleStageEvent(payload) {
      if (payload.node) {
        state.nodeStatus[payload.node] = payload.status || "done";
        if (payload.status === "done" && payload.duration_ms != null) {
          const previous = state.nodeTimings[payload.node] || { total_ms: 0, runs: 0 };
          state.nodeTimings[payload.node] = {
            total_ms: previous.total_ms + Number(payload.duration_ms || 0),
            runs: previous.runs + 1
          };
        }
      }
      appendPipelineEvent(payload);
      renderGraph();
    }

    function parseSseMessages(buffer) {
      const messages = [];
      let boundary = buffer.indexOf("\\n\\n");
      while (boundary >= 0) {
        const raw = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const eventLine = raw.split("\\n").find((line) => line.startsWith("event:"));
        const dataLines = raw.split("\\n").filter((line) => line.startsWith("data:"));
        if (eventLine && dataLines.length) {
          messages.push({
            event: eventLine.slice(6).trim(),
            data: JSON.parse(dataLines.map((line) => line.slice(5).trim()).join("\\n"))
          });
        }
        boundary = buffer.indexOf("\\n\\n");
      }
      return { messages, buffer };
    }

    function renderModules(modules) {
      $("moduleList").innerHTML = modules.map((module) => `
        <div class="module-row">
          <span class="dot ${module.ready ? "ready" : ""}"></span>
          <div>
            <div class="module-name">${escapeHtml(module.name)}</div>
            <div class="module-role">${escapeHtml(module.role)}</div>
          </div>
          <span class="pill">${escapeHtml(module.metric || "")}</span>
        </div>
      `).join("");
    }

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        renderModules(data.modules || []);
        $("statusLine").textContent = data.ready
          ? `${(data.images || 0).toLocaleString()} 张图片 · ${data.llm_ready ? data.llm_model : "LLM 未连接"}`
          : (data.errors || ["未就绪"]).join("；");
      } catch (error) {
        $("statusLine").textContent = `状态读取失败：${error.message}`;
      }
    }

    function setRunning(running) {
      state.running = running;
      $("runButton").disabled = running;
      $("runButton").textContent = running ? "运行中" : "运行 Agent";
    }

    function renderResult(data) {
      const cards = data.evidence_cards || [];
      $("answerText").textContent = data.answer || "";
      renderInsights(data);
      $("traceList").innerHTML = (data.trace || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
      if (Array.isArray(data.stage_timings)) {
        data.stage_timings.forEach((timing) => {
          if (!timing.node || timing.duration_ms == null) return;
          const previous = state.nodeTimings[timing.node] || { total_ms: 0, runs: 0 };
          if (previous.runs > 0) return;
          state.nodeTimings[timing.node] = {
            total_ms: Number(timing.duration_ms || 0),
            runs: 1
          };
        });
      }
      $("summaryBox").classList.add("show");
      renderGraph(data.trace || []);

      $("emptyState").classList.toggle("hidden", cards.length > 0);
      $("emptyState").textContent = cards.length ? "" : "没有召回结果";
      $("resultsGrid").innerHTML = cards.map((card) => {
        const size = card.width && card.height ? `${card.width} x ${card.height}` : "unknown";
        const visual = card.visual_score == null ? "" : `<span>visual ${Number(card.visual_score).toFixed(3)}</span>`;
        const metadata = card.metadata_score == null ? "" : `<span>meta ${Number(card.metadata_score).toFixed(3)}</span>`;
        return `
          <article class="card">
            <div class="thumb-wrap">
              <img class="thumb" src="${escapeHtml(card.image_url)}" alt="${escapeHtml(card.title)}" loading="eager">
            </div>
            <div class="card-body">
              <div class="title">${escapeHtml(card.title)}</div>
              <div class="meta">
                <span class="score">${Number(card.score || 0).toFixed(3)}</span>
                <span>${escapeHtml(card.confidence)}</span>
                <span>${escapeHtml(size)}</span>
                ${visual}
                ${metadata}
              </div>
              <div class="meta"><span>${escapeHtml(card.folder)}</span></div>
            </div>
          </article>
        `;
      }).join("");
      $("statusLine").textContent = `${cards.length} 条结果 · ${data.latency_ms || 0} ms`;
    }

    function renderInsights(data) {
      const plan = data.agent_plan || {};
      const verification = data.agent_verification || {};
      $("planText").innerHTML = `
        <div>${escapeHtml(plan.mode || "unknown")} · confidence ${Number(plan.confidence || 0).toFixed(2)}</div>
        <div>${escapeHtml(plan.rationale || "")}</div>
        ${renderQueryList(plan.rewritten_queries || [])}
      `;
      $("verifyText").innerHTML = `
        <div>${escapeHtml(verification.mode || "unknown")} · ${verification.is_sufficient ? "通过" : "建议重试"} · confidence ${Number(verification.confidence || 0).toFixed(2)}</div>
        <div>${escapeHtml(verification.reason || data.reason || "")}</div>
        ${renderQueryList(verification.improved_queries || [])}
        <div>retry ${Number(data.retry_count || 0)}</div>
      `;
    }

    function renderQueryList(queries) {
      if (!queries.length) return "";
      return `<div class="query-list">${queries.map((query) => `<span class="pill">${escapeHtml(query)}</span>`).join("")}</div>`;
    }

    async function runAgent(event) {
      event.preventDefault();
      const query = $("queryInput").value.trim();
      if ((!query && !state.imagePayload) || state.running) return;
      setRunning(true);
      resetPipeline();
      $("emptyState").classList.remove("hidden");
      $("emptyState").textContent = state.imagePayload
        ? "LangGraph 正在处理图文输入，流水线会逐步更新"
        : "LangGraph 正在执行，流水线会逐步更新";
      $("resultsGrid").innerHTML = "";
      $("summaryBox").classList.remove("show");
      try {
        const response = await fetch("/api/ask/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            input_images: state.imagePayload ? [state.imagePayload] : [],
            top_k: Number($("topKInput").value),
            max_retries: 1,
            modules: ["picretrieve"]
          })
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || response.statusText);
        }
        if (!response.body) throw new Error("浏览器不支持流式响应");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalPayload = null;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parsed = parseSseMessages(buffer);
          buffer = parsed.buffer;
          for (const message of parsed.messages) {
            if (message.event === "pipeline") {
              $("pipelineStatus").textContent = `使用 ${message.data.llm_model || "无 LLM"} 执行`;
            } else if (message.event === "stage") {
              handleStageEvent(message.data);
            } else if (message.event === "final") {
              finalPayload = message.data;
              $("pipelineStatus").textContent = "流水线完成";
            } else if (message.event === "error") {
              handleStageEvent({ ...message.data, status: "error" });
              throw new Error(message.data.detail || "stream failed");
            }
          }
        }
        if (!finalPayload) throw new Error("流水线未返回最终结果");
        renderResult(finalPayload);
      } catch (error) {
        $("emptyState").textContent = `请求失败：${error.message}`;
        $("pipelineStatus").textContent = `执行失败：${error.message}`;
        renderGraph();
      } finally {
        setRunning(false);
      }
    }

    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("图片读取失败"));
        reader.readAsDataURL(file);
      });
    }

    async function handleImageInput() {
      const file = $("imageInput").files[0];
      if (!file) {
        clearImage();
        return;
      }
      if (!file.type.startsWith("image/")) {
        clearImage();
        $("emptyState").textContent = "请选择图片文件";
        return;
      }
      if (file.size > 8 * 1024 * 1024) {
        clearImage();
        $("emptyState").textContent = "图片不能超过 8 MB";
        return;
      }
      const dataUrl = await readFileAsDataUrl(file);
      state.imagePayload = {
        name: file.name,
        mime_type: file.type || "image/jpeg",
        size: file.size,
        data_url: dataUrl
      };
      $("imagePreviewThumb").src = dataUrl;
      $("imagePreviewText").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
      $("imagePreview").classList.add("show");
    }

    function clearImage() {
      state.imagePayload = null;
      $("imageInput").value = "";
      $("imagePreviewThumb").removeAttribute("src");
      $("imagePreviewText").textContent = "";
      $("imagePreview").classList.remove("show");
    }

    $("topKInput").addEventListener("input", () => {
      $("topKValue").textContent = $("topKInput").value;
    });
    $("imageInput").addEventListener("change", () => {
      handleImageInput().catch((error) => {
        clearImage();
        $("emptyState").textContent = `图片读取失败：${error.message}`;
      });
    });
    $("attachImageButton").addEventListener("click", () => $("imageInput").click());
    $("clearImageButton").addEventListener("click", clearImage);
    $("searchForm").addEventListener("submit", runAgent);

    renderGraph();
    loadStatus();
  </script>
</body>
</html>
""".replace("__GRAPH_NODES__", repr(GRAPH_NODES))


class ImageInput(BaseModel):
    name: str = Field(default="", max_length=240)
    mime_type: str = Field(default="image/jpeg", max_length=80)
    size: int = Field(default=0, ge=0, le=8 * 1024 * 1024)
    data_url: str = Field(default="", min_length=1, max_length=12 * 1024 * 1024)


class AskBody(BaseModel):
    query: str = Field(default="", max_length=1000)
    input_images: list[ImageInput] = Field(default_factory=list, max_length=3)
    top_k: int = Field(default=8, ge=1, le=50)
    max_retries: int = Field(default=1, ge=0, le=3)
    modules: list[str] = Field(default_factory=lambda: ["picretrieve"])


def _validate_ask_body(body: AskBody) -> None:
    if not body.query.strip() and not body.input_images:
        raise HTTPException(status_code=400, detail="请输入文字查询或上传查询图片")
    for image in body.input_images:
        if not image.mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="input_images 只支持 image/*")
        if not image.data_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="input_images 必须是 data:image data URL")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/favicon.ico", status_code=204)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> dict[str, Any]:
    runtime = _get_runtime()
    modules = _module_status(runtime)
    images = next(
        (module.get("items", 0) for module in modules if module["id"] == "picretrieve"), 0
    )
    return {
        "ready": runtime.ready,
        "llm_ready": runtime.llm_ready,
        "llm_model": runtime.llm_model,
        "llm_base_url": runtime.llm_base_url,
        "modules": modules,
        "graph_nodes": GRAPH_NODES,
        "images": images,
        "model": runtime.model_dir.name if runtime.model_dir else "",
        "data_dir": str(runtime.data_dir) if runtime.data_dir else "",
        "errors": runtime.errors,
    }


@app.post("/api/ask")
def ask(body: AskBody) -> dict[str, Any]:
    runtime = _get_runtime()
    _validate_ask_body(body)
    if "picretrieve" not in body.modules:
        raise HTTPException(status_code=400, detail="picretrieve module must be enabled")
    if runtime.graph is None:
        raise HTTPException(status_code=503, detail="LangGraph runtime is not initialized")

    t0 = time.perf_counter()
    try:
        result = runtime.graph.invoke(_build_agent_state(body, runtime))
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"agent execution failed: {exc}") from exc

    return _result_payload(result, body, runtime, round((time.perf_counter() - t0) * 1000))


@app.post("/api/ask/stream")
def ask_stream(body: AskBody) -> StreamingResponse:
    runtime = _get_runtime()
    _validate_ask_body(body)
    if "picretrieve" not in body.modules:
        raise HTTPException(status_code=400, detail="picretrieve module must be enabled")
    if runtime.graph is None:
        raise HTTPException(status_code=503, detail="LangGraph runtime is not initialized")

    def generate():
        t0 = time.perf_counter()
        stage_started_at: dict[str, float] = {}
        stage_runs: dict[str, int] = {}
        stage_timings: list[dict[str, Any]] = []
        accumulated: dict[str, Any] = {
            "query": body.query.strip(),
            "top_k": body.top_k,
            "max_retries": body.max_retries,
            "active_modules": body.modules,
            "trace": [],
        }
        yield _sse(
            "pipeline",
            {
                "nodes": GRAPH_NODES,
                "query": body.query.strip(),
                "input_images": len(body.input_images),
                "llm_model": runtime.llm_model if runtime.llm_ready else "",
            },
        )

        def elapsed_ms(now: float | None = None) -> int:
            current = now if now is not None else time.perf_counter()
            return round((current - t0) * 1000)

        def start_stage(
            node: str, title: str, detail: str, items: list[Any] | None = None
        ) -> dict[str, Any]:
            now = time.perf_counter()
            stage_started_at[node] = now
            stage_runs[node] = stage_runs.get(node, 0) + 1
            return {
                "node": node,
                "status": "running",
                "title": title,
                "detail": detail,
                "items": items or [],
                "meta": {},
                "run": stage_runs[node],
                "started_at_ms": elapsed_ms(now),
                "elapsed_ms": elapsed_ms(now),
            }

        def finish_stage(node: str, summary: dict[str, Any]) -> dict[str, Any]:
            now = time.perf_counter()
            if node not in stage_started_at:
                stage_started_at[node] = t0
                stage_runs[node] = stage_runs.get(node, 0) + 1
            duration_ms = round((now - stage_started_at[node]) * 1000)
            record = {
                "node": node,
                "run": stage_runs.get(node, 1),
                "duration_ms": duration_ms,
                "elapsed_ms": elapsed_ms(now),
            }
            stage_timings.append(record)
            return {"node": node, "status": "done", **summary, **record}

        yield _sse(
            "stage",
            start_stage(
                "picretrieve_planner",
                "LLM Planner 运行中",
                "正在调用本地模型进行 query 规划和改写。",
            ),
        )

        try:
            for chunk in runtime.graph.stream(
                _build_agent_state(body, runtime), stream_mode="updates"
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    _merge_update(accumulated, update)
                    summary = _stage_summary(node, update)
                    yield _sse("stage", finish_stage(node, summary))

                    next_node = _next_node(node, update, accumulated)
                    if next_node:
                        yield _sse(
                            "stage",
                            start_stage(
                                next_node,
                                f"{next_node} 运行中",
                                "等待该节点完成并返回中间结果。",
                            ),
                        )

            yield _sse(
                "final",
                _result_payload(
                    accumulated,
                    body,
                    runtime,
                    round((time.perf_counter() - t0) * 1000),
                    stage_timings=stage_timings,
                ),
            )
        except Exception as exc:
            logger.exception("Agent stream failed")
            yield _sse(
                "error",
                {
                    "node": "",
                    "status": "error",
                    "title": "流水线执行失败",
                    "detail": str(exc),
                    "items": [],
                    "meta": {},
                },
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/images/{item_id}")
def image_file(item_id: int) -> FileResponse:
    runtime = _get_runtime()
    if runtime.store is None:
        raise HTTPException(status_code=503, detail="PicRetrieve store is not initialized")

    item = runtime.store.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="image not found")

    path = Path(item["path"]).expanduser().resolve()
    allowed_roots = [root for root in (runtime.image_root, runtime.data_dir) if root is not None]
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="image is outside allowed roots")
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mrag.web:app", host="127.0.0.1", port=8000, reload=True)
