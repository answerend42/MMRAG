"""FastAPI HTTP 接口。"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field

from app.config import load_settings, resolve_image_root
from app.embedder import ClipEmbedder, select_device
from app.index_store import IndexStore
from app.retrieval import PROFILES, Retriever

settings = load_settings()
app = FastAPI(title="PicRetrieve", version="0.1.0")

logger = logging.getLogger("picretrieve")

_setup_logging_done = False


def setup_logging() -> None:
    """! @brief 配置 picretrieve 日志格式和级别。"""
    global _setup_logging_done
    if _setup_logging_done:
        return
    _setup_logging_done = True
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger("picretrieve")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # 抑制第三方库的调试日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


@app.on_event("startup")
async def on_startup() -> None:
    """! @brief 服务启动时初始化日志并打印配置摘要。"""
    setup_logging()
    logger.info(
        "PicRetrieve %s starting — device=%s model=%s data=%s",
        app.version,
        settings.device or "auto",
        settings.model_name,
        settings.data_dir,
    )

_store: IndexStore | None = None
_embedder: ClipEmbedder | None = None
_retriever: Retriever | None = None


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PicRetrieve</title>
  <style>
    :root {
      --ink: oklch(22% 0.018 230);
      --muted: oklch(48% 0.018 230);
      --line: oklch(84% 0.014 230);
      --paper: oklch(97% 0.01 230);
      --panel: oklch(99% 0.006 230);
      --field: oklch(94% 0.012 230);
      --accent: oklch(55% 0.18 35);
      --accent-ink: oklch(99% 0.006 35);
      --ok: oklch(58% 0.13 150);
      --space-xs: 4px;
      --space-sm: 8px;
      --space-md: 12px;
      --space-lg: 16px;
      --space-xl: 24px;
      --space-2xl: 32px;
      color-scheme: light;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      min-height: 100vh;
    }
    button, input, select, textarea { font: inherit; }
    .shell {
      display: grid;
      grid-template-columns: minmax(320px, 440px) minmax(0, 1fr);
      gap: var(--space-2xl);
      min-height: 100vh;
      padding: clamp(16px, 3vw, 40px);
    }
    .console {
      display: flex;
      flex-direction: column;
      gap: var(--space-xl);
      min-width: 0;
    }
    .mast {
      display: grid;
      gap: var(--space-md);
      border-bottom: 1px solid var(--line);
      padding-bottom: var(--space-xl);
    }
    h1 {
      margin: 0;
      font-size: 2rem;
      line-height: 1;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-sm);
      color: var(--muted);
      font-size: 0.88rem;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: color-mix(in oklch, var(--panel), var(--paper) 25%);
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--field);
    }
    .tab {
      min-height: 42px;
      border: 0;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
    }
    .tab[aria-selected="true"] {
      background: var(--ink);
      color: var(--panel);
    }
    .panel {
      display: grid;
      gap: var(--space-lg);
    }
    .hidden { display: none; }
    label {
      display: grid;
      gap: var(--space-sm);
      color: var(--muted);
      font-size: 0.86rem;
      font-weight: 650;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 11px 12px;
      outline: none;
    }
    textarea { min-height: 112px; resize: vertical; }
    textarea:focus, input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent), transparent 78%);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: var(--space-md);
    }
    .file-zone {
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px dashed color-mix(in oklch, var(--line), var(--ink) 15%);
      border-radius: 8px;
      background: var(--panel);
      padding: var(--space-lg);
      cursor: pointer;
      text-align: center;
      color: var(--muted);
    }
    .file-zone strong { color: var(--ink); }
    .preview {
      max-width: 100%;
      max-height: 210px;
      border-radius: 6px;
      object-fit: contain;
      background: var(--field);
    }
    .actions {
      display: flex;
      gap: var(--space-md);
      align-items: center;
    }
    .primary {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 12px 16px;
      font-weight: 750;
      cursor: pointer;
      min-height: 44px;
    }
    .primary:disabled { opacity: 0.55; cursor: progress; }
    .ghost {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: transparent;
      color: var(--ink);
      padding: 11px 14px;
      cursor: pointer;
      min-height: 44px;
    }
    .hint { color: var(--muted); font-size: 0.88rem; }
    .results-wrap {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: var(--space-xl);
      min-width: 0;
    }
    .results-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: var(--space-lg);
      border-bottom: 1px solid var(--line);
      padding-bottom: var(--space-lg);
    }
    h2 {
      margin: 0;
      font-size: 1.2rem;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .result-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: var(--space-lg);
      align-content: start;
    }
    .result {
      display: grid;
      grid-template-rows: 170px auto;
      gap: var(--space-md);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: var(--space-md);
      min-width: 0;
    }
    .thumb {
      width: 100%;
      height: 170px;
      object-fit: contain;
      border-radius: 6px;
      background: var(--field);
    }
    .result-body { display: grid; gap: var(--space-sm); min-width: 0; }
    .path {
      color: var(--ink);
      font-weight: 750;
      overflow-wrap: anywhere;
      line-height: 1.25;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-xs);
      color: var(--muted);
      font-size: 0.82rem;
    }
    .score-row {
      display: grid;
      grid-template-columns: 64px 1fr auto;
      gap: var(--space-sm);
      align-items: center;
      color: var(--muted);
      font-size: 0.78rem;
    }
    .bar {
      height: 7px;
      border-radius: 999px;
      background: var(--field);
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      width: calc(var(--value) * 100%);
      background: var(--ok);
    }
    .empty {
      display: grid;
      place-items: center;
      min-height: 320px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      text-align: center;
      padding: var(--space-xl);
    }
    @media (max-width: 860px) {
      .shell { grid-template-columns: 1fr; }
      .grid-2 { grid-template-columns: 1fr; }
      .results-head { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="console" aria-label="检索控制台">
      <div class="mast">
        <h1>PicRetrieve</h1>
        <div id="status" class="status">
          <span class="pill">loading</span>
        </div>
      </div>

      <div class="tabs" role="tablist" aria-label="检索模式">
        <button id="textTab" class="tab" role="tab" aria-selected="true">文本检索</button>
        <button id="imageTab" class="tab" role="tab" aria-selected="false">图片检索</button>
      </div>

      <form id="textPanel" class="panel">
        <label>
          查询文本
          <textarea id="textQuery" placeholder="A black and white dog is running in the grass.">A black and white dog is running in the grass.</textarea>
        </label>
        <label>
          元信息关键词
          <input id="textMeta" placeholder="可选，例如 AMD Ryzen 或 invoice" />
        </label>
        <div class="grid-2">
          <label>
            Profile
            <select id="textProfile">
              <option value="general">general</option>
              <option value="document">document</option>
            </select>
          </label>
          <label>
            Top K
            <input id="textTopK" type="number" min="1" max="50" value="8" />
          </label>
        </div>
        <div class="actions">
          <button id="textButton" class="primary" type="submit">搜索</button>
          <button class="ghost" type="button" data-sample="A man wears an orange hat and glasses.">Person</button>
          <button class="ghost" type="button" data-sample="A black and white dog is running in the grass.">Dog</button>
        </div>
      </form>

      <form id="imagePanel" class="panel hidden">
        <label>
          查询图片
          <input id="imageFile" class="hidden" type="file" accept="image/*" />
          <div id="dropZone" class="file-zone" tabindex="0">
            <span><strong>选择或拖入图片</strong><br />JPG、PNG、TIFF 转换后的图片都可以</span>
          </div>
        </label>
        <label>
          元信息关键词
          <input id="imageMeta" placeholder="可选，document profile 下更有用" />
        </label>
        <div class="grid-2">
          <label>
            Profile
            <select id="imageProfile">
              <option value="general">general</option>
              <option value="document">document</option>
            </select>
          </label>
          <label>
            Top K
            <input id="imageTopK" type="number" min="1" max="50" value="8" />
          </label>
        </div>
        <div class="actions">
          <button id="imageButton" class="primary" type="submit">搜索</button>
          <span id="fileName" class="hint"></span>
        </div>
      </form>
    </section>

    <section class="results-wrap" aria-live="polite">
      <div class="results-head">
        <h2 id="resultTitle">结果</h2>
        <span id="resultHint" class="hint">等待查询</span>
      </div>
      <div id="results" class="result-grid">
        <div class="empty">输入文本或上传图片后，这里会显示缩略图、路径和融合分数。</div>
      </div>
    </section>
  </main>

  <script>
    const state = { mode: "text", busy: false };
    const $ = (id) => document.getElementById(id);

    function setMode(mode) {
      state.mode = mode;
      $("textTab").setAttribute("aria-selected", mode === "text");
      $("imageTab").setAttribute("aria-selected", mode === "image");
      $("textPanel").classList.toggle("hidden", mode !== "text");
      $("imagePanel").classList.toggle("hidden", mode !== "image");
    }

    function resultMeta(result) {
      const metadata = result.metadata || {};
      return metadata.metadata_jsonl || metadata.sidecar || {};
    }

    function scoreRow(label, value) {
      const safe = Math.max(0, Math.min(1, Number(value || 0)));
      return `<div class="score-row"><span>${label}</span><div class="bar"><span style="--value:${safe}"></span></div><span>${safe.toFixed(3)}</span></div>`;
    }

    function renderResults(payload, elapsedMs) {
      const results = payload.results || [];
      $("resultTitle").textContent = `${payload.query_type === "image" ? "图片" : "文本"}检索结果`;
      $("resultHint").textContent = `${results.length} 条 · ${elapsedMs} ms`;
      if (!results.length) {
        $("results").innerHTML = `<div class="empty">没有结果。</div>`;
        return;
      }
      $("results").innerHTML = results.map((result) => {
        const meta = resultMeta(result);
        const title = meta.title || result.rel_path;
        const category = meta.category || "";
        const brand = meta.brand || "";
        return `
          <article class="result">
            <img class="thumb" src="/files/${result.id}" alt="${result.rel_path}">
            <div class="result-body">
              <div class="path">${title}</div>
              <div class="meta">
                <span class="pill">${category || "item"}</span>
                ${brand ? `<span class="pill">${brand}</span>` : ""}
              </div>
              <div class="hint">${result.rel_path}</div>
              ${scoreRow("final", result.score)}
              ${scoreRow("visual", result.visual_score)}
              ${scoreRow("meta", result.metadata_score)}
            </div>
          </article>
        `;
      }).join("");
    }

    async function requestJson(url, options) {
      const started = performance.now();
      const response = await fetch(url, options);
      const elapsed = Math.round(performance.now() - started);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || response.statusText);
      }
      renderResults(payload, elapsed);
    }

    function setBusy(button, busy) {
      state.busy = busy;
      button.disabled = busy;
      button.textContent = busy ? "搜索中" : "搜索";
    }

    async function loadHealth() {
      try {
        const response = await fetch("/health");
        const health = await response.json();
        $("status").innerHTML = `
          <span class="pill">${health.items} items</span>
          <span class="pill">${health.device}</span>
          <span class="pill">${health.model}</span>
        `;
      } catch {
        $("status").innerHTML = `<span class="pill">offline</span>`;
      }
    }

    $("textTab").addEventListener("click", () => setMode("text"));
    $("imageTab").addEventListener("click", () => setMode("image"));

    document.querySelectorAll("[data-sample]").forEach((button) => {
      button.addEventListener("click", () => {
        $("textQuery").value = button.dataset.sample;
        setMode("text");
      });
    });

    $("textPanel").addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = $("textButton");
      setBusy(button, true);
      try {
        await requestJson("/search/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: $("textQuery").value,
            metadata_query: $("textMeta").value || null,
            profile: $("textProfile").value,
            top_k: Number($("textTopK").value || 8)
          })
        });
      } catch (error) {
        $("results").innerHTML = `<div class="empty">${error.message}</div>`;
      } finally {
        setBusy(button, false);
      }
    });

    function showFile(file) {
      if (!file) return;
      $("fileName").textContent = file.name;
      const url = URL.createObjectURL(file);
      $("dropZone").innerHTML = `<img class="preview" src="${url}" alt="${file.name}">`;
    }

    $("dropZone").addEventListener("click", () => $("imageFile").click());
    $("dropZone").addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") $("imageFile").click();
    });
    $("dropZone").addEventListener("dragover", (event) => event.preventDefault());
    $("dropZone").addEventListener("drop", (event) => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) {
        const box = new DataTransfer();
        box.items.add(file);
        $("imageFile").files = box.files;
        showFile(file);
      }
    });
    $("imageFile").addEventListener("change", () => showFile($("imageFile").files[0]));

    $("imagePanel").addEventListener("submit", async (event) => {
      event.preventDefault();
      const file = $("imageFile").files[0];
      if (!file) {
        $("results").innerHTML = `<div class="empty">请选择一张查询图片。</div>`;
        return;
      }
      const button = $("imageButton");
      setBusy(button, true);
      const form = new FormData();
      form.append("file", file);
      form.append("top_k", $("imageTopK").value || "8");
      form.append("profile", $("imageProfile").value);
      if ($("imageMeta").value) form.append("metadata_query", $("imageMeta").value);
      try {
        await requestJson("/search/image", { method: "POST", body: form });
      } catch (error) {
        $("results").innerHTML = `<div class="empty">${error.message}</div>`;
      } finally {
        setBusy(button, false);
      }
    });

    loadHealth();
  </script>
</body>
</html>
"""


class TextSearchRequest(BaseModel):
    """! @brief 文本检索请求体。"""

    text: str = Field(min_length=1)
    metadata_query: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    profile: str = "general"
    candidate_k: int = Field(default=200, ge=1, le=1000)


def get_store() -> IndexStore:
    """! @brief 懒加载 SQLite/NumPy 索引存储。"""

    global _store
    if _store is None:
        _store = IndexStore(settings.data_dir)
        _store.create_tables()
    return _store


def get_embedder() -> ClipEmbedder:
    """! @brief 懒加载 CLIP 模型，避免 API import 时就下载模型。"""

    global _embedder
    if _embedder is None:
        _embedder = ClipEmbedder(settings.model_name, settings.device)
    return _embedder


def get_retriever() -> Retriever:
    """! @brief 懒加载混合检索器。"""

    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_store(), get_embedder())
    return _retriever


def ensure_profile(profile: str) -> str:
    """! @brief 校验检索 profile。"""

    if profile not in PROFILES:
        raise HTTPException(status_code=400, detail=f"unknown profile: {profile}")
    return profile


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """! @brief 返回本地试用网页。"""

    return INDEX_HTML


@app.get("/favicon.ico", status_code=204)
def favicon() -> Response:
    """! @brief 避免浏览器默认 favicon 请求产生 404 噪声。"""

    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    """! @brief 返回服务、设备、模型和索引条目状态。"""

    device = str(_embedder.device) if _embedder is not None else select_device(settings.device)
    return {
        "status": "ok",
        "device": device,
        "model": settings.model_name,
        "items": get_store().count_items(),
    }


@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    profile: str = Form("general"),
    metadata_query: str | None = Form(None),
    candidate_k: int = Form(200),
) -> dict[str, Any]:
    """! @brief 上传查询图片并返回相似图片列表。"""

    ensure_profile(profile)
    try:
        payload = await file.read()
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    try:
        results = get_retriever().search_by_image(
            image=image,
            top_k=top_k,
            profile=profile,
            metadata_query=metadata_query,
            candidate_k=candidate_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query_type": "image", "top_k": top_k, "results": results}


@app.post("/search/text")
def search_text(request: TextSearchRequest) -> dict[str, Any]:
    """! @brief 文本到图片检索。"""

    ensure_profile(request.profile)
    try:
        results = get_retriever().search_by_text(
            text=request.text,
            top_k=request.top_k,
            profile=request.profile,
            metadata_query=request.metadata_query,
            candidate_k=request.candidate_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query_type": "text", "top_k": request.top_k, "results": results}


@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict[str, Any]:
    """! @brief 返回单个 item 的完整元信息。"""

    item = get_store().get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item


@app.get("/files/{item_id}")
def get_file(item_id: int) -> FileResponse:
    """! @brief 返回图片文件，并校验路径仍位于允许根目录内。"""

    item = get_store().get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    path = Path(item["path"]).expanduser().resolve()
    image_root = resolve_image_root(settings)
    if not path.is_relative_to(image_root):
        raise HTTPException(status_code=403, detail="file is outside allowed image root")
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(path)
