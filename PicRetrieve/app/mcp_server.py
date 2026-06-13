"""PicRetrieve MCP Server — 可在 LM Studio / Cherry Studio 中作为工具调用。

启动方式:
    cd PicRetrieve
    PICRETRIEVE_DATA_DIR=data/flickr30k_index \\
    PICRETRIEVE_IMAGE_ROOT=data/flickr30k_test \\
    uv run python app/mcp_server.py

然后在 LM Studio 中连接:
    Settings → MCP Server → 添加 → 类型 stdio → 命令填入上面的 uv run ...
"""

from __future__ import annotations

import base64
import io

from mcp.server import FastMCP
from PIL import Image

from app.config import load_settings
from app.embedder import ClipEmbedder
from app.index_store import IndexStore
from app.retrieval import Retriever

settings = load_settings()
store = IndexStore(settings.data_dir)
embedder = ClipEmbedder(settings.model_name, settings.device)
retriever = Retriever(store, embedder)

server = FastMCP("PicRetrieve")


@server.tool()
def search_text(text: str, top_k: int = 5) -> str:
    """以文搜图：输入文本描述，返回相似图片列表。

    Args:
        text: 查询文本（支持中文），如"一个戴橙色帽子的男人"
        top_k: 返回结果数量，默认 5，最大 20
    """
    results = retriever.search_by_text(text, top_k=min(top_k, 20))
    return _format_results(results)


@server.tool()
def search_image(image_data: str, top_k: int = 5) -> str:
    """以图搜图：输入 base64 编码的图片数据，返回相似图片列表。

    Args:
        image_data: base64 编码的图片数据（不含 data:image/ 前缀）
        top_k: 返回结果数量，默认 5，最大 20
    """
    raw = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    results = retriever.search_by_image(image, top_k=min(top_k, 20))
    return _format_results(results)


def _format_results(results: list[dict]) -> str:
    lines = ["## 检索结果\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"].get("metadata_jsonl", {}).get("metadata", {})
        caption = (meta.get("captions") or [""])[0]
        lines.append(
            f"### #{i}  {r['rel_path']}\n"
            f"- 综合分: {r['score']:.4f}\n"
            f"- 视觉分: {r['visual_score']:.4f}\n"
            f"- 元信息分: {r['metadata_score']:.4f}\n"
            f"- 描述: {caption[:80]}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    server.run(transport="stdio")
