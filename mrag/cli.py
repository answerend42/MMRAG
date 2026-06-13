"""MMRAG CLI — 多模态 RAG 问答。

用法:
  mrag "你的问题"

环境变量:
  MRAG_LLM_API_KEY   LLM API 密钥（可选，无 key 时展示检索结果）
  MRAG_LLM_MODEL     模型名（默认 gpt-4o-mini）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mrag.models.evidence import Modality
from mrag.retrievers.base import Retriever

logger = logging.getLogger("mrag")


def _log(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger("mrag")
    root.addHandler(h)
    root.setLevel(level)


def _resolve(p: str) -> Path:
    """解析路径，支持相对于项目根或 CWD。"""
    p = Path(p)
    if p.exists():
        return p.resolve()
    for base in [Path.cwd(), Path.cwd().parent]:
        c = base / p
        if c.exists():
            return c.resolve()
    return p.resolve()


def _build_registry():
    """初始化所有召回器，返回注册表 dict[Modality, Retriever]。"""
    registry: dict[Modality, Retriever] = {}

    # ── CLIP 编码器（文本和图片共享） ──
    model_dir = _resolve("PicRetrieve/data/models/openai_clip-vit-base-patch32")
    data_dir = _resolve("PicRetrieve/data")

    if not model_dir.exists() or not data_dir.exists():
        logger.warning("未找到 CLIP 模型或图片索引，跳过检索初始化")
        return registry

    from app.embedder import ClipEmbedder
    from app.index_store import IndexStore

    embedder = ClipEmbedder(model_name=str(model_dir))
    store = IndexStore(data_dir)
    store.create_tables()
    logger.info("CLIP 就绪: %d 张图片", store.count_items())

    # ── ImageRetriever ──
    from mrag.retrievers import ImageRetriever

    img = ImageRetriever(embedder, store)
    registry[Modality.IMAGE] = img
    logger.info("  ImageRetriever 注册")

    # ── TextRetriever ──
    from mrag.retrievers import TextRetriever

    txt = TextRetriever()
    txt.bind_embedder(embedder)

    docs_dir = _resolve("docs")
    if docs_dir.exists():
        n = txt.load(str(docs_dir))
        logger.info("  TextRetriever 注册: %d 个文本块", n)
    else:
        logger.warning("  未找到 docs/ 目录，文本检索不可用")

    registry[Modality.TEXT] = txt

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="多模态 RAG 问答系统")
    parser.add_argument("query", nargs="?", help="你的问题")
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    parser.add_argument("--json", action="store_true", help="JSON 格式")
    args = parser.parse_args()

    _log(args.verbose)

    query = args.query or input("请输入问题: ").strip()
    if not query:
        print("问题不能为空")
        sys.exit(1)

    # LLM
    from mrag.utils.llm import LLMClient

    llm = None
    try:
        llm = LLMClient()
    except Exception as e:
        logger.warning("LLM 不可用: %s", e)

    registry = _build_registry()

    from mrag.graph import build_graph
    from mrag.state import MMRAGState

    graph = build_graph(registry=registry, llm=llm)
    result = graph.invoke(MMRAGState(query=query))

    def get(k):
        return result[k] if isinstance(result, dict) else getattr(result, k)

    if args.json:
        out = {
            "query": get("query"),
            "answer": get("answer"),
            "confident": get("is_confident"),
            "reason": get("confidence_reason"),
            "evidence": len(get("evidence_cards")),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print()
    print("=" * 60)
    print(f"Q: {get('query')}")
    print("=" * 60)
    print(f"\n{get('answer')}")
    icon = "✅" if get("is_confident") else "⚠️"
    print(f"\n{icon} {get('confidence_reason')}")

    if args.verbose:
        print("\n执行轨迹:")
        for s in get("trace"):
            print(f"  • {s}")


if __name__ == "__main__":
    main()
