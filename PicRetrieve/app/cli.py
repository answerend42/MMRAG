"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from app.embedder import ClipEmbedder
from app.index_store import IndexStore
from app.metadata import ImageItem, scan_images
from app.retrieval import PROFILES, Retriever

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """! @brief 配置 CLI 日志输出格式。"""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    root = logging.getLogger("picretrieve")
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def print_json(payload: dict[str, Any]) -> None:
    """! @brief 以中文友好的 JSON 格式打印 CLI 结果。"""

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def infer_metadata_path(image_dir: Path, metadata_path: Path | None) -> Path | None:
    """! @brief 未显式传入 metadata 时，尝试发现 bootstrap 输出的 metadata.jsonl。"""

    if metadata_path is not None:
        return metadata_path
    candidate = image_dir.expanduser().resolve().parent / "metadata.jsonl"
    return candidate if candidate.exists() else None


def load_batch_images(items: list[ImageItem]) -> list[Image.Image]:
    """! @brief 为一个索引 batch 打开图片并统一转 RGB。"""

    images: list[Image.Image] = []
    for item in items:
        with Image.open(item.path) as image:
            images.append(image.convert("RGB"))
    return images


def run_index(args: argparse.Namespace) -> None:
    """! @brief 执行离线索引：扫描元信息、编码 CLIP 向量、保存 SQLite/NumPy。"""

    image_dir = args.image_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    metadata_path = infer_metadata_path(image_dir, args.metadata)

    store = IndexStore(data_dir)
    if args.reset:
        logger.info("Resetting index at %s", data_dir)
        store.reset()
    else:
        store.create_tables()

    items = scan_images(image_dir, metadata_path, args.ocr)
    if not items:
        store.save_embeddings(np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64))
        (data_dir / "image_root.txt").write_text(str(image_dir), encoding="utf-8")
        logger.info("No images found at %s", image_dir)
        print_json({"indexed": 0, "data_dir": str(data_dir), "image_root": str(image_dir)})
        return

    embedder = ClipEmbedder(args.model_name, args.device)
    logger.info("Indexing %d images from %s", len(items), image_dir)
    all_embeddings: list[np.ndarray] = []
    all_ids: list[int] = []

    for start in tqdm(range(0, len(items), args.batch_size), desc="index", unit="batch"):
        batch_items = items[start : start + args.batch_size]
        item_ids = [store.upsert_item(item) for item in batch_items]
        images = load_batch_images(batch_items)
        embeddings = embedder.encode_images(images, batch_size=args.batch_size)
        all_ids.extend(item_ids)
        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    ids = np.asarray(all_ids, dtype=np.int64)
    store.save_embeddings(embeddings, ids)
    (data_dir / "image_root.txt").write_text(str(image_dir), encoding="utf-8")
    logger.info("Indexed %d images (shape %s)", len(ids), list(embeddings.shape))
    print_json(
        {
            "indexed": len(ids),
            "embedding_shape": list(embeddings.shape),
            "data_dir": str(data_dir),
            "image_root": str(image_dir),
            "metadata": str(metadata_path) if metadata_path else None,
        }
    )


def run_search_image(args: argparse.Namespace) -> None:
    """! @brief 执行 CLI 以图搜图。"""

    ensure_profile(args.profile)
    store = IndexStore(args.data_dir)
    embedder = ClipEmbedder(args.model_name, args.device)
    retriever = Retriever(store, embedder)
    with Image.open(args.image) as image:
        results = retriever.search_by_image(
            image=image.convert("RGB"),
            top_k=args.top_k,
            profile=args.profile,
            metadata_query=args.metadata_query,
            candidate_k=args.candidate_k,
        )
    print_json({"query_type": "image", "top_k": args.top_k, "results": results})


def run_search_text(args: argparse.Namespace) -> None:
    """! @brief 执行 CLI 文本到图片检索。"""

    ensure_profile(args.profile)
    store = IndexStore(args.data_dir)
    embedder = ClipEmbedder(args.model_name, args.device)
    retriever = Retriever(store, embedder)
    results = retriever.search_by_text(
        text=args.text,
        top_k=args.top_k,
        profile=args.profile,
        metadata_query=args.metadata_query,
        candidate_k=args.candidate_k,
    )
    print_json({"query_type": "text", "top_k": args.top_k, "results": results})


def ensure_profile(profile: str) -> None:
    """! @brief 校验 profile 名称。"""

    if profile not in PROFILES:
        raise SystemExit(f"unknown profile: {profile}; choose one of {', '.join(PROFILES)}")


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建 PicRetrieve CLI 参数解析器。"""

    parser = argparse.ArgumentParser(prog="picretrieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="build local image index")
    index.add_argument(
        "--image-dir", "--root", dest="image_dir", type=Path, default=Path("samples")
    )
    index.add_argument("--metadata", type=Path, default=None)
    index.add_argument("--data-dir", type=Path, default=Path("data"))
    index.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    index.add_argument("--device", default=None)
    index.add_argument("--batch-size", type=int, default=16)
    index.add_argument("--ocr", choices=["none", "tesseract"], default="none")
    index.add_argument("--reset", action="store_true")
    index.set_defaults(func=run_index)

    parser.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")

    search_image = subparsers.add_parser("search-image", help="search by query image")
    search_image.add_argument("--image", type=Path, required=True)
    search_image.add_argument("--data-dir", type=Path, default=Path("data"))
    search_image.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    search_image.add_argument("--device", default=None)
    search_image.add_argument("--top-k", type=int, default=10)
    search_image.add_argument("--candidate-k", type=int, default=200)
    search_image.add_argument("--profile", choices=sorted(PROFILES), default="general")
    search_image.add_argument("--metadata-query", default=None)
    search_image.set_defaults(func=run_search_image)

    search_text = subparsers.add_parser("search-text", help="search by text")
    search_text.add_argument("--text", required=True)
    search_text.add_argument("--data-dir", type=Path, default=Path("data"))
    search_text.add_argument("--model-name", default="data/models/openai_clip-vit-base-patch32")
    search_text.add_argument("--device", default=None)
    search_text.add_argument("--top-k", type=int, default=10)
    search_text.add_argument("--candidate-k", type=int, default=200)
    search_text.add_argument("--profile", choices=sorted(PROFILES), default="general")
    search_text.add_argument("--metadata-query", default=None)
    search_text.set_defaults(func=run_search_text)
    return parser


def main() -> None:
    """! @brief CLI 主入口。"""

    parser = build_parser()
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
