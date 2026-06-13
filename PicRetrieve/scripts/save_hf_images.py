#!/usr/bin/env python
"""通用 Hugging Face 图片数据集保存脚本。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from PIL import Image


def slug(value: Any) -> str:
    """! @brief 生成目录安全的标签名。"""

    text = str(value or "all").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_-]+", "-", text)
    return text.strip("-") or "all"


def label_text(dataset: Any, column: str | None, value: Any) -> str:
    """! @brief 把可选 label 值转换成可读目录名。"""

    if column is None:
        return "all"
    try:
        feature = dataset.features.get(column)
        names = getattr(feature, "names", None)
        if names and isinstance(value, int):
            return str(names[value])
    except Exception:
        pass
    return str(value)


def save_hf_images(args: argparse.Namespace) -> None:
    """! @brief 读取 HF dataset 并把 image column 保存为普通 JPG 文件夹。"""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("missing datasets; run: uv sync --extra data") from exc

    dataset = load_dataset(args.dataset, split=args.split)
    out = args.out.expanduser().resolve()
    saved = 0
    for idx, row in enumerate(dataset):
        image = row.get(args.image_column)
        if not isinstance(image, Image.Image):
            continue
        label = label_text(dataset, args.label_column, row.get(args.label_column))
        folder = out / slug(label)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{idx:06d}.jpg"
        image.convert("RGB").save(path, format="JPEG", quality=92)
        saved += 1
    print(f"saved {saved} images to {out}")


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建保存脚本参数解析器。"""

    parser = argparse.ArgumentParser(prog="save_hf_images.py")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train[:100]")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--out", type=Path, default=Path("samples/hf"))
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    parser = build_parser()
    save_hf_images(parser.parse_args())


if __name__ == "__main__":
    main()
