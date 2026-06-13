"""图片元信息扫描。"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(slots=True)
class ImageItem:
    """! @brief 一张被索引图片的结构化元信息。"""

    path: Path
    rel_path: str
    filename: str
    folder: str
    ext: str
    width: int | None
    height: int | None
    size_bytes: int
    mtime: float
    metadata: dict[str, Any]
    metadata_text: str
    ocr_text: str


def scan_image_paths(image_root: Path) -> list[Path]:
    """! @brief 按稳定顺序扫描图片路径。"""

    root = image_root.expanduser().resolve()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def flatten_mapping(value: Any) -> str:
    """! @brief 把嵌套 JSON/字典压平成适合 FTS 的短文本。"""

    parts: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                parts.append(str(key))
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif node is not None:
            parts.append(str(node))

    visit(value)
    return " ".join(part for part in parts if part)


def load_metadata_jsonl(metadata_jsonl: Path | None, image_root: Path) -> dict[str, dict[str, Any]]:
    """! @brief 加载 `metadata.jsonl`，并建立多种路径键便于合并。"""

    if metadata_jsonl is None:
        return {}
    metadata_path = metadata_jsonl.expanduser().resolve()
    if not metadata_path.exists():
        return {}

    root = image_root.expanduser().resolve()
    corpus_root = metadata_path.parent
    records: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_name = str(record.get("file_name") or "")
            if not file_name:
                continue
            normalized = file_name.replace("\\", "/")
            records[normalized] = record
            if normalized.startswith("images/"):
                records[normalized[len("images/") :]] = record
            abs_path = (corpus_root / normalized).resolve()
            records[str(abs_path)] = record
            with suppress(ValueError):
                records[str(abs_path.relative_to(root)).replace("\\", "/")] = record
    return records


def read_sidecar(image_path: Path) -> tuple[dict[str, Any], str]:
    """! @brief 读取同名 JSON/TXT/MD sidecar 元信息。"""

    sidecar_json: dict[str, Any] = {}
    sidecar_text_parts: list[str] = []
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            sidecar_json = loaded if isinstance(loaded, dict) else {"value": loaded}
        except (OSError, json.JSONDecodeError):
            sidecar_json = {}

    for suffix in (".txt", ".md"):
        text_path = image_path.with_suffix(suffix)
        if text_path.exists():
            try:
                sidecar_text_parts.append(text_path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return sidecar_json, "\n".join(sidecar_text_parts).strip()


def read_exif(image: Image.Image) -> dict[str, Any]:
    """! @brief 从 Pillow 图片对象读取可读 EXIF 字段。"""

    try:
        raw_exif = image.getexif()
    except Exception:
        return {}
    exif: dict[str, Any] = {}
    for key, value in raw_exif.items():
        name = ExifTags.TAGS.get(key, str(key))
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        exif[str(name)] = value
    return exif


def run_optional_ocr(image: Image.Image, ocr_mode: str) -> str:
    """! @brief 可选 OCR；依赖缺失时静默返回空字符串。"""

    if ocr_mode == "none":
        return ""
    if ocr_mode != "tesseract":
        return ""
    if shutil.which("tesseract") is None:
        return ""
    try:
        import pytesseract

        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


def find_extra_record(
    image_path: Path,
    image_root: Path,
    metadata_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """! @brief 按绝对路径、相对路径和 `images/` 前缀匹配外部元信息。"""

    if not metadata_records:
        return {}
    abs_key = str(image_path.resolve())
    if abs_key in metadata_records:
        return metadata_records[abs_key]
    rel_key = str(image_path.resolve().relative_to(image_root.resolve())).replace("\\", "/")
    return metadata_records.get(rel_key) or metadata_records.get(f"images/{rel_key}") or {}


def build_metadata_text(
    item_core: Iterable[str],
    extra_record: dict[str, Any],
    sidecar_json: dict[str, Any],
    sidecar_text: str,
    ocr_text: str,
) -> str:
    """! @brief 拼接目录、文件名、sidecar、metadata.jsonl 和 OCR 文本。"""

    pieces = list(item_core)
    if "index_text" in extra_record:
        pieces.append(str(extra_record.get("index_text") or ""))
    else:
        pieces.append(str(extra_record.get("text") or ""))
        pieces.append(flatten_mapping(extra_record.get("metadata") or {}))
    pieces.append(flatten_mapping(sidecar_json))
    pieces.append(sidecar_text)
    pieces.append(ocr_text)
    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def scan_image(
    image_path: Path,
    image_root: Path,
    metadata_records: dict[str, dict[str, Any]] | None = None,
    ocr_mode: str = "none",
) -> ImageItem | None:
    """! @brief 扫描单张图片并合并文件系统、Pillow、sidecar 和外部元信息。"""

    root = image_root.expanduser().resolve()
    path = image_path.expanduser().resolve()
    try:
        stat = path.stat()
        rel_path = str(path.relative_to(root)).replace("\\", "/")
    except (OSError, ValueError):
        return None

    sidecar_json, sidecar_text = read_sidecar(path)
    extra_record = find_extra_record(path, root, metadata_records or {})

    width: int | None = None
    height: int | None = None
    exif: dict[str, Any] = {}
    ocr_text = ""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            exif = read_exif(image)
            ocr_text = run_optional_ocr(image, ocr_mode)
    except Exception:
        return None

    folder = str(Path(rel_path).parent)
    if folder == ".":
        folder = ""
    metadata = {
        "folder": folder,
        "exif": exif,
        "sidecar": sidecar_json,
        "metadata_jsonl": extra_record,
    }
    metadata_text = build_metadata_text(
        [folder, path.name, rel_path],
        extra_record,
        sidecar_json,
        sidecar_text,
        ocr_text,
    )
    return ImageItem(
        path=path,
        rel_path=rel_path,
        filename=path.name,
        folder=folder,
        ext=path.suffix.lower(),
        width=width,
        height=height,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        metadata=metadata,
        metadata_text=metadata_text,
        ocr_text=ocr_text,
    )


def scan_images(
    image_root: Path,
    metadata_jsonl: Path | None = None,
    ocr_mode: str = "none",
) -> list[ImageItem]:
    """! @brief 扫描图片根目录，返回可直接写入索引的元信息列表。"""

    root = image_root.expanduser().resolve()
    metadata_records = load_metadata_jsonl(metadata_jsonl, root)
    items: list[ImageItem] = []
    for path in scan_image_paths(root):
        item = scan_image(path, root, metadata_records, ocr_mode)
        if item is not None:
            items.append(item)
    return items
