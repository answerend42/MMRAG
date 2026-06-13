#!/usr/bin/env python
"""公开数据集到本地 ImageFolder 的轻量引导脚本。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import tarfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

LICENSE_NOTE = (
    "For local research/MVP only; do not redistribute downloaded images without checking source terms."
)
DEFAULT_HARDWARE_CATEGORIES = [
    "cpu",
    "video-card",
    "internal-hard-drive",
    "motherboard",
    "memory",
]
IMAGENETTE_LABELS = {
    "n01440764": "tench fish",
    "n02102040": "English springer dog",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def slug(value: Any) -> str:
    """! @brief 生成适合目录名的短 slug。"""

    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_-]+", "-", text)
    return text.strip("-") or "unknown"


def safe_image_id(row: dict[str, Any]) -> str:
    """! @brief 从数据行生成稳定图片 id。"""

    raw = str(
        row.get("source_id")
        or row.get("product_tag")
        or row.get("id")
        or row.get("name")
        or row.get("image_url")
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_specs(specs: Any) -> dict[str, Any]:
    """! @brief 把 specs 字段规整成 dict。"""

    if isinstance(specs, dict):
        return specs
    if isinstance(specs, str):
        try:
            loaded = json.loads(specs)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {"specs": specs}
    return {}


def flatten_specs(specs: dict[str, Any] | None) -> str:
    """! @brief 把规格字典压平成文本。"""

    if not isinstance(specs, dict):
        return ""
    parts: list[str] = []
    for key, value in specs.items():
        if value is not None:
            parts.append(f"{key} {value}")
    return " ".join(parts)


def build_hardware_text(row: dict[str, Any]) -> str:
    """! @brief 拼接硬件样本的 FTS 辅助文本。"""

    specs = parse_specs(row.get("specs"))
    parts = [
        row.get("name", ""),
        row.get("category", ""),
        row.get("brand", ""),
        flatten_specs(specs),
    ]
    return " ".join(str(part) for part in parts if part)


def first_url(value: Any) -> str:
    """! @brief 从字符串或列表字段取第一个 URL。"""

    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def download_image(url: str, max_size: int) -> Image.Image | None:
    """! @brief 下载并校验图片，统一转 RGB 且限制最长边。"""

    if not url:
        return None
    try:
        import requests

        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "picretrieve/0.1"},
        )
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.thumbnail((max_size, max_size))
        return image
    except Exception:
        return None


def download_hf_dataset_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """! @brief 通过 HF 镜像入口下载数据集文件到本地缓存。"""

    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo_id}/resolve/main/{filename}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / repo_id.replace("/", "__") / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "picretrieve/0.1"}) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return path


def save_jpg(image: Image.Image, path: Path) -> None:
    """! @brief 保存 RGB JPG，并确保父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92, optimize=True)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """! @brief 追加写入 metadata.jsonl。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_seen_metadata(path: Path) -> set[str]:
    """! @brief 读取已有 metadata.jsonl 的 file_name，用于续跑去重。"""

    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_name = record.get("file_name")
            if file_name:
                seen.add(str(file_name))
    return seen


def write_sidecar(image_path: Path, record: dict[str, Any]) -> None:
    """! @brief 为图片写同名 sidecar JSON。"""

    image_path.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """! @brief 读取本地 JSONL 行，跳过坏行。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def select_rows_by_category(
    rows: Iterable[dict[str, Any]],
    categories: list[str],
    per_category: int,
) -> list[dict[str, Any]]:
    """! @brief 按类别和每类上限选择数据行。"""

    targets = set(categories)
    counts = {category: 0 for category in categories}
    selected: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category") or "").strip()
        if category not in targets or counts[category] >= per_category:
            continue
        selected.append(row)
        counts[category] += 1
        if all(counts[category] >= per_category for category in categories):
            break
    return selected


def build_hardware_record(row: dict[str, Any], out: Path, max_image_size: int) -> dict[str, Any] | None:
    """! @brief 下载或复用单条硬件图片，返回 metadata 记录。"""

    dataset_id = "Doshiba/pcpartpicker-parts-dataset"
    category = str(row.get("category") or "").strip()
    image_url = first_url(row.get("image_url"))
    source_id = safe_image_id(row)
    rel_path = Path("images") / "pcpartpicker" / slug(category) / f"{source_id}.jpg"
    image_path = out / rel_path
    sidecar_path = image_path.with_suffix(".json")
    if image_path.exists() and sidecar_path.exists():
        try:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    image = download_image(image_url, max_image_size)
    if image is None:
        return None
    save_jpg(image, image_path)

    specs = parse_specs(row.get("specs"))
    record = {
        "file_name": rel_path.as_posix(),
        "dataset": dataset_id,
        "category": category,
        "title": row.get("name") or "",
        "brand": row.get("brand") or "",
        "source_id": source_id,
        "source_url": row.get("url") or "",
        "image_url": image_url,
        "text": build_hardware_text(row),
        "metadata": {
            "price_eur": row.get("price_eur"),
            "rating_count": row.get("rating_count"),
            "specs": specs,
        },
        "license_note": LICENSE_NOTE,
    }
    write_sidecar(image_path, record)
    return record


def run_hardware(args: argparse.Namespace) -> None:
    """! @brief 下载 PCPartPicker 硬件图片和元信息。"""

    try:
        import requests  # noqa: F401
    except ImportError as exc:
        raise SystemExit("missing requests; run: uv sync --extra data") from exc

    random.seed(args.seed)
    dataset_id = "Doshiba/pcpartpicker-parts-dataset"
    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    data_file = download_hf_dataset_file(
        dataset_id,
        "pcpartpicker_parts.jsonl",
        out / ".cache",
    )
    rows = load_jsonl_rows(data_file)
    if args.seed is not None and not args.no_shuffle:
        random.shuffle(rows)
    selected = select_rows_by_category(rows, args.categories, args.per_category)
    seen = load_seen_metadata(metadata_path)
    counts = {category: 0 for category in args.categories}
    written = 0
    buffer: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(build_hardware_record, row, out, args.max_image_size)
            for row in selected
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="hardware", unit="img"):
            record = future.result()
            if record is None:
                continue
            category = str(record.get("category") or "")
            counts[category] = counts.get(category, 0) + 1
            written += 1
            file_name = str(record.get("file_name") or "")
            if file_name and file_name not in seen:
                seen.add(file_name)
                buffer.append(record)
            if len(buffer) >= 200:
                append_jsonl(metadata_path, buffer)
                buffer.clear()
    if buffer:
        append_jsonl(metadata_path, buffer)
    print(json.dumps({"written": written, "selected": len(selected), "counts": counts, "out": str(out)}, ensure_ascii=False))


def label_name(dataset: Any, value: Any) -> str:
    """! @brief 把 RVL-CDIP 数字 label 转成可读标签。"""

    try:
        feature = dataset.features.get("label")
        names = getattr(feature, "names", None)
        if names and isinstance(value, int):
            return str(names[value])
    except Exception:
        pass
    return str(value)


def find_image(row: dict[str, Any]) -> Image.Image | None:
    """! @brief 从数据行中找到 PIL 图片字段。"""

    for key in ("image", "img", "page_image"):
        value = row.get(key)
        if isinstance(value, Image.Image):
            return value
    for value in row.values():
        if isinstance(value, Image.Image):
            return value
    return None


def run_documents(args: argparse.Namespace) -> None:
    """! @brief 下载 RVL-CDIP small-200 文档图片。"""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("missing datasets; run: uv sync --extra data") from exc

    dataset_id = "vaclavpechtor/rvl_cdip-small-200"
    try:
        dataset = load_dataset(dataset_id, split="train")
    except FileNotFoundError:
        run_documents_from_hub_files(args, dataset_id)
        return
    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    counts: dict[str, int] = {}
    written: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        label = label_name(dataset, row.get("label", "unknown"))
        label_slug = slug(label)
        if counts.get(label_slug, 0) >= args.per_label:
            continue
        image = find_image(row)
        if image is None:
            continue
        image = image.convert("RGB")
        image.thumbnail((args.max_image_size, args.max_image_size))
        source_id = f"{idx:06d}"
        rel_path = Path("images") / "rvl_cdip" / label_slug / f"{source_id}.jpg"
        image_path = out / rel_path
        save_jpg(image, image_path)
        text = f"scanned {label} document rvl_cdip"
        record = {
            "file_name": rel_path.as_posix(),
            "dataset": dataset_id,
            "category": label,
            "title": f"RVL-CDIP {label} {source_id}",
            "brand": "",
            "source_id": source_id,
            "source_url": "",
            "image_url": "",
            "text": text,
            "metadata": {"label": label},
            "license_note": LICENSE_NOTE,
        }
        write_sidecar(image_path, record)
        written.append(record)
        counts[label_slug] = counts.get(label_slug, 0) + 1

    append_jsonl(metadata_path, written)
    print(json.dumps({"written": len(written), "counts": counts, "out": str(out)}, ensure_ascii=False))


def run_documents_from_hub_files(args: argparse.Namespace, dataset_id: str) -> None:
    """! @brief 当 datasets 无法自动加载时，按 Hub 文件树直接抽样文档图片。"""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("missing huggingface_hub; run: uv sync --extra data") from exc

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    api = HfApi(endpoint=endpoint)
    info = api.dataset_info(dataset_id)
    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    counts: dict[str, int] = {}
    seen = load_seen_metadata(metadata_path)
    written = 0
    buffer: list[dict[str, Any]] = []
    candidates = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("train/")
        and Path(sibling.rfilename).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    selected: list[str] = []
    selected_counts: dict[str, int] = {}
    for filename in candidates:
        parts = Path(filename).parts
        if len(parts) < 3:
            continue
        label = parts[1]
        label_slug = slug(label)
        if selected_counts.get(label_slug, 0) >= args.per_label:
            continue
        selected.append(filename)
        selected_counts[label_slug] = selected_counts.get(label_slug, 0) + 1

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(build_document_record, filename, out, dataset_id, endpoint, args.max_image_size)
            for filename in selected
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="documents", unit="img"):
            try:
                record = future.result()
            except Exception:
                continue
            if record is None:
                continue
            label_slug = slug(record.get("category"))
            counts[label_slug] = counts.get(label_slug, 0) + 1
            written += 1
            file_name = str(record.get("file_name") or "")
            if file_name and file_name not in seen:
                seen.add(file_name)
                buffer.append(record)
            if len(buffer) >= 100:
                append_jsonl(metadata_path, buffer)
                buffer.clear()
    if buffer:
        append_jsonl(metadata_path, buffer)
    print(json.dumps({"written": written, "selected": len(selected), "counts": counts, "out": str(out)}, ensure_ascii=False))


def build_document_record(
    filename: str,
    out: Path,
    dataset_id: str,
    endpoint: str,
    max_image_size: int,
) -> dict[str, Any] | None:
    """! @brief 下载或复用单张 RVL-CDIP 图片，返回 metadata 记录。"""

    parts = Path(filename).parts
    if len(parts) < 3:
        return None
    label = parts[1]
    label_slug = slug(label)
    source_id = slug(Path(filename).stem)
    rel_path = Path("images") / "rvl_cdip" / label_slug / f"{source_id}.jpg"
    image_path = out / rel_path
    sidecar_path = image_path.with_suffix(".json")
    if image_path.exists() and sidecar_path.exists():
        try:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    local_file = download_hf_dataset_file(dataset_id, filename, out / ".cache")
    try:
        with Image.open(local_file) as image:
            image = image.convert("RGB")
            image.thumbnail((max_image_size, max_image_size))
    except Exception:
        return None
    save_jpg(image, image_path)
    text = f"scanned {label} document rvl_cdip"
    record = {
        "file_name": rel_path.as_posix(),
        "dataset": dataset_id,
        "category": label,
        "title": f"RVL-CDIP {label} {source_id}",
        "brand": "",
        "source_id": source_id,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}/blob/main/{filename}",
        "image_url": f"{endpoint.rstrip('/')}/datasets/{dataset_id}/resolve/main/{filename}",
        "text": text,
        "metadata": {"label": label},
        "license_note": LICENSE_NOTE,
    }
    write_sidecar(image_path, record)
    return record


def load_dataset_info_labels(path: Path, label_column: str = "label") -> list[str]:
    """! @brief 从 HF dataset_infos.json 读取 ClassLabel 名称。"""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = next(iter(payload.values()))
        names = info["features"][label_column]["names"]
    except Exception:
        return []
    return [str(name) for name in names]


def imagenette_label_text(label_id: Any, label_names: list[str]) -> tuple[str, str]:
    """! @brief 把 Imagenette label id 转成 synset 与可读文本。"""

    try:
        synset = label_names[int(label_id)]
    except (TypeError, ValueError, IndexError):
        synset = str(label_id)
    return synset, IMAGENETTE_LABELS.get(synset, synset)


def run_imagenette(args: argparse.Namespace) -> None:
    """! @brief 下载并保存 Imagenette-320 自然图像补量数据。"""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("missing pyarrow; run: uv sync --extra data") from exc

    dataset_id = "johnowhitaker/imagenette2-320"
    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    parquet_path = download_hf_dataset_file(
        dataset_id,
        "data/train-00000-of-00001.parquet",
        out / ".cache",
    )
    info_path = download_hf_dataset_file(dataset_id, "dataset_infos.json", out / ".cache")
    label_names = load_dataset_info_labels(info_path)
    seen = load_seen_metadata(metadata_path)
    written = 0
    counts: dict[str, int] = {}
    buffer: list[dict[str, Any]] = []

    parquet = pq.ParquetFile(parquet_path)
    total = min(args.max_samples, parquet.metadata.num_rows) if args.max_samples else parquet.metadata.num_rows
    with tqdm(total=total, desc="imagenette", unit="img") as progress:
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["image", "label"])
            for row in table.to_pylist():
                if args.max_samples and written >= args.max_samples:
                    break
                image_info = row.get("image") or {}
                image_bytes = image_info.get("bytes")
                if not image_bytes:
                    progress.update(1)
                    continue
                synset, label = imagenette_label_text(row.get("label"), label_names)
                source_key = f"{dataset_id}:{synset}:{written}:{image_info.get('path') or ''}"
                source_id = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
                label_slug = slug(label)
                rel_path = Path("images") / "imagenette" / label_slug / f"{source_id}.jpg"
                image_path = out / rel_path
                sidecar_path = image_path.with_suffix(".json")
                if image_path.exists() and sidecar_path.exists():
                    try:
                        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        record = None
                else:
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image = image.convert("RGB")
                            image.thumbnail((args.max_image_size, args.max_image_size))
                    except Exception:
                        progress.update(1)
                        continue
                    save_jpg(image, image_path)
                    record = {
                        "file_name": rel_path.as_posix(),
                        "dataset": dataset_id,
                        "category": label,
                        "title": f"Imagenette {label} {source_id}",
                        "brand": "",
                        "source_id": source_id,
                        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
                        "image_url": "",
                        "text": f"natural image imagenette {label} {synset}",
                        "metadata": {"label": label, "synset": synset},
                        "license_note": LICENSE_NOTE,
                    }
                    write_sidecar(image_path, record)

                written += 1
                counts[label_slug] = counts.get(label_slug, 0) + 1
                file_name = str(record.get("file_name") if record else "")
                if record and file_name and file_name not in seen:
                    seen.add(file_name)
                    buffer.append(record)
                if len(buffer) >= 200:
                    append_jsonl(metadata_path, buffer)
                    buffer.clear()
                progress.update(1)
            if args.max_samples and written >= args.max_samples:
                break
    if buffer:
        append_jsonl(metadata_path, buffer)
    print(json.dumps({"written": written, "counts": counts, "out": str(out)}, ensure_ascii=False))


def read_webdataset_captions(tar: tarfile.TarFile, txt_member: tarfile.TarInfo) -> list[str]:
    """! @brief 从 WebDataset 文本成员读取多条 caption。"""

    extracted = tar.extractfile(txt_member)
    if extracted is None:
        return []
    text = extracted.read().decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_flickr30k_record(
    tar: tarfile.TarFile,
    jpg_member: tarfile.TarInfo,
    txt_member: tarfile.TarInfo,
    out: Path,
    split: str,
    shard_name: str,
    max_image_size: int,
) -> dict[str, Any] | None:
    """! @brief 从 Flickr30k WebDataset shard 保存图片并生成评测元信息。"""

    dataset_id = "clip-benchmark/wds_flickr30k"
    source_id = Path(jpg_member.name).stem
    rel_path = Path("images") / "flickr30k" / split / f"{source_id}.jpg"
    image_path = out / rel_path

    captions = read_webdataset_captions(tar, txt_member)
    if not captions:
        return None
    if not image_path.exists():
        extracted = tar.extractfile(jpg_member)
        if extracted is None:
            return None
        try:
            with Image.open(io.BytesIO(extracted.read())) as image:
                image = image.convert("RGB")
                image.thumbnail((max_image_size, max_image_size))
        except Exception:
            return None
        save_jpg(image, image_path)

    return {
        "file_name": rel_path.as_posix(),
        "dataset": dataset_id,
        "category": f"flickr30k-{split}",
        "title": f"Flickr30k {split} {source_id}",
        "brand": "",
        "source_id": source_id,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}/blob/main/{shard_name}",
        "image_url": "",
        "text": "flickr30k benchmark image",
        "index_text": f"flickr30k benchmark image {split}",
        "metadata": {
            "benchmark": "flickr30k",
            "task": "text-to-image-retrieval",
            "split": split,
            "image_key": source_id,
            "caption_count": len(captions),
            "captions": captions,
            "shard": shard_name,
        },
        "license_note": LICENSE_NOTE,
    }


def flickr30k_shard_names(args: argparse.Namespace, out: Path, dataset_id: str) -> list[str]:
    """! @brief 解析 Flickr30k WebDataset shard 文件名列表。"""

    if args.shards:
        shard_count = args.shards
    else:
        nshards_path = download_hf_dataset_file(dataset_id, f"{args.split}/nshards.txt", out / ".cache")
        shard_count = int(nshards_path.read_text(encoding="utf-8").strip())
    return [f"{args.split}/{idx}.tar" for idx in range(shard_count)]


def run_flickr30k_test(args: argparse.Namespace) -> None:
    """! @brief 下载 clip-benchmark Flickr30k test 集，用于标准图文检索指标。"""

    dataset_id = "clip-benchmark/wds_flickr30k"
    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    seen = load_seen_metadata(metadata_path)
    written = 0
    skipped_seen = 0
    buffer: list[dict[str, Any]] = []

    for shard_name in flickr30k_shard_names(args, out, dataset_id):
        shard_path = download_hf_dataset_file(dataset_id, shard_name, out / ".cache")
        with tarfile.open(shard_path) as tar:
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            jpg_members = sorted(
                (
                    member
                    for member in members.values()
                    if Path(member.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ),
                key=lambda member: member.name,
            )
            for jpg_member in tqdm(jpg_members, desc=shard_name, unit="img"):
                if args.max_samples and written >= args.max_samples:
                    break
                txt_member = members.get(f"{Path(jpg_member.name).stem}.txt")
                if txt_member is None:
                    continue
                record = build_flickr30k_record(
                    tar=tar,
                    jpg_member=jpg_member,
                    txt_member=txt_member,
                    out=out,
                    split=args.split,
                    shard_name=shard_name,
                    max_image_size=args.max_image_size,
                )
                if record is None:
                    continue
                file_name = str(record.get("file_name") or "")
                if file_name in seen:
                    skipped_seen += 1
                    written += 1
                    continue
                seen.add(file_name)
                buffer.append(record)
                written += 1
                if len(buffer) >= 200:
                    append_jsonl(metadata_path, buffer)
                    buffer.clear()
            if args.max_samples and written >= args.max_samples:
                break
    if buffer:
        append_jsonl(metadata_path, buffer)
    print(
        json.dumps(
            {
                "dataset": dataset_id,
                "split": args.split,
                "written": written,
                "skipped_seen": skipped_seen,
                "out": str(out),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
        )
    )


def run_receipts(args: argparse.Namespace) -> None:
    """! @brief 可选下载 receipt 数据；依赖缺失时正常退出。"""

    try:
        import fiftyone.utils.huggingface as fouh
    except ImportError:
        print("fiftyone is not installed; skip receipts. Run hardware/documents first.")
        return

    out = args.out.expanduser().resolve()
    metadata_path = out / "metadata.jsonl"
    dataset = fouh.load_from_hub(
        "Voxel51/consolidated_receipt_dataset",
        max_samples=args.max_samples,
        overwrite=False,
    )
    written: list[dict[str, Any]] = []
    for idx, sample in enumerate(dataset.iter_samples()):
        src = Path(sample.filepath)
        if not src.exists():
            continue
        try:
            with Image.open(src) as image:
                image = image.convert("RGB")
                image.thumbnail((args.max_image_size, args.max_image_size))
                source_id = f"{idx:06d}"
                rel_path = Path("images") / "receipts" / f"{source_id}.jpg"
                image_path = out / rel_path
                save_jpg(image, image_path)
        except Exception:
            continue
        fields = sample.to_dict()
        fields.pop("filepath", None)
        text = " ".join(str(value) for value in fields.values() if value is not None)
        record = {
            "file_name": rel_path.as_posix(),
            "dataset": "Voxel51/consolidated_receipt_dataset",
            "category": "receipt",
            "title": f"receipt {source_id}",
            "brand": "",
            "source_id": source_id,
            "source_url": "",
            "image_url": "",
            "text": text,
            "metadata": fields,
            "license_note": LICENSE_NOTE,
        }
        write_sidecar(image_path, record)
        written.append(record)
    append_jsonl(metadata_path, written)
    print(json.dumps({"written": len(written), "out": str(out)}, ensure_ascii=False))


def run_patent(args: argparse.Namespace) -> None:
    """! @brief 保留 PatFig 入口，但默认不下载 2GB+ 数据。"""

    print(
        "PatFig is intentionally not downloaded by default because it is large and license-sensitive. "
        "Use the dataset page directly when you are ready for that experiment."
    )


def build_parser() -> argparse.ArgumentParser:
    """! @brief 构建数据引导 CLI。"""

    parser = argparse.ArgumentParser(prog="bootstrap_data.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("hardware")
    hardware.add_argument("--out", type=Path, default=Path("data/corpus"))
    hardware.add_argument("--per-category", type=int, default=60)
    hardware.add_argument("--categories", nargs="+", default=DEFAULT_HARDWARE_CATEGORIES)
    hardware.add_argument("--max-image-size", type=int, default=768)
    hardware.add_argument("--seed", type=int, default=42)
    hardware.add_argument("--no-shuffle", action="store_true")
    hardware.add_argument("--workers", type=int, default=16)
    hardware.set_defaults(func=run_hardware)

    documents = subparsers.add_parser("documents")
    documents.add_argument("--out", type=Path, default=Path("data/corpus"))
    documents.add_argument("--dataset", default="rvl_cdip_small")
    documents.add_argument("--per-label", type=int, default=20)
    documents.add_argument("--max-image-size", type=int, default=1024)
    documents.add_argument("--workers", type=int, default=12)
    documents.set_defaults(func=run_documents)

    imagenette = subparsers.add_parser("imagenette")
    imagenette.add_argument("--out", type=Path, default=Path("data/corpus"))
    imagenette.add_argument("--max-samples", type=int, default=13394)
    imagenette.add_argument("--max-image-size", type=int, default=512)
    imagenette.set_defaults(func=run_imagenette)

    flickr30k = subparsers.add_parser("flickr30k-test")
    flickr30k.add_argument("--out", type=Path, default=Path("data/flickr30k_test"))
    flickr30k.add_argument("--split", default="test")
    flickr30k.add_argument("--shards", type=int, default=0)
    flickr30k.add_argument("--max-samples", type=int, default=0)
    flickr30k.add_argument("--max-image-size", type=int, default=512)
    flickr30k.set_defaults(func=run_flickr30k_test)

    receipts = subparsers.add_parser("receipts")
    receipts.add_argument("--out", type=Path, default=Path("data/corpus"))
    receipts.add_argument("--max-samples", type=int, default=100)
    receipts.add_argument("--max-image-size", type=int, default=1024)
    receipts.set_defaults(func=run_receipts)

    patent = subparsers.add_parser("patent")
    patent.set_defaults(func=run_patent)
    return parser


def main() -> None:
    """! @brief 脚本主入口。"""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
