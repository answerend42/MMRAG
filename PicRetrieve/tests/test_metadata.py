import json
from pathlib import Path

from PIL import Image

from app.metadata import scan_images


def make_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    image = Image.new("RGB", (32, 24), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_scan_image_reads_sidecars_without_ocr(tmp_path: Path) -> None:
    root = tmp_path / "samples"
    image_path = root / "invoice" / "invoice_001.png"
    make_image(image_path)
    image_path.with_suffix(".json").write_text(
        json.dumps({"vendor": "AMD", "invoice_no": "INV-2026-001"}),
        encoding="utf-8",
    )
    image_path.with_suffix(".txt").write_text("amount 1280 USD", encoding="utf-8")

    items = scan_images(root, ocr_mode="none")

    assert len(items) == 1
    item = items[0]
    assert item.rel_path == "invoice/invoice_001.png"
    assert item.width == 32
    assert item.height == 24
    assert item.metadata["sidecar"]["vendor"] == "AMD"
    assert "INV-2026-001" in item.metadata_text
    assert "amount 1280 USD" in item.metadata_text
    assert item.ocr_text == ""


def test_scan_image_merges_metadata_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    image_root = corpus / "images"
    image_path = image_root / "pcpartpicker" / "cpu" / "a.jpg"
    make_image(image_path, (0, 255, 0))
    metadata_path = corpus / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "file_name": "images/pcpartpicker/cpu/a.jpg",
                "category": "cpu",
                "brand": "AMD",
                "text": "AMD Ryzen CPU",
                "metadata": {"cores": 16},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = scan_images(image_root, metadata_path)

    assert len(items) == 1
    item = items[0]
    assert item.metadata["metadata_jsonl"]["brand"] == "AMD"
    assert "AMD Ryzen CPU" in item.metadata_text
    assert "cores 16" in item.metadata_text


def test_index_text_prevents_benchmark_label_leakage(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    image_root = corpus / "images"
    image_path = image_root / "flickr30k" / "test" / "s0000000.jpg"
    make_image(image_path, (0, 0, 255))
    metadata_path = corpus / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "file_name": "images/flickr30k/test/s0000000.jpg",
                "dataset": "clip-benchmark/wds_flickr30k",
                "text": "this text should not be indexed",
                "index_text": "flickr30k benchmark image test",
                "metadata": {
                    "captions": ["A very specific caption that should stay out of FTS"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = scan_images(image_root, metadata_path)

    assert len(items) == 1
    item = items[0]
    assert "flickr30k benchmark image test" in item.metadata_text
    assert "very specific caption" not in item.metadata_text
    assert "should not be indexed" not in item.metadata_text
