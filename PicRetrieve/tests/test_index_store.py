from pathlib import Path

import numpy as np

from app.index_store import IndexStore
from app.metadata import ImageItem


def make_item(path: Path, metadata_text: str = "AMD invoice gpu") -> ImageItem:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-used-by-store")
    return ImageItem(
        path=path.resolve(),
        rel_path="gpu/item.jpg",
        filename="item.jpg",
        folder="gpu",
        ext=".jpg",
        width=10,
        height=10,
        size_bytes=path.stat().st_size,
        mtime=path.stat().st_mtime,
        metadata={"folder": "gpu", "sidecar": {"vendor": "AMD"}},
        metadata_text=metadata_text,
        ocr_text="invoice text",
    )


def test_sqlite_tables_and_fts_search(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "data")
    store.create_tables()
    item_id = store.upsert_item(make_item(tmp_path / "item.jpg"))

    rows = store.search_metadata("AMD invoice", limit=10)

    assert rows == [(item_id, 1.0)]
    item = store.get_item(item_id)
    assert item is not None
    assert item["metadata"]["sidecar"]["vendor"] == "AMD"


def test_embeddings_and_ids_row_counts_match(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "data")
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.asarray([10, 11], dtype=np.int64)

    store.save_embeddings(embeddings, ids)
    loaded_embeddings, loaded_ids = store.load_embeddings()

    assert loaded_embeddings.shape == (2, 2)
    assert loaded_ids.tolist() == [10, 11]
    assert store.validate_embeddings()


def test_upsert_replaces_old_fts_tokens(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "data")
    image_path = tmp_path / "item.jpg"
    item = make_item(image_path, "oldtoken")
    store.upsert_item(item)
    item.metadata_text = "newtoken"
    store.upsert_item(item)

    assert store.search_metadata("oldtoken", limit=10) == []
    assert store.search_metadata("newtoken", limit=10)
