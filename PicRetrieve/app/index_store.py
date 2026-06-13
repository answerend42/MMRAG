"""SQLite 与 NumPy 索引存储。"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from app.metadata import ImageItem

logger = logging.getLogger(__name__)


class IndexStore:
    """! @brief 管理图片条目、FTS 文本索引和向量文件。"""

    def __init__(self, data_dir: Path):
        """! @brief 绑定索引数据目录。"""

        self.data_dir = data_dir.expanduser().resolve()
        self.db_path = self.data_dir / "index.sqlite"
        self.embeddings_path = self.data_dir / "embeddings.npy"
        self.ids_path = self.data_dir / "ids.npy"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """! @brief 打开 SQLite 连接并启用字典式行访问。"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_tables(self) -> None:
        """! @brief 创建 items 表和手动维护的 FTS5 表。"""

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                  id INTEGER PRIMARY KEY,
                  path TEXT NOT NULL UNIQUE,
                  rel_path TEXT NOT NULL,
                  filename TEXT NOT NULL,
                  folder TEXT,
                  ext TEXT,
                  width INTEGER,
                  height INTEGER,
                  size_bytes INTEGER,
                  mtime REAL,
                  metadata_json TEXT,
                  metadata_text TEXT,
                  ocr_text TEXT,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                  filename,
                  folder,
                  rel_path,
                  metadata_text,
                  ocr_text,
                  content=''
                )
                """
            )

    def reset(self) -> None:
        """! @brief 删除当前索引文件，准备全量重建。"""

        for path in (self.db_path, self.embeddings_path, self.ids_path):
            if path.exists():
                path.unlink()
        self.create_tables()

    def upsert_item(self, item: ImageItem) -> int:
        """! @brief 写入或更新图片条目，并同步 FTS rowid。"""

        self.create_tables()
        metadata_json = json.dumps(item.metadata, ensure_ascii=False, default=str)
        with self.connect() as conn:
            old_row = conn.execute(
                "SELECT id, filename, folder, rel_path,"
                " metadata_text, ocr_text FROM items WHERE path = ?",
                (str(item.path),),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO items (
                  path, rel_path, filename, folder, ext, width, height, size_bytes,
                  mtime, metadata_json, metadata_text, ocr_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  rel_path=excluded.rel_path,
                  filename=excluded.filename,
                  folder=excluded.folder,
                  ext=excluded.ext,
                  width=excluded.width,
                  height=excluded.height,
                  size_bytes=excluded.size_bytes,
                  mtime=excluded.mtime,
                  metadata_json=excluded.metadata_json,
                  metadata_text=excluded.metadata_text,
                  ocr_text=excluded.ocr_text
                """,
                (
                    str(item.path),
                    item.rel_path,
                    item.filename,
                    item.folder,
                    item.ext,
                    item.width,
                    item.height,
                    item.size_bytes,
                    item.mtime,
                    metadata_json,
                    item.metadata_text,
                    item.ocr_text,
                ),
            )
            row = conn.execute("SELECT id FROM items WHERE path = ?", (str(item.path),)).fetchone()
            if row is None:
                raise RuntimeError("failed to fetch inserted item id")
            item_id = int(row["id"])
            if old_row is not None:
                conn.execute(
                    """
                    INSERT INTO items_fts(
                      items_fts, rowid, filename, folder, rel_path, metadata_text, ocr_text
                    )
                    VALUES ('delete', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(old_row["id"]),
                        old_row["filename"],
                        old_row["folder"],
                        old_row["rel_path"],
                        old_row["metadata_text"],
                        old_row["ocr_text"],
                    ),
                )
            conn.execute(
                """
                INSERT INTO items_fts(rowid, filename, folder, rel_path, metadata_text, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    item.filename,
                    item.folder,
                    item.rel_path,
                    item.metadata_text,
                    item.ocr_text,
                ),
            )
            return item_id

    def count_items(self) -> int:
        """! @brief 返回索引条目数量。"""

        self.create_tables()
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()
        return int(row["n"] if row is not None else 0)

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        """! @brief 读取单个 item 的完整元信息。"""

        self.create_tables()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return row_to_item_dict(row)

    def list_items_by_ids(self, item_ids: list[int]) -> list[dict[str, Any]]:
        """! @brief 按给定 id 顺序批量读取 item。"""

        if not item_ids:
            return []
        placeholders = ",".join("?" for _ in item_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM items WHERE id IN ({placeholders})", item_ids
            ).fetchall()
        by_id = {int(row["id"]): row_to_item_dict(row) for row in rows}
        return [by_id[item_id] for item_id in item_ids if item_id in by_id]

    def list_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        """! @brief 按 id 顺序读取索引中的 item，用于评测抽样。"""

        self.create_tables()
        sql = "SELECT * FROM items ORDER BY id"
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [row_to_item_dict(row) for row in rows]

    def search_metadata(self, query: str, limit: int) -> list[tuple[int, float]]:
        """! @brief 用 FTS5 搜索元信息，并返回名次归一化分数。"""

        safe_query = sanitize_fts_query(query)
        if not safe_query or limit <= 0:
            return []
        self.create_tables()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT rowid, bm25(items_fts) AS rank
                FROM items_fts
                WHERE items_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
        if not rows:
            return []
        denom = max(len(rows) - 1, 1)
        return [(int(row["rowid"]), max(0.0, 1.0 - idx / denom)) for idx, row in enumerate(rows)]

    def save_embeddings(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """! @brief 保存已 L2 归一化的向量矩阵和行对齐 id。"""

        embeddings = np.asarray(embeddings, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        np.save(self.embeddings_path, embeddings)
        np.save(self.ids_path, ids)

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """! @brief 加载向量矩阵和 id；缺失时返回空数组。"""

        if not self.embeddings_path.exists() or not self.ids_path.exists():
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        embeddings = np.load(self.embeddings_path)
        ids = np.load(self.ids_path)
        return np.asarray(embeddings, dtype=np.float32), np.asarray(ids, dtype=np.int64)

    def validate_embeddings(self) -> bool:
        """! @brief 检查向量矩阵和 id 行数是否一致。"""

        embeddings, ids = self.load_embeddings()
        return embeddings.shape[0] == ids.shape[0]


def sanitize_fts_query(query: str) -> str:
    """! @brief 清洗 FTS 查询，避免特殊字符触发 MATCH 语法错误。"""

    query = query.replace("-", " ")
    tokens = re.findall(r"[0-9A-Za-z_\u4e00-\u9fff]+", query)
    return " ".join(tokens)


def row_to_item_dict(row: sqlite3.Row) -> dict[str, Any]:
    """! @brief 把 SQLite 行转为 API/检索使用的字典。"""

    metadata_raw = row["metadata_json"] or "{}"
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": int(row["id"]),
        "path": row["path"],
        "rel_path": row["rel_path"],
        "filename": row["filename"],
        "folder": row["folder"] or "",
        "ext": row["ext"] or "",
        "width": row["width"],
        "height": row["height"],
        "size_bytes": row["size_bytes"],
        "mtime": row["mtime"],

        "metadata": metadata,
        "metadata_text": row["metadata_text"] or "",
        "ocr_text": row["ocr_text"] or "",
    }
