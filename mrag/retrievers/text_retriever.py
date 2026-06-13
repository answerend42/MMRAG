"""文本召回器 — 基于 CLIP 语义向量的文本片段检索。

独立于图片检索存在，使用相同的 CLIP 编码器但维护自己的索引。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from mrag.models.evidence import Confidence, EvidenceCard, Modality
from mrag.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class TextRetriever(Retriever):
    """文本语义召回器。

    将文档按 Markdown 标题分块，用 CLIP 编码器转为向量。
    搜索时使用余弦相似度。

    用法:
        retriever = TextRetriever()
        retriever.bind_embedder(embedder)
        retriever.load("docs/")
        cards = retriever.search("你的问题", top_k=10)
    """

    modality = Modality.TEXT

    def __init__(self, embedder=None):
        self._embedder = embedder
        self._chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None

    # ── embedder ──────────────────────────────────────────────

    def bind_embedder(self, embedder) -> None:
        """绑定 CLIP 编码器。调用 load 前必须绑定。"""
        self._embedder = embedder

    @property
    def _enc(self):
        if self._embedder is None:
            raise RuntimeError("TextRetriever 未绑定 embedder，请先调用 bind_embedder()")
        return self._embedder

    # ── 索引 ──────────────────────────────────────────────────

    def load(self, directory: str | Path) -> int:
        """加载目录下 .md / .txt 文档，按标题分块建索引。

        Args:
            directory: 文档目录。

        Returns:
            索引的文本块数量。
        """
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"{root}")

        chunks: list[dict] = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in (".md", ".txt"):
                continue
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue

            rel = str(path.relative_to(root))
            sections = re.split(r"\n(?=##+\s)", text)
            for si, sec in enumerate(sections):
                sec = sec.strip()
                if len(sec) < 30:
                    continue
                chunks.append({"id": f"{rel}#s{si}", "text": sec[:800], "source": rel})

        if not chunks:
            logger.warning("TextRetriever: 目录 %s 无有效文本", directory)
            return 0

        self._chunks = chunks
        self._build_index()
        logger.info("TextRetriever: %d 个文本块", len(chunks))
        return len(chunks)

    def _build_index(self) -> None:
        texts = [c["text"] for c in self._chunks]
        if not texts:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            return
        self._embeddings = self._enc.encode_texts(texts, batch_size=32)

    # ── 搜索 ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[EvidenceCard]:
        """语义检索，返回 EvidenceCard 列表。

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            按相似度降序排列的 EvidenceCard 列表。
        """
        if self._embeddings is None or self._embeddings.size == 0:
            return []

        qvec = self._enc.encode_texts([query])
        scores = self._embeddings @ qvec.T
        scores = scores.flatten()

        n = min(top_k, len(scores))
        indices = np.argpartition(-scores, n - 1)[:n]
        indices = indices[np.argsort(-scores[indices])]

        cards: list[EvidenceCard] = []
        for rank, idx in enumerate(indices):
            chunk = self._chunks[idx]
            score = max(0.0, float(scores[idx]))
            if rank == 0:
                conf = Confidence.HIGH
            elif rank <= 2:
                conf = Confidence.MEDIUM
            else:
                conf = Confidence.LOW
            cards.append(
                EvidenceCard(
                    modality=Modality.TEXT,
                    source_id=chunk["id"],
                    locator=chunk.get("source", chunk["id"]),
                    content_ref=chunk["text"],
                    score=score,
                    raw_reference=chunk.get("source", ""),
                    confidence=conf,
                )
            )
        return cards

    # ── 状态 ──────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._embeddings is not None and self._embeddings.size > 0

    @property
    def size(self) -> int:
        return len(self._chunks)
