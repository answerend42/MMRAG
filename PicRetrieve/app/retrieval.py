"""检索与分数融合。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.embedder import ClipEmbedder, l2_normalize
from app.index_store import IndexStore

logger = logging.getLogger(__name__)

PROFILES = {
    "general": {"visual": 0.85, "metadata": 0.15},
    "document": {"visual": 0.45, "metadata": 0.55},
}


@dataclass(slots=True)
class SearchScores:
    """! @brief 单个候选结果的分数字段。"""

    item_id: int
    score: float
    visual_score: float
    metadata_score: float


def cosine_topk(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """! @brief 使用矩阵乘法执行暴力余弦 top-k。"""

    if embeddings.size == 0 or top_k <= 0:
        return []
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 2:
        query = query[0]
    scores = embeddings @ query
    limit = min(top_k, scores.shape[0])
    if limit == scores.shape[0]:
        order = np.argsort(-scores)
    else:
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        order = candidates[np.argsort(-scores[candidates])]
    return [(int(idx), float(scores[idx])) for idx in order]


def cosine_scores(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """! @brief 计算所有图片与查询向量的余弦分数。"""

    if embeddings.size == 0:
        return np.empty((0,), dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 2:
        query = query[0]
    return embeddings @ query


def topk_from_scores(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """! @brief 从已计算的分数向量中取 top-k 行号和分数。"""

    if scores.size == 0 or top_k <= 0:
        return []
    limit = min(top_k, scores.shape[0])
    if limit == scores.shape[0]:
        order = np.argsort(-scores)
    else:
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        order = candidates[np.argsort(-scores[candidates])]
    return [(int(idx), float(scores[idx])) for idx in order]


def normalize_visual_score(cosine_score: float) -> float:
    """! @brief 把 [-1, 1] 的余弦分数归一化到 [0, 1]。"""

    return max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))


def combine_scores(
    visual_score: float,
    metadata_score: float,
    profile: str,
) -> float:
    """! @brief 按 profile 权重融合视觉和元信息分数。"""

    weights = PROFILES.get(profile, PROFILES["general"])
    return weights["visual"] * visual_score + weights["metadata"] * metadata_score


class Retriever:
    """! @brief 基于内存 NumPy 向量和 SQLite 元信息执行混合检索。"""

    def __init__(self, store: IndexStore, embedder: ClipEmbedder):
        """! @brief 绑定索引存储和 CLIP 编码器。"""

        self.store = store
        self.embedder = embedder

    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        profile: str = "general",
        metadata_query: str | None = None,
        candidate_k: int = 200,
    ) -> list[dict[str, Any]]:
        """! @brief 以图搜图，并可融合额外元信息查询。"""

        query_embedding = self.embedder.encode_images([image.convert("RGB")], batch_size=1)
        return self._search(
            query_embedding=query_embedding,
            metadata_query=metadata_query,
            top_k=top_k,
            candidate_k=candidate_k,
            profile=profile,
        )

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        profile: str = "general",
        metadata_query: str | None = None,
        candidate_k: int = 200,
    ) -> list[dict[str, Any]]:
        """! @brief 以文本搜图，并默认把文本也用于元信息检索。"""

        if not text.strip():
            return []
        query_embedding = self.embedder.encode_texts([text], batch_size=1)
        return self._search(
            query_embedding=query_embedding,
            metadata_query=metadata_query or text,
            top_k=top_k,
            candidate_k=candidate_k,
            profile=profile,
        )

    def _search(
        self,
        query_embedding: np.ndarray,
        metadata_query: str | None,
        top_k: int,
        candidate_k: int,
        profile: str,
    ) -> list[dict[str, Any]]:
        """! @brief 执行共享的候选召回、分数融合和结果格式化。"""

        embeddings, ids = self.store.load_embeddings()
        if embeddings.shape[0] == 0:
            raise ValueError("index embeddings are empty; run index first")
        if embeddings.shape[0] != ids.shape[0]:
            raise ValueError("embeddings.npy and ids.npy row counts differ")

        candidate_limit = max(top_k, candidate_k)
        query_embedding = l2_normalize(query_embedding)
        raw_visual_scores = cosine_scores(embeddings, query_embedding)
        visual_rows = topk_from_scores(raw_visual_scores, candidate_limit)
        visual_ids = [int(ids[row_idx]) for row_idx, _cosine in visual_rows]
        metadata_scores = dict(self.store.search_metadata(metadata_query or "", candidate_limit))
        candidate_ids = list(dict.fromkeys([*visual_ids, *metadata_scores.keys()]))
        row_by_id = {int(item_id): row_idx for row_idx, item_id in enumerate(ids)}
        items = self.store.list_items_by_ids(candidate_ids)

        scored: list[tuple[SearchScores, dict[str, Any]]] = []
        for item in items:
            item_id = int(item["id"])
            row_idx = row_by_id.get(item_id)
            visual_score = (
                normalize_visual_score(float(raw_visual_scores[row_idx]))
                if row_idx is not None
                else 0.0
            )
            metadata_score = metadata_scores.get(item_id, 0.0)
            score = combine_scores(visual_score, metadata_score, profile)
            scored.append(
                (
                    SearchScores(
                        item_id=item_id,
                        score=score,
                        visual_score=visual_score,
                        metadata_score=metadata_score,
                    ),
                    item,
                )
            )

        scored.sort(key=lambda pair: pair[0].score, reverse=True)
        return [format_result(scores, item) for scores, item in scored[:top_k]]


def format_result(scores: SearchScores, item: dict[str, Any]) -> dict[str, Any]:
    """! @brief 把内部 item 和分数转换为 API/CLI 统一 schema。"""

    return {
        "id": scores.item_id,
        "rel_path": item["rel_path"],
        "score": round(scores.score, 6),
        "visual_score": round(scores.visual_score, 6),
        "metadata_score": round(scores.metadata_score, 6),
        "width": item["width"],
        "height": item["height"],
        "metadata": item["metadata"],
    }
