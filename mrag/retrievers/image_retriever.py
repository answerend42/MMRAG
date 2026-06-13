"""PicRetrieve-backed image retriever."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

from mrag.models.evidence import Confidence, EvidenceCard, Modality
from mrag.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class ImageRetriever(Retriever):
    """Expose PicRetrieve text-to-image and image-to-image search as EvidenceCards."""

    modality = Modality.IMAGE

    def __init__(self, embedder, index_store):
        from app.retrieval import Retriever as _PicRetriever

        self._retriever = _PicRetriever(index_store, embedder)
        self._item_count = index_store.count_items()
        self._embedder = embedder
        logger.info("ImageRetriever: %d 张图片就绪", self._item_count)

    def search(self, query: str, top_k: int = 10) -> list[EvidenceCard]:
        """Run PicRetrieve text-to-image search."""
        if not self.is_ready:
            return []
        if not query.strip():
            return []

        raw = self._retriever.search_by_text(text=query, top_k=top_k, profile="general")
        return self._to_cards(raw, query_type="text")

    def search_by_image_data(
        self, image_data: dict[str, Any], top_k: int = 10, metadata_query: str | None = None
    ) -> list[EvidenceCard]:
        """Use an uploaded image directly as the retrieval query."""

        if not self.is_ready:
            return []
        data_url = str(image_data.get("data_url") or "").strip()
        if not data_url:
            return []

        try:
            image = _decode_data_url_image(data_url)
        except ValueError as exc:
            logger.warning("ImageRetriever: invalid uploaded image data: %s", exc)
            return []

        raw = self._retriever.search_by_image(
            image=image, top_k=top_k, profile="general", metadata_query=metadata_query
        )
        return self._to_cards(raw, query_type="image")

    def _to_cards(self, raw: list[dict], query_type: str) -> list[EvidenceCard]:
        """PicRetrieve 原始结果 → EvidenceCard。"""
        cards: list[EvidenceCard] = []
        for item in raw:
            score = float(item.get("score", 0))
            cards.append(
                EvidenceCard(
                    modality=Modality.IMAGE,
                    source_id=str(item.get("id", "")),
                    locator=item.get("rel_path", ""),
                    content_ref=f"图片: {item.get('rel_path', '')}",
                    score=score,
                    raw_reference=item.get("rel_path", ""),
                    confidence=_confidence_from_score(score),
                    metadata={
                        "query_type": query_type,
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "visual_score": item.get("visual_score"),
                        "metadata_score": item.get("metadata_score"),
                        "metadata": item.get("metadata", {}),
                    },
                )
            )
        return cards

    @property
    def is_ready(self) -> bool:
        return self._item_count > 0


def _decode_data_url_image(data_url: str) -> Image.Image:
    prefix = "base64,"
    encoded = data_url.split(prefix, 1)[1] if prefix in data_url else data_url
    try:
        payload = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise ValueError("cannot decode image data URL") from exc


def _confidence_from_score(score: float) -> Confidence:
    if score > 0.55:
        return Confidence.HIGH
    if score > 0.40:
        return Confidence.MEDIUM
    return Confidence.LOW
