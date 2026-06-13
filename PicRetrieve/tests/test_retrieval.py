import numpy as np

from app.retrieval import Retriever, combine_scores, cosine_topk


def test_cosine_topk_orders_by_dot_product() -> None:
    embeddings = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)

    rows = cosine_topk(embeddings, query, top_k=2)

    assert [row for row, _score in rows] == [1, 2]


def test_profile_fusion_changes_ranking_pressure() -> None:
    visual_first_general = combine_scores(0.8, 0.0, "general")
    metadata_first_general = combine_scores(0.2, 1.0, "general")
    visual_first_document = combine_scores(0.8, 0.0, "document")
    metadata_first_document = combine_scores(0.2, 1.0, "document")

    assert visual_first_general > metadata_first_general
    assert metadata_first_document > visual_first_document


def test_metadata_candidates_keep_real_visual_scores() -> None:
    class Store:
        def load_embeddings(self):
            return (
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.8, 0.6],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([1, 2, 3], dtype=np.int64),
            )

        def search_metadata(self, query: str, limit: int):
            return [(3, 1.0)]

        def list_items_by_ids(self, item_ids: list[int]):
            return [
                {
                    "id": item_id,
                    "rel_path": f"{item_id}.jpg",
                    "width": 10,
                    "height": 10,
                    "metadata": {},
                }
                for item_id in item_ids
            ]

    class Embedder:
        def encode_texts(self, texts: list[str], batch_size: int = 32):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    retriever = Retriever(Store(), Embedder())

    results = retriever.search_by_text("needle", top_k=1, candidate_k=1)

    assert results[0]["id"] == 3
    assert results[0]["visual_score"] == 0.9
    assert results[0]["metadata_score"] == 1.0
