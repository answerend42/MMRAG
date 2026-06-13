from __future__ import annotations

from mrag.graph import build_picretrieve_graph
from mrag.models.evidence import Confidence, EvidenceCard, Modality
from mrag.retrievers.base import Retriever
from mrag.state import MMRAGState


class FakeImageRetriever(Retriever):
    modality = Modality.IMAGE

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.image_calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, top_k: int = 10) -> list[EvidenceCard]:
        batch = len(self.calls)
        self.calls.append((query, top_k))
        return [
            EvidenceCard(
                modality=Modality.IMAGE,
                source_id=f"{batch}-{i}",
                locator=f"demo/{batch}/image-{i}.jpg",
                content_ref=f"图片: demo/{batch}/image-{i}.jpg",
                score=0.8 - i * 0.05,
                raw_reference=f"demo/{batch}/image-{i}.jpg",
                confidence=Confidence.HIGH,
            )
            for i in range(top_k)
        ]

    def search_by_image_data(
        self, image_data: dict, top_k: int = 10, metadata_query: str | None = None
    ) -> list[EvidenceCard]:
        batch = len(self.calls) + len(self.image_calls)
        self.image_calls.append((image_data.get("name", ""), top_k, metadata_query))
        return [
            EvidenceCard(
                modality=Modality.IMAGE,
                source_id=f"image-{batch}-{i}",
                locator=f"demo/image/{batch}/image-{i}.jpg",
                content_ref=f"图片: demo/image/{batch}/image-{i}.jpg",
                score=0.9 - i * 0.05,
                raw_reference=f"demo/image/{batch}/image-{i}.jpg",
                confidence=Confidence.HIGH,
            )
            for i in range(top_k)
        ]

    @property
    def is_ready(self) -> bool:
        return True


def test_picretrieve_graph_runs_image_module_only() -> None:
    retriever = FakeImageRetriever()
    graph = build_picretrieve_graph(registry={Modality.IMAGE: retriever})

    result = graph.invoke(MMRAGState(query="black dog", top_k=3))

    assert retriever.calls == [("black dog", 3)]
    assert result["query_plan"].required_modalities == [Modality.IMAGE]
    assert len(result["reranked_cards"]) == 3
    assert result["answer"].startswith("已为「black dog」召回 3 张候选图片")
    assert "picretrieve_planner: fallback 1 queries" in result["trace"]


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.response_formats: list[dict | None] = []

    def chat(self, messages, **kwargs) -> str:
        system = messages[0]["content"]
        self.calls.append(system)
        self.response_formats.append(kwargs.get("response_format"))
        if "Agent Planner" in system:
            return (
                '{"intent":"visual_localization",'
                '"retrieval_mode":"text_only",'
                '"rewritten_queries":["black dog running grass","dark dog on grass"],'
                '"rationale":"把自然语言问题改写成两个视觉检索 query。",'
                '"confidence":0.88}'
            )
        if "结果 Verifier" in system:
            return (
                '{"is_sufficient":true,'
                '"reason":"top score 足够，候选路径没有明显冲突。",'
                '"improved_queries":[],'
                '"confidence":0.82}'
            )
        return "LLM 总结：Planner 改写了查询，PicRetrieve 返回了可信候选。"


def test_picretrieve_graph_uses_llm_planner_and_answer() -> None:
    retriever = FakeImageRetriever()
    llm = FakeLLM()
    graph = build_picretrieve_graph(registry={Modality.IMAGE: retriever}, llm=llm)

    result = graph.invoke(MMRAGState(query="black dog", top_k=2, max_retries=1))

    assert retriever.calls == [("black dog running grass", 2), ("dark dog on grass", 2)]
    assert llm.response_formats[0]["type"] == "json_schema"
    assert result["agent_plan"]["mode"] == "llm"
    assert result["agent_verification"]["mode"] == "llm"
    assert result["answer"].startswith("LLM 总结")


def test_picretrieve_graph_uses_direct_image_retrieval_without_text() -> None:
    retriever = FakeImageRetriever()
    graph = build_picretrieve_graph(registry={Modality.IMAGE: retriever})

    result = graph.invoke(
        MMRAGState(
            query="",
            top_k=2,
            input_images=[
                {
                    "name": "query.jpg",
                    "mime_type": "image/jpeg",
                    "size": 12,
                    "data_url": "data:image/jpeg;base64,ZmFrZQ==",
                }
            ],
        )
    )

    assert retriever.calls == []
    assert retriever.image_calls == [("query.jpg", 2, None)]
    assert result["agent_plan"]["mode"] == "fallback"
    assert result["agent_plan"]["retrieval_mode"] == "image_direct"
    assert result["query_plan"].sub_queries == []
    assert len(result["reranked_cards"]) == 2


def test_picretrieve_graph_lets_llm_inspect_image_without_forcing_text_retrieval() -> None:
    retriever = FakeImageRetriever()
    llm = FakeLLM()
    graph = build_picretrieve_graph(registry={Modality.IMAGE: retriever}, llm=llm)

    result = graph.invoke(
        MMRAGState(
            query="",
            top_k=1,
            max_retries=0,
            input_images=[
                {
                    "name": "query.png",
                    "mime_type": "image/png",
                    "size": 12,
                    "data_url": "data:image/png;base64,ZmFrZQ==",
                }
            ],
        )
    )

    assert retriever.calls == []
    assert retriever.image_calls == [("query.png", 1, None)]
    assert result["agent_plan"]["mode"] == "llm"
    assert result["agent_plan"]["retrieval_mode"] == "image_direct"
    assert result["agent_plan"]["suggested_queries"] == [
        "black dog running grass",
        "dark dog on grass",
    ]
    assert result["query_plan"].sub_queries == []


def test_picretrieve_graph_does_not_parallelize_reference_image_instruction() -> None:
    retriever = FakeImageRetriever()
    llm = FakeLLM()
    graph = build_picretrieve_graph(registry={Modality.IMAGE: retriever}, llm=llm)

    result = graph.invoke(
        MMRAGState(
            query="按照对应的图片查找",
            top_k=1,
            max_retries=0,
            input_images=[
                {
                    "name": "reference.png",
                    "mime_type": "image/png",
                    "size": 12,
                    "data_url": "data:image/png;base64,ZmFrZQ==",
                }
            ],
        )
    )

    assert retriever.calls == []
    assert retriever.image_calls == [("reference.png", 1, "按照对应的图片查找")]
    assert result["agent_plan"]["retrieval_mode"] == "image_direct"
    assert result["query_plan"].sub_queries == []
    assert len(result["reranked_cards"]) == 1
