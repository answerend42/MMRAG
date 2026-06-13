"""Retriever 基类 — 所有模态召回器的统一协议。

每个模态的召回器独立成类，全部继承此基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mrag.models.evidence import EvidenceCard, Modality


class Retriever(ABC):
    """召回器基类。

    子类必须实现 search() 和 is_ready。
    """

    modality: Modality
    """本召回器负责的模态。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[EvidenceCard]:
        """执行召回，返回 EvidenceCard 列表。"""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """召回器是否可用（索引已加载）。"""
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} modality={self.modality.value} ready={self.is_ready}>"
