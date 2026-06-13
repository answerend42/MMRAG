"""模态召回器。

每个模态的召回器独立成类，统一继承 Retriever 基类。
所有召回器都返回 EvidenceCard，retrievers_node 用注册表分发。
"""

from mrag.retrievers.base import Retriever
from mrag.retrievers.image_retriever import ImageRetriever
from mrag.retrievers.text_retriever import TextRetriever

__all__ = ["Retriever", "TextRetriever", "ImageRetriever"]
