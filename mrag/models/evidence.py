"""EvidenceCard — 统一证据卡片格式。

所有检索器输出的证据都必须转换成这个格式，
保证重排、验证和生成阶段可以统一处理。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """证据来源模态。"""

    TEXT = "text"
    IMAGE = "image"
    PAGE = "page"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    SQL = "sql"
    KG = "kg"


class Confidence(StrEnum):
    """证据置信度等级。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceCard(BaseModel):
    """统一证据卡片，所有模态的检索结果都转换为此格式。

    字段设计遵循架构图中 Runtime Evidence Contract 的规范。
    """

    modality: Modality
    """证据来源类型。"""

    source_id: str
    """原始数据或资产 ID（文档 ID、图片 ID、音频 ID、表名等）。"""

    locator: str
    """可定位引用（chunk id、page+bbox、start/end time、row id、KG path）。"""

    content_ref: str
    """可供生成器读取的内容引用（文本片段、OCR 块、caption、SQL 结果、子图）。"""

    score: float = 0.0
    """召回或重排分数，归一化到 [0, 1]。"""

    raw_reference: str = ""
    """回溯到原始文件或查询的引用（file path、object key、SQL query、graph query）。"""

    confidence: Confidence = Confidence.MEDIUM
    """分支或 verifier 置信度。"""

    notes: str = ""
    """风险、冲突或人工备注。"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """额外元信息，供调试和追踪使用。"""


class EvidenceBundle(BaseModel):
    """一组证据卡片及其聚合信息。"""

    cards: list[EvidenceCard] = Field(default_factory=list)
    coverage: float = 0.0
    """证据覆盖率，用于判断是否足够生成答案。"""

    max_confidence: Confidence = Confidence.LOW
    """当前证据集中的最高置信度。"""

    has_conflict: bool = False
    """是否存在证据冲突。"""
