"""Serializable domain records and versioned report validation."""

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class Segment:
    text: str
    location: dict[str, Any] = field(default_factory=dict)
    start: float | None = None
    end: float | None = None
    original_text: str | None = None


@dataclass
class Document:
    document_id: str
    path: str
    source_format: str
    status: str = "processed"
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    source_format: str
    sequence: int
    original_text: str
    normalized_text: str
    location: dict[str, Any] = field(default_factory=dict)


def as_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


class MatchLocation(BaseModel):
    model_config = ConfigDict(extra="allow")

    query_page: int | None = None
    matched_page: int | None = None
    query_paragraph: int | None = None
    matched_paragraph: int | None = None
    query_table: int | None = None
    matched_table: int | None = None
    query_start_seconds: float | None = None
    query_end_seconds: float | None = None
    matched_start_seconds: float | None = None
    matched_end_seconds: float | None = None


class MatchRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    matched_file: str | None = None
    similarity_score: float
    match_type: str
    query_passage: str
    matched_passage: str
    location: MatchLocation = Field(default_factory=MatchLocation)
    boilerplate_discounted: bool = False
    explanation: str = ""


class ScoreBlock(BaseModel):
    peer_similarity_percentage: float
    external_similarity_percentage: float
    ai_generated_probability: float


class DocumentReport(BaseModel):
    """Versioned per-document report. Extra keys are preserved for metadata."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    document_id: str
    status: str
    total_integrity_risk: str
    scores: ScoreBlock
    peer_matches: list[dict[str, Any]] = Field(default_factory=list)
    multimodal_matches: list[dict[str, Any]] = Field(default_factory=list)
    external_matches: list[dict[str, Any]] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
