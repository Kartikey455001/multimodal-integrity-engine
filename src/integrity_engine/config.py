"""Validated configuration loading. CLI overrides take priority over YAML."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .exceptions import ConfigurationError


class ChunkSettings(BaseModel):
    target_words: int = Field(default=260, ge=20)
    overlap_words: int = Field(default=45, ge=0)


class SimilaritySettings(BaseModel):
    peer_semantic_threshold: float = Field(default=0.80, ge=0, le=1)
    external_semantic_threshold: float = Field(default=0.82, ge=0, le=1)
    exact_threshold: float = Field(default=0.92, ge=0, le=1)
    top_k: int = Field(default=3, ge=1)


class EngineConfig(BaseModel):
    chunking: ChunkSettings = ChunkSettings()
    embedding: dict[str, Any] = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "cpu",
        "cache_dir": ".cache/embeddings",
    }
    similarity: SimilaritySettings = SimilaritySettings()
    boilerplate: dict[str, Any] = {
        "min_document_frequency": 5,
        "min_sentence_words": 5,
        "phrases": [
            "Explain your selected case study, describe the central problem, and recommend a practical solution.",
        ],
    }
    ai_indicator: dict[str, Any] = {"min_words": 100}
    risk: dict[str, Any] = {
        "medium_peer": 15,
        "high_peer": 35,
        "medium_external": 15,
        "high_external": 35,
    }
    video: dict[str, Any] = {
        "whisper_model": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "language": "en",
    }
    references: str = "reference_corpus/references.json"
    reports: dict[str, Any] = {"max_matches": 12, "schema_version": "1.0"}
    logging: dict[str, Any] = {"level": "INFO"}
    offline: bool = False


def load_config(path: str | Path | None = None) -> EngineConfig:
    source = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    if path and not source.exists():
        raise ConfigurationError(f"Configuration file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) if source.exists() else {}
    return EngineConfig.model_validate(data or {})
