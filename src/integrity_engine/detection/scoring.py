"""Transparent risk labels. Percentages are screening signals, not misconduct probabilities."""

from __future__ import annotations

from typing import Any


def risk_level(
    peer: float,
    external: float,
    ai: float,
    has_match: bool,
    rules: dict[str, Any] | None = None,
) -> str:
    rules = rules or {}
    high_peer = float(rules.get("high_peer", 35))
    medium_peer = float(rules.get("medium_peer", 15))
    high_external = float(rules.get("high_external", 35))
    medium_external = float(rules.get("medium_external", 15))
    if peer >= high_peer or external >= high_external:
        return "HIGH"
    if peer >= medium_peer or external >= medium_external:
        return "MEDIUM"
    if has_match:
        return "MEDIUM"
    # AI-content indicator never promotes a document to HIGH plagiarism risk by itself.
    _ = ai
    return "LOW"


def human_review_recommendation(
    status: str,
    warnings: list[str],
    peer: float,
    external: float,
    has_match: bool,
    extraction_error: str | None,
) -> str:
    if status in {"failed", "missing"} or extraction_error:
        return "Human review recommended: extraction failed or the file could not be read."
    if status == "empty":
        return "Human review recommended: no extractable text; this is not evidence of originality or plagiarism."
    if any("OCR" in w or "transcript" in w.lower() or "silent" in w.lower() or "lexical" in w.lower() for w in warnings):
        return "Human review recommended: extraction quality or matcher limitations affect interpretation."
    if has_match and peer < 35 and external < 35:
        return "Human review recommended: matches exist but coverage is below the high-risk threshold."
    if has_match:
        return "Human review recommended before any integrity decision; scores are screening signals only."
    return "No high-coverage plagiarism evidence was reported. Still review if the assignment context requires it."
