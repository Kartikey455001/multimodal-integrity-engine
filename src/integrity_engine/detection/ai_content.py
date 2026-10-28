"""Deterministic, non-calibrated stylistic AI-content indicator."""

from __future__ import annotations

import re
import statistics


def ai_content_indicator(text: str, min_words: int = 100) -> dict:
    words = re.findall(r"\b[\w'-]+\b", (text or "").casefold())
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if part]
    if not words:
        return {"score": 0.0, "confidence": "insufficient", "features": {}}
    lengths = [len(re.findall(r"\b[\w'-]+\b", sentence)) for sentence in sentences] or [len(words)]
    diversity = len(set(words)) / len(words)
    if len(lengths) > 1:
        variability = statistics.pstdev(lengths) / max(statistics.mean(lengths), 1)
    else:
        variability = 0.0
    bigrams = list(zip(words, words[1:]))
    repetition = 1 - (len(set(bigrams)) / max(len(bigrams), 1))
    # Weights: lexical diversity 45%, sentence-length CV 30%, repeated bigrams 25%.
    diversity_term = max(0.0, min(1.0, (0.58 - diversity) / 0.30))
    variability_term = max(0.0, min(1.0, (0.35 - variability) / 0.35))
    score = max(
        0.0,
        min(100.0, round(100 * (0.45 * diversity_term + 0.30 * variability_term + 0.25 * repetition), 2)),
    )
    enough = len(words) >= min_words
    return {
        "score": score,
        "confidence": "low" if enough else "insufficient",
        "features": {
            "word_count": len(words),
            "lexical_diversity": round(diversity, 3),
            "sentence_length_cv": round(variability, 3),
            "repeated_bigram_rate": round(repetition, 3),
            "formula": "100*(0.45*diversity_gap + 0.30*low_length_cv + 0.25*repeated_bigram_rate)",
        },
    }
