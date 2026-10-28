"""Frequency-based boilerplate marking without deleting evidence."""

from __future__ import annotations

import re
from collections import defaultdict

from ..preprocessing.normalization import comparison_text
from ..schemas import Document


class BoilerplateDetector:
    def __init__(self, min_document_frequency: int = 5, phrases: list[str] | None = None, min_sentence_words: int = 5):
        self.minimum = min_document_frequency
        self.min_sentence_words = min_sentence_words
        self.phrases = {comparison_text(p) for p in (phrases or []) if comparison_text(p)}
        self.frequent: set[str] = set(self.phrases)

    def fit(self, docs: list[Document]) -> "BoilerplateDetector":
        owners: dict[str, set[str]] = defaultdict(set)
        for doc in docs:
            for segment in doc.segments:
                for sentence in re.split(r"(?<=[.!?])\s+", segment.text):
                    key = comparison_text(sentence)
                    if len(key.split()) >= self.min_sentence_words:
                        owners[key].add(doc.document_id)
        self.frequent = {key for key, docs_with in owners.items() if len(docs_with) >= self.minimum} | self.phrases
        return self

    def distinctive_text(self, text: str) -> str:
        """Strip configured/frequent template sentences, leaving independent wording."""
        key = comparison_text(text)
        for phrase in sorted(self.frequent | self.phrases, key=len, reverse=True):
            if phrase:
                key = key.replace(phrase, " ")
        return re.sub(r"\s+", " ", key).strip()

    def is_boilerplate(self, text: str) -> bool:
        key = comparison_text(text)
        if key in self.frequent or key in self.phrases:
            return True
        distinctive = self.distinctive_text(text)
        contains_template = any(phrase and phrase in key for phrase in (self.phrases | self.frequent) if phrase)
        if not distinctive:
            return contains_template
        return contains_template and len(distinctive.split()) < self.min_sentence_words

    def reason(self, text: str) -> str | None:
        if not self.is_boilerplate(text):
            return None
        return (
            "Discounted as common assignment/template language because it matches a configured phrase "
            f"or appears in at least {self.minimum} distinct documents. Independent analysis was not "
            "treated as copied solely because of the shared prompt."
        )
