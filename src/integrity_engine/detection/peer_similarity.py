"""Peer matching with document-identity self-exclusion and coverage scoring."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..preprocessing.normalization import comparison_text
from ..schemas import Chunk
from .boilerplate import BoilerplateDetector


def lexical_similarity(a: str, b: str) -> float:
    left, right = comparison_text(a), comparison_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def flatten_location(query: dict, source: dict) -> dict:
    return {
        "query_page": query.get("page"),
        "matched_page": source.get("page"),
        "query_paragraph": query.get("paragraph"),
        "matched_paragraph": source.get("paragraph"),
        "query_table": query.get("table"),
        "matched_table": source.get("table"),
        "query_start_seconds": query.get("start_seconds"),
        "query_end_seconds": query.get("end_seconds"),
        "matched_start_seconds": source.get("start_seconds"),
        "matched_end_seconds": source.get("end_seconds"),
    }


def _dedupe_key(match: dict) -> tuple:
    return (
        match["query_document_id"],
        match["source_document_id"],
        match["query_passage"][:180],
        match["matched_passage"][:180],
    )


def find_peer_matches(
    chunks: list[Chunk],
    threshold: float = 0.80,
    exact_threshold: float = 0.92,
    top_k: int = 3,
    boilerplate: BoilerplateDetector | None = None,
    semantic_scores: dict[tuple[str, str], float] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    matches: list[dict] = []
    covered_chunks: dict[str, set[str]] = {}
    eligible_words: dict[str, int] = {}
    matched_words: dict[str, int] = {}

    for query in chunks:
        eligible_words[query.document_id] = eligible_words.get(query.document_id, 0) + len(query.normalized_text.split())
        ranked: list[tuple[float, Chunk, bool, float]] = []
        for source in chunks:
            if query.document_id == source.document_id:
                continue
            lexical = lexical_similarity(query.original_text, source.original_text)
            if semantic_scores is not None:
                sim = float(semantic_scores.get((query.chunk_id, source.chunk_id), lexical))
            else:
                sim = lexical
            exact = query.normalized_text == source.normalized_text and bool(query.normalized_text)
            if exact:
                sim = 1.0
            if sim >= threshold:
                ranked.append((sim, source, exact, lexical))
        ranked.sort(key=lambda item: (-item[0], item[1].document_id, item[1].sequence))
        seen_sources: set[str] = set()
        for sim, source, exact, lexical in ranked[:top_k]:
            if source.chunk_id in seen_sources:
                continue
            seen_sources.add(source.chunk_id)
            boiler = False
            if boilerplate:
                if boilerplate.is_boilerplate(query.original_text) or boilerplate.is_boilerplate(source.original_text):
                    boiler = True
                else:
                    q_dist = boilerplate.distinctive_text(query.original_text)
                    s_dist = boilerplate.distinctive_text(source.original_text)
                    q_has_bp = bool(q_dist != comparison_text(query.original_text))
                    s_has_bp = bool(s_dist != comparison_text(source.original_text))
                    if q_has_bp or s_has_bp:
                        min_words = int(getattr(boilerplate, "min_sentence_words", 5))
                        if not q_dist or not s_dist or len(q_dist.split()) < min_words or len(s_dist.split()) < min_words:
                            boiler = True
                        elif semantic_scores is None and lexical_similarity(q_dist, s_dist) < threshold:
                            boiler = True
            if semantic_scores is not None:
                method = "exact" if exact else "near-exact" if lexical >= exact_threshold else "semantic"
                explanation = (
                    "The passages are textually identical."
                    if exact
                    else "The passages have high lexical overlap."
                    if lexical >= exact_threshold
                    else "The configured sentence embedding vectors are close in cosine space."
                )
            else:
                method = "exact" if exact else "near-exact" if lexical >= exact_threshold else "lexical_overlap"
                explanation = (
                    "The passages are textually identical."
                    if exact
                    else "The passages have high lexical overlap."
                    if lexical >= exact_threshold
                    else "The passages share lexical sequence overlap; semantic paraphrase detection is unavailable."
                )
            if boiler and boilerplate:
                explanation += " " + (boilerplate.reason(query.original_text) or "")
            match = {
                "query_document_id": query.document_id,
                "query_filename": query.filename,
                "source_document_id": source.document_id,
                "matched_file": source.filename,
                "query_passage": query.original_text,
                "matched_passage": source.original_text,
                "similarity_score": round(float(sim), 6),
                "match_type": method,
                "location": flatten_location(query.location, source.location),
                "query_location": query.location,
                "source_location": source.location,
                "boilerplate_discounted": boiler,
                "explanation": explanation.strip(),
            }
            matches.append(match)
            if not boiler:
                covered = covered_chunks.setdefault(query.document_id, set())
                if query.chunk_id not in covered:
                    covered.add(query.chunk_id)
                    matched_words[query.document_id] = matched_words.get(query.document_id, 0) + len(
                        query.normalized_text.split()
                    )

    matches.sort(key=lambda m: (-m["similarity_score"], m["query_document_id"], m["matched_file"]))
    unique: list[dict] = []
    seen: set[tuple] = set()
    for match in matches:
        key = _dedupe_key(match)
        if key in seen:
            continue
        seen.add(key)
        unique.append(match)

    scores = {
        doc_id: round(100 * min(1.0, matched_words.get(doc_id, 0) / max(1, words)), 2)
        for doc_id, words in eligible_words.items()
    }
    for chunk in chunks:
        scores.setdefault(chunk.document_id, 0.0)
    return unique, scores
