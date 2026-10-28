"""Offline local reference corpus matching. This is not a live internet search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..preprocessing.chunking import chunk_segments
from ..schemas import Chunk, Segment
from .peer_similarity import flatten_location, lexical_similarity

LOG = logging.getLogger(__name__)


def load_references(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Reference corpus must be a JSON list")
    valid = []
    for item in data:
        if not item.get("reference_id") or not item.get("text"):
            LOG.warning("Skipping invalid reference entry: %s", item)
            continue
        valid.append(item)
    return valid


def find_external_matches(
    chunks: list[Chunk],
    references: list[dict],
    threshold: float = 0.82,
    exact_threshold: float = 0.92,
    semantic_scores: dict[tuple[str, str], float] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    totals: dict[str, int] = {}
    for chunk in chunks:
        totals[chunk.document_id] = totals.get(chunk.document_id, 0) + len(chunk.normalized_text.split())
    zero_scores = {doc_id: 0.0 for doc_id in totals}

    if not references:
        LOG.warning("Reference corpus empty or unavailable; external similarity is zero-evidence, not a web search.")
        return [], zero_scores

    refs: list[tuple[dict, Chunk]] = []
    for ref in references:
        for ch in chunk_segments(
            ref["reference_id"],
            ref.get("title") or ref["reference_id"],
            "reference",
            [Segment(ref["text"], {"reference_id": ref["reference_id"]})],
            260,
            0,
        ):
            refs.append((ref, ch))

    matches: list[dict] = []
    covered: dict[str, set[str]] = {}
    for query in chunks:
        candidates: list[tuple[float, dict, Chunk, bool, float]] = []
        for ref, ref_chunk in refs:
            lexical = lexical_similarity(query.original_text, ref_chunk.original_text)
            if semantic_scores is not None:
                sim = float(semantic_scores.get((query.chunk_id, ref_chunk.chunk_id), lexical))
            else:
                sim = lexical
            exact = query.normalized_text == ref_chunk.normalized_text and bool(query.normalized_text)
            if exact:
                sim = 1.0
            if sim >= threshold:
                candidates.append((sim, ref, ref_chunk, exact, lexical))
        if not candidates:
            continue
        sim, ref, ref_chunk, exact, lexical = max(candidates, key=lambda item: item[0])
        method = "exact" if exact else "near-exact" if lexical >= exact_threshold else (
            "semantic" if semantic_scores is not None else "lexical_overlap"
        )
        matches.append(
            {
                "query_document_id": query.document_id,
                "matched_file": ref.get("title"),
                "matched_reference_id": ref["reference_id"],
                "reference_title": ref.get("title"),
                "source_description": ref.get("source_description"),
                "url": ref.get("url"),
                "query_passage": query.original_text,
                "matched_passage": ref_chunk.original_text,
                "similarity_score": round(float(sim), 6),
                "match_type": method,
                "location": flatten_location(query.location, ref_chunk.location),
                "query_location": query.location,
                "source_location": ref_chunk.location,
                "boilerplate_discounted": False,
                "explanation": "Offline local reference match; this is not a live web search.",
            }
        )
        bucket = covered.setdefault(query.document_id, set())
        if query.chunk_id not in bucket:
            bucket.add(query.chunk_id)

    matched_words = {
        doc_id: sum(len(c.normalized_text.split()) for c in chunks if c.document_id == doc_id and c.chunk_id in ids)
        for doc_id, ids in covered.items()
    }
    scores = {
        doc_id: round(100 * min(1.0, matched_words.get(doc_id, 0) / max(1, words)), 2)
        for doc_id, words in totals.items()
    }
    return matches, scores
