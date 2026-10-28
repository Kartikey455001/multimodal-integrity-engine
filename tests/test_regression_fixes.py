"""Regression tests covering root cause fixes:
1. Paraphrase detection retention (03_paraphrase_a ↔ 04_paraphrase_b).
2. Cross-modal detection retention (05_crossmodal_written ↔ 12_pitch.mp4).
3. Video timestamp and PDF page/paragraph provenance preservation.
4. Boilerplate discounting without distinctive content suppression.
5. Cross-modal classification rules and self-match exclusion.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from integrity_engine.detection.boilerplate import BoilerplateDetector
from integrity_engine.detection.peer_similarity import find_peer_matches
from integrity_engine.extractors.base import extract
from integrity_engine.extractors.pdf_extractor import extract_pdf
from integrity_engine.pipeline import _is_cross_modal, run_pipeline
from integrity_engine.preprocessing.chunking import chunk_segments
from integrity_engine.schemas import Chunk, Document, Segment


def test_cross_modal_classification_logic():
    """Verify cross-modal rule: MP4 <-> PDF/DOCX only; PDF <-> DOCX is internal peer."""
    assert _is_cross_modal("mp4", "pdf") is True
    assert _is_cross_modal("pdf", "mp4") is True
    assert _is_cross_modal("mp4", "docx") is True
    assert _is_cross_modal("docx", "mp4") is True
    assert _is_cross_modal("pdf", "docx") is False
    assert _is_cross_modal("docx", "pdf") is False
    assert _is_cross_modal("pdf", "pdf") is False
    assert _is_cross_modal("docx", "docx") is False


def test_pdf_page_and_paragraph_provenance(tmp_path):
    """Verify PDF extraction preserves page and paragraph boundaries."""
    from generate_sample_data import write_pdf
    pdf_path = tmp_path / "test_doc.pdf"
    write_pdf(pdf_path, "Paragraph 1 prompt header.\n\nParagraph 2 distinctive analysis text.")
    segments, _ = extract_pdf(pdf_path)
    assert len(segments) >= 2
    assert segments[0].location["page"] == 1
    assert segments[0].location["paragraph"] == 1
    assert segments[1].location["page"] == 1
    assert segments[1].location["paragraph"] == 2


def test_video_timestamp_provenance_preservation():
    """Verify video transcript slices are chunked into coherent groups with valid timestamps."""
    segments = [
        Segment("First part of pitch.", start=0.0, end=5.2),
        Segment("Second part of pitch.", start=5.2, end=11.4),
    ]
    chunks = chunk_segments("pitch.mp4", "pitch.mp4", "mp4", segments, target_words=50)
    assert len(chunks) == 1
    assert chunks[0].location["start_seconds"] == 0.0
    assert chunks[0].location["end_seconds"] == 11.4

    # Match against PDF chunk and verify match record provenance
    pdf_chunk = Chunk(
        chunk_id="doc.pdf::chunk-0000",
        document_id="doc.pdf",
        filename="doc.pdf",
        source_format="pdf",
        sequence=0,
        original_text="First part of pitch. Second part of pitch.",
        normalized_text="first part of pitch second part of pitch",
        location={"page": 1, "paragraph": 2},
    )
    matches, scores = find_peer_matches([chunks[0], pdf_chunk], threshold=0.8)
    assert len(matches) == 2
    hit_from_pdf = next(m for m in matches if m["query_document_id"] == "doc.pdf")
    assert hit_from_pdf["location"]["query_page"] == 1
    assert hit_from_pdf["location"]["query_paragraph"] == 2
    assert hit_from_pdf["location"]["matched_start_seconds"] == 0.0
    assert hit_from_pdf["location"]["matched_end_seconds"] == 11.4


def test_boilerplate_discounting_preserves_distinctive_content():
    """Verify boilerplate discounting discounts common templates but retains distinctive analysis."""
    prompt = "Explain your selected case study, describe the central problem, and recommend a practical solution."
    prompt_chunk_a = Chunk("a::0", "a", "a.docx", "docx", 0, prompt, prompt.lower(), {"paragraph": 1})
    prompt_chunk_b = Chunk("b::0", "b", "b.docx", "docx", 0, prompt, prompt.lower(), {"paragraph": 1})
    analysis_chunk_a = Chunk("a::1", "a", "a.docx", "docx", 1, "Distinctive solar battery pilot.", "distinctive solar battery pilot", {"paragraph": 2})
    analysis_chunk_b = Chunk("b::1", "b", "b.docx", "docx", 1, "Distinctive solar battery pilot.", "distinctive solar battery pilot", {"paragraph": 2})

    detector = BoilerplateDetector(2, phrases=[prompt])
    all_chunks = [prompt_chunk_a, prompt_chunk_b, analysis_chunk_a, analysis_chunk_b]
    matches, scores = find_peer_matches(all_chunks, threshold=0.8, boilerplate=detector)

    # Prompt matches must be discounted as boilerplate
    prompt_matches = [m for m in matches if m["query_passage"] == prompt]
    assert all(m.get("boilerplate_discounted") for m in prompt_matches)

    # Distinctive analysis matches must NOT be discounted
    analysis_matches = [m for m in matches if m["query_passage"] == "Distinctive solar battery pilot."]
    assert any(not m.get("boilerplate_discounted") for m in analysis_matches)
    assert scores["a"] > 0


def test_self_match_exclusion():
    """Verify self-matching is strictly excluded."""
    c1 = Chunk("a::0", "a", "a.docx", "docx", 0, "Some sample analysis text.", "some sample analysis text", {})
    c2 = Chunk("a::1", "a", "a.docx", "docx", 1, "Some sample analysis text.", "some sample analysis text", {})
    matches, _ = find_peer_matches([c1, c2], threshold=0.8)
    assert len(matches) == 0


def test_paraphrase_regression_on_sample_submissions():
    """Verify 03_paraphrase_a and 04_paraphrase_b are matched without being discounted as boilerplate."""
    root = Path("sample_submissions")
    if not (root / "03_paraphrase_a.docx").exists() or not (root / "04_paraphrase_b.docx").exists():
        pytest.skip("Sample submissions not present.")

    doc3 = extract(root / "03_paraphrase_a.docx", root)
    doc4 = extract(root / "04_paraphrase_b.docx", root)
    chunks3 = chunk_segments(doc3.document_id, doc3.path, doc3.source_format, doc3.segments)
    chunks4 = chunk_segments(doc4.document_id, doc4.path, doc4.source_format, doc4.segments)

    # The distinctive paragraphs must match under semantic cosine similarity
    from integrity_engine.embeddings.encoder import SentenceEncoder, cosine_matrix
    encoder = SentenceEncoder()
    v3 = encoder.encode([c.original_text for c in chunks3])
    v4 = encoder.encode([c.original_text for c in chunks4])
    sim = cosine_matrix(v3, v4)

    scores = {}
    for i, c3 in enumerate(chunks3):
        for j, c4 in enumerate(chunks4):
            scores[(c3.chunk_id, c4.chunk_id)] = float(sim[i, j])
            scores[(c4.chunk_id, c3.chunk_id)] = float(sim[i, j])

    detector = BoilerplateDetector(5, phrases=["Explain your selected case study"])
    matches, coverages = find_peer_matches(chunks3 + chunks4, threshold=0.8, semantic_scores=scores, boilerplate=detector)

    valid_matches = [m for m in matches if not m.get("boilerplate_discounted")]
    assert len(valid_matches) >= 2
    assert coverages["03_paraphrase_a.docx"] > 0
    assert coverages["04_paraphrase_b.docx"] > 0
