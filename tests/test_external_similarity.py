from integrity_engine.detection.external_similarity import find_external_matches
from integrity_engine.schemas import Chunk


def test_reference_attribution():
    text = "A distinctive synthetic passage about testing a resilient service carefully."
    chunk = Chunk("d", "d", "x.pdf", "pdf", 0, text, text.lower(), {"page": 1})
    matches, scores = find_external_matches([chunk], [{"reference_id": "r1", "title": "Synthetic ref", "text": text}], 0.8)
    assert matches[0]["matched_reference_id"] == "r1" and scores["d"] == 100
    assert matches[0]["location"]["query_page"] == 1


def test_empty_corpus_is_zero_evidence():
    text = "Anything at all"
    chunk = Chunk("d", "d", "x.pdf", "pdf", 0, text, text.lower(), {"page": 1})
    matches, scores = find_external_matches([chunk], [], 0.8)
    assert matches == [] and scores["d"] == 0
