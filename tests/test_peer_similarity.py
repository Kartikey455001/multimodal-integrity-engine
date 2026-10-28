from integrity_engine.detection.boilerplate import BoilerplateDetector
from integrity_engine.detection.peer_similarity import find_peer_matches
from integrity_engine.schemas import Chunk, Document, Segment


def ch(doc, text, fmt="docx"):
    return Chunk(doc, doc, f"{doc}.{fmt}", fmt, 0, text, text.lower(), {})


def test_exact_match_and_self_exclusion():
    matches, scores = find_peer_matches(
        [ch("a", "distinct copied sentence passage"), ch("b", "distinct copied sentence passage")], 0.8
    )
    assert len(matches) == 2 and all(m["match_type"] == "exact" for m in matches)
    assert scores["a"] > 0
    assert all(m["query_document_id"] != m["source_document_id"] for m in matches)


def test_unrelated_documents_do_not_match():
    matches, _scores = find_peer_matches([ch("a", "blue ocean"), ch("b", "mountain stone")], 0.8)
    assert not matches


def test_cross_modal_locations_preserved():
    pdf = Chunk("w", "w", "w.pdf", "pdf", 0, "spoken distinctive lumen sentence", "spoken distinctive lumen sentence", {"page": 2})
    video = Chunk(
        "v",
        "v",
        "v.mp4",
        "mp4",
        0,
        "spoken distinctive lumen sentence",
        "spoken distinctive lumen sentence",
        {"start_seconds": 1.0, "end_seconds": 4.5},
    )
    matches, _ = find_peer_matches([pdf, video], 0.8)
    hit = next(m for m in matches if m["query_document_id"] == "w")
    assert hit["location"]["query_page"] == 2
    assert hit["location"]["matched_start_seconds"] == 1.0
    assert hit["matched_file"].endswith(".mp4")


def test_semantic_scores_are_used_when_provided():
    left = ch("a", "alpha wording here")
    right = ch("b", "completely different words")
    semantic = {(left.chunk_id, right.chunk_id): 0.91, (right.chunk_id, left.chunk_id): 0.91}
    matches, _ = find_peer_matches([left, right], 0.8, semantic_scores=semantic)
    assert matches and matches[0]["match_type"] == "semantic"


def test_boilerplate_does_not_drive_coverage():
    prompt = "Explain your selected case study, describe the central problem, and recommend a practical solution."
    answers = [
        " Independent mica bay tool libraries can cut household repair cost.",
        " Independent sunward farms should publish dry-season irrigation logs.",
        " Independent cedar quay ferries can post hourly occupancy counts.",
        " Independent northlamp clinics should share unused appointment slots.",
        " Independent velora mist gardens rotate salt-tolerant seedlings daily.",
    ]
    docs = [
        Document(str(i), f"{i}.docx", "docx", segments=[Segment(prompt + answers[i])])
        for i in range(5)
    ]
    detector = BoilerplateDetector(5, phrases=[prompt]).fit(docs)
    chunks = [ch("0", prompt + answers[0]), ch("1", prompt + answers[1])]
    matches, scores = find_peer_matches(chunks, 0.8, boilerplate=detector)
    assert scores.get("0", 0) == 0
    assert all(m.get("boilerplate_discounted") for m in matches) or not matches
