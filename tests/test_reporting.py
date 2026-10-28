import json

from integrity_engine.reporting.json_report import write_json
from integrity_engine.reporting.markdown_report import write_markdown
from integrity_engine.reporting.summary import write_summary
from integrity_engine.schemas import Document


def test_reports_round_trip(tmp_path):
    report = {
        "document_id": "x",
        "status": "processed",
        "total_integrity_risk": "LOW",
        "scores": {
            "peer_similarity_percentage": 0,
            "external_similarity_percentage": 0,
            "ai_generated_probability": 0,
        },
        "peer_matches": [],
        "multimodal_matches": [],
        "external_matches": [],
        "explanations": ["none"],
        "warnings": [],
        "human_review_recommendation": "Human review is recommended before drawing conclusions.",
    }
    write_json(report, tmp_path / "r.json")
    write_markdown(report, tmp_path / "r.md")
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["schema_version"] == "1.0"
    assert "Human review" in (tmp_path / "r.md").read_text()


def test_summary_matrix_blanks_self_match(tmp_path):
    docs = [Document("a.pdf", "a.pdf", "pdf"), Document("b.docx", "b.docx", "docx")]
    reports = {
        "a.pdf": {
            "total_integrity_risk": "LOW",
            "scores": {"peer_similarity_percentage": 0, "external_similarity_percentage": 0, "ai_generated_probability": 1},
            "peer_matches": [{"source_document_id": "b.docx", "similarity_score": 0.9}],
            "multimodal_matches": [],
        },
        "b.docx": {
            "total_integrity_risk": "LOW",
            "scores": {"peer_similarity_percentage": 0, "external_similarity_percentage": 0, "ai_generated_probability": 1},
            "peer_matches": [],
            "multimodal_matches": [],
        },
    }
    write_summary(reports, docs, tmp_path, 0.12)
    matrix = (tmp_path / "similarity_matrix.csv").read_text().splitlines()
    assert matrix[1].startswith("a.pdf,")
    assert ",," in matrix[1] or matrix[1].split(",")[1] == ""
