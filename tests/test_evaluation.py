from integrity_engine.evaluation import evaluate_pairwise


def test_metrics_are_computed_from_manifest_pairs():
    manifest = {
        "documents": [
            {"document_id": "a.pdf", "category": "exact_copy"},
            {"document_id": "b.docx", "category": "exact_copy"},
            {"document_id": "c.pdf", "category": "original"},
        ],
        "positive_pairs": [["a.pdf", "b.docx"]],
    }
    reports = {"a.pdf": {"peer_matches": [{"query_document_id": "a.pdf", "source_document_id": "b.docx"}]}}
    result = evaluate_pairwise([], reports, manifest)
    assert result["true_positives"] == 1 and result["false_negatives"] == 0
    assert result["pairs_evaluated"] == 3
