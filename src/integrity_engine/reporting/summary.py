"""Aggregate JSON, Markdown, CSV, and peer similarity matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..schemas import Document


def write_summary(reports: dict, docs: list[Document], output: Path, duration: float) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = list(reports.values())
    risks = {label: sum(item.get("total_integrity_risk") == label for item in records) for label in ("LOW", "MEDIUM", "HIGH")}
    summary = {
        "discovered_documents": len(docs),
        "successfully_processed": sum(doc.status == "processed" for doc in docs),
        "failed": sum(doc.status == "failed" for doc in docs),
        "missing": sum(doc.status == "missing" for doc in docs),
        "empty_or_unextractable": sum(doc.status == "empty" for doc in docs),
        "peer_matches": sum(len(item.get("peer_matches") or []) for item in records),
        "cross_modal_matches": sum(len(item.get("multimodal_matches") or []) for item in records),
        "risk_level_counts": risks,
        "processing_duration_seconds": round(duration, 3),
        "important_warnings": [f"{doc.document_id}: {doc.error}" for doc in docs if doc.error],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = ["# Run summary", ""]
    for key, value in summary.items():
        markdown.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    (out / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["document_id", "status", "risk", "peer_similarity", "external_similarity", "ai_content_indicator"]
        )
        for doc in docs:
            report = reports.get(doc.document_id, {})
            scores = report.get("scores", {})
            writer.writerow(
                [
                    doc.document_id,
                    doc.status,
                    report.get("total_integrity_risk"),
                    scores.get("peer_similarity_percentage"),
                    scores.get("external_similarity_percentage"),
                    scores.get("ai_generated_probability"),
                ]
            )

    ids = [doc.document_id for doc in sorted(docs, key=lambda item: item.document_id)]
    with (out / "similarity_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document_id"] + ids)
        for left in ids:
            row = [left]
            for right in ids:
                if left == right:
                    row.append("")
                    continue
                score = next(
                    (
                        match["similarity_score"]
                        for match in reports.get(left, {}).get("peer_matches", [])
                        if match.get("source_document_id") == right
                    ),
                    0,
                )
                row.append(score)
            writer.writerow(row)
    return summary
