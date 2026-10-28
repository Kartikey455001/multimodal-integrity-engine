"""Human-readable Markdown reports with ranked evidence."""

from __future__ import annotations

from pathlib import Path


def _match_block(match: dict) -> list[str]:
    title = match.get("matched_file") or match.get("reference_title") or "match"
    loc = match.get("location") or {}
    return [
        f"### {title} ({match.get('match_type')}, similarity {match.get('similarity_score')})",
        match.get("explanation") or "",
        f"Query location: `{match.get('query_location')}`; source location: `{match.get('source_location')}`",
        f"Flattened location: `{loc}`",
        f"Boilerplate discounted: {match.get('boilerplate_discounted', False)}",
        "",
        f"> Query: {match.get('query_passage', '')}",
        "",
        f"> Source: {match.get('matched_passage', '')}",
        "",
    ]


def write_markdown(report: dict, path: Path) -> None:
    scores = report["scores"]
    lines = [
        f"# Integrity report: {report['document_id']}",
        "",
        f"- **Status:** {report['status']}",
        f"- **Risk:** {report['total_integrity_risk']}",
        f"- **Peer similarity:** {scores['peer_similarity_percentage']}%",
        f"- **External similarity:** {scores['external_similarity_percentage']}%",
        f"- **AI-content indicator:** {scores['ai_generated_probability']} / 100 (uncalibrated heuristic)",
        f"- **Human review:** {report.get('human_review_recommendation', 'Recommended.')}",
        "",
        "## Explanation",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("explanations", []))
    for title, key in (
        ("Peer matches", "peer_matches"),
        ("Cross-modal matches", "multimodal_matches"),
        ("External references", "external_matches"),
    ):
        lines += ["", f"## {title}", ""]
        items = report.get(key) or []
        if not items:
            lines.append("No qualifying evidence reported.")
        for match in items:
            lines.extend(_match_block(match))
    lines += ["", "## Warnings and limitations", ""]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No extractor warnings were recorded.")
    lines += [
        "",
        "## Review guidance",
        "",
        report.get(
            "human_review_recommendation",
            "Human review is recommended before drawing conclusions. Similarity and style indicators are screening signals, not findings of misconduct.",
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
