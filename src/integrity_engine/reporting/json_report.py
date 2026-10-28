"""Versioned JSON report writing with schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import DocumentReport


def write_json(report: dict, path: Path) -> dict:
    payload = {"schema_version": report.get("schema_version", "1.0"), **report}
    validated = DocumentReport.model_validate(payload).model_dump()
    Path(path).write_text(json.dumps(validated, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return validated
