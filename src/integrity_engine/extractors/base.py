"""File discovery helpers and per-file extraction dispatch."""

from __future__ import annotations

from pathlib import Path

from ..schemas import Document


def document_id(path: Path, root: Path) -> str:
    """Stable ID based on the POSIX relative path, avoiding basename collisions."""
    return path.relative_to(root).as_posix()


def extract(
    path: Path,
    root: Path,
    video_model: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    offline: bool = False,
) -> Document:
    fmt = path.suffix.lower().lstrip(".")
    doc = Document(document_id(path, root), path.relative_to(root).as_posix(), fmt)
    if not path.exists():
        doc.status = "missing"
        doc.error = "file_not_found"
        return doc
    if path.stat().st_size == 0:
        doc.status = "empty"
        doc.error = "zero_byte_file"
        return doc
    try:
        if fmt == "pdf":
            from .pdf_extractor import extract_pdf

            doc.segments, doc.warnings = extract_pdf(path)
        elif fmt == "docx":
            from .docx_extractor import extract_docx

            doc.segments, doc.warnings = extract_docx(path)
        elif fmt == "mp4":
            from .video_extractor import extract_video

            doc.segments, doc.warnings = extract_video(
                path, video_model, device, compute_type, language, offline=offline
            )
        else:
            doc.status = "failed"
            doc.error = "unsupported_extension"
    except Exception as exc:
        doc.status = "failed"
        doc.error = f"{type(exc).__name__}: {exc}"
        doc.warnings.append(doc.error)

    if doc.status == "processed" and not any(s.text.strip() for s in doc.segments):
        if fmt == "mp4":
            # Inspected video with no transcript is not the same as a missing or corrupt file.
            doc.status = "processed"
            if not any("inspected" in w.lower() or "silent" in w.lower() or "no audio" in w.lower() for w in doc.warnings):
                doc.warnings.append("No transcript text was produced after inspection.")
        else:
            doc.status = "empty"
            if fmt == "pdf":
                doc.warnings.append("No embedded text found; scanned PDFs require optional OCR.")
    return doc
