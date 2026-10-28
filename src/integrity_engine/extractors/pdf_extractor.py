"""PyMuPDF page-level extraction with OCR limitation warnings."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import OptionalDependencyError
from ..preprocessing.normalization import normalize_text, repeated_header_footer_lines
from ..schemas import Segment


def extract_pdf(path: Path) -> tuple[list[Segment], list[str]]:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise OptionalDependencyError("PDF extraction requires PyMuPDF: pip install PyMuPDF") from exc

    warnings: list[str] = []
    segments: list[Segment] = []
    try:
        with fitz.open(path) as pdf:
            if pdf.is_encrypted and not pdf.authenticate(""):
                raise ValueError("PDF is encrypted and cannot be opened without a password")
            pages: list[str] = []
            for idx, page in enumerate(pdf):
                raw = page.get_text("text") or ""
                pages.append(raw)
            headers = repeated_header_footer_lines(pages)
            if headers:
                warnings.append("Repeated header/footer lines were detected and recorded; they remain in evidence.")
            for page_idx, page in enumerate(pdf):
                blocks = page.get_text("blocks") or []
                para_num = 1
                page_segments: list[Segment] = []
                for b in blocks:
                    if len(b) > 6 and b[6] != 0:
                        continue
                    raw_block = b[4] if len(b) > 4 else ""
                    display = normalize_text(raw_block)
                    if not display:
                        continue
                    location = {"page": page_idx + 1, "paragraph": para_num}
                    if headers:
                        location["possible_headers_or_footers"] = sorted(headers)
                    page_segments.append(Segment(text=display, location=location, original_text=raw_block))
                    para_num += 1
                if page_segments:
                    segments.extend(page_segments)
                elif pages[page_idx]:
                    display = normalize_text(pages[page_idx])
                    if display:
                        location = {"page": page_idx + 1, "paragraph": 1}
                        if headers:
                            location["possible_headers_or_footers"] = sorted(headers)
                        segments.append(Segment(text=display, location=location, original_text=pages[page_idx]))
    except OptionalDependencyError:
        raise
    except Exception as exc:
        raise ValueError(f"Unreadable PDF: {exc}") from exc

    if not segments:
        warnings.append("No extractable embedded text; OCR is not enabled. Scanned PDFs require optional OCR.")
    return segments, warnings
