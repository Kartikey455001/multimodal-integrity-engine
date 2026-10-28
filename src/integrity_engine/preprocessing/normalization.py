"""Conservative text normalization. Original evidence is retained separately."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\t\r\f\v ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def comparison_text(text: str) -> str:
    """Lowercased punctuation-light form for matching. Numbers and words remain."""
    text = normalize_text(text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def repeated_header_footer_lines(pages: list[str], min_pages: int = 3) -> set[str]:
    """Optional PDF header/footer detection: identical first/last lines across pages."""
    if len(pages) < min_pages:
        return set()
    first = Counter()
    last = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        first[lines[0]] += 1
        last[lines[-1]] += 1
    threshold = max(min_pages, int(0.6 * len(pages)))
    repeated = {line for line, n in first.items() if n >= threshold}
    repeated |= {line for line, n in last.items() if n >= threshold}
    return repeated
