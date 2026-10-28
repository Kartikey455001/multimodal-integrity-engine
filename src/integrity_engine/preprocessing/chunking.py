"""Deterministic sentence-aware chunking with per-segment provenance."""

from __future__ import annotations

import re

from ..schemas import Chunk, Segment
from .normalization import comparison_text, normalize_text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_segments(
    document_id: str,
    filename: str,
    source_format: str,
    segments: list[Segment],
    target_words: int = 260,
    overlap_words: int = 45,
) -> list[Chunk]:
    """Chunk each provenance segment independently so locations are not merged.
    
    For MP4 video, consecutive Whisper transcript slices are grouped into coherent
    chunks up to target_words while preserving start_seconds and end_seconds.
    For documents, paragraph boundaries are respected so common assignment prompt
    headers are separated from analytical student text.
    """
    out: list[Chunk] = []

    # Handle MP4 speech transcript slices
    if source_format == "mp4" and any(seg.start is not None for seg in segments):
        valid_segs = [s for s in segments if normalize_text(s.original_text or s.text)]
        if not valid_segs:
            return []
        groups: list[list[Segment]] = []
        current_group: list[Segment] = []
        current_count = 0
        for seg in valid_segs:
            text = normalize_text(seg.original_text or seg.text)
            n = len(text.split())
            if current_group and current_count + n > target_words:
                groups.append(current_group)
                keep: list[Segment] = []
                kept = 0
                for prior in reversed(current_group):
                    pn = len(normalize_text(prior.original_text or prior.text).split())
                    if kept + pn > overlap_words:
                        break
                    keep.insert(0, prior)
                    kept += pn
                current_group = list(keep)
                current_count = kept
            current_group.append(seg)
            current_count += n
        if current_group:
            groups.append(current_group)

        for group in groups:
            original = " ".join(s.original_text if s.original_text is not None else s.text for s in group).strip()
            if not original:
                continue
            start_sec = float(min(s.start for s in group if s.start is not None))
            end_sec = float(max(s.end for s in group if s.end is not None))
            loc = {
                "start_seconds": start_sec,
                "end_seconds": end_sec,
            }
            confidences = [s.location.get("confidence") for s in group if s.location.get("confidence")]
            if confidences:
                loc["confidence"] = confidences[0]
            seq = len(out)
            out.append(
                Chunk(
                    chunk_id=f"{document_id}::chunk-{seq:04d}",
                    document_id=document_id,
                    filename=filename,
                    source_format=source_format,
                    sequence=seq,
                    original_text=original,
                    normalized_text=comparison_text(original),
                    location=loc,
                )
            )
        return out

    # Handle written documents (PDF, DOCX, references)
    for segment in segments:
        display = segment.original_text if segment.original_text is not None else segment.text
        raw = normalize_text(display)
        if not raw:
            continue

        raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", display) if p.strip()]
        if len(raw_paras) > 1:
            for p_idx, para in enumerate(raw_paras):
                p_raw = normalize_text(para)
                if not p_raw:
                    continue
                p_sentences = split_sentences(p_raw)
                p_groups: list[list[str]] = []
                p_curr: list[str] = []
                p_count = 0
                for s in p_sentences:
                    sn = len(s.split())
                    if p_curr and p_count + sn > target_words:
                        p_groups.append(p_curr)
                        p_keep: list[str] = []
                        p_kept = 0
                        for prior in reversed(p_curr):
                            pn = len(prior.split())
                            if p_kept + pn > overlap_words:
                                break
                            p_keep.insert(0, prior)
                            p_kept += pn
                        p_curr, p_count = p_keep, p_kept
                    p_curr.append(s)
                    p_count += sn
                if p_curr:
                    p_groups.append(p_curr)
                p_loc = dict(segment.location)
                if "paragraph" not in p_loc:
                    p_loc["paragraph"] = p_idx + 1
                for words in p_groups:
                    orig = " ".join(words).strip()
                    if not orig:
                        continue
                    seq = len(out)
                    out.append(
                        Chunk(
                            chunk_id=f"{document_id}::chunk-{seq:04d}",
                            document_id=document_id,
                            filename=filename,
                            source_format=source_format,
                            sequence=seq,
                            original_text=orig,
                            normalized_text=comparison_text(orig),
                            location=p_loc,
                        )
                    )
            continue

        sentences = split_sentences(raw)
        groups: list[list[str]] = []
        current: list[str] = []
        count = 0
        for sentence in sentences:
            n = len(sentence.split())
            if current and count + n > target_words:
                groups.append(current)
                keep: list[str] = []
                kept = 0
                for prior in reversed(current):
                    pn = len(prior.split())
                    if kept + pn > overlap_words:
                        break
                    keep.insert(0, prior)
                    kept += pn
                current, count = keep, kept
            current.append(sentence)
            count += n
        if current:
            groups.append(current)
        location = dict(segment.location)
        if segment.start is not None:
            location.setdefault("start_seconds", float(segment.start))
        if segment.end is not None:
            location.setdefault("end_seconds", float(segment.end))
        for words in groups:
            original = " ".join(words).strip()
            if not original:
                continue
            seq = len(out)
            out.append(
                Chunk(
                    chunk_id=f"{document_id}::chunk-{seq:04d}",
                    document_id=document_id,
                    filename=filename,
                    source_format=source_format,
                    sequence=seq,
                    original_text=original,
                    normalized_text=comparison_text(original),
                    location=location,
                )
            )
    return out
