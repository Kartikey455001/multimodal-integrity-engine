"""Extractor tests. Video transcription is mocked; FFmpeg absence is simulated."""

from unittest.mock import patch

import pytest

from generate_sample_data import write_docx, write_pdf, write_silent_mp4
from integrity_engine.exceptions import OptionalDependencyError
from integrity_engine.extractors.base import extract
from integrity_engine.extractors.pdf_extractor import extract_pdf
from integrity_engine.extractors.video_extractor import extract_video


def test_pdf_page_provenance(tmp_path):
    path = tmp_path / "page.pdf"
    write_pdf(path, "First page sentence one.\n")
    segments, _warnings = extract_pdf(path)
    assert segments and segments[0].location["page"] == 1


def test_docx_paragraph_and_invalid_file(tmp_path):
    path = tmp_path / "ok.docx"
    write_docx(path, "Paragraph one.\nParagraph two.")
    doc = extract(path, tmp_path)
    assert doc.status == "processed"
    assert {seg.location.get("paragraph") for seg in doc.segments} >= {1, 2}
    bad = tmp_path / "bad.docx"
    bad.write_text("not a docx")
    failed = extract(bad, tmp_path)
    assert failed.status == "failed"


def test_missing_ffmpeg_is_actionable(tmp_path):
    path = tmp_path / "clip.mp4"
    write_silent_mp4(path)
    with patch("integrity_engine.extractors.video_extractor._validate_container", return_value=None):
        with patch("integrity_engine.extractors.video_extractor._has_audio_stream", return_value=True):
            with patch(
                "integrity_engine.extractors.video_extractor._extract_wav",
                side_effect=OptionalDependencyError("MP4 transcription requires FFmpeg on PATH."),
            ):
                with pytest.raises(OptionalDependencyError):
                    extract_video(path)


def test_transcription_failure_is_reported(tmp_path):
    path = tmp_path / "clip.mp4"
    write_silent_mp4(path)
    doc = extract(path, tmp_path)
    assert doc.status in {"processed", "failed"}
    if doc.status == "failed":
        assert doc.error
    else:
        assert doc.segments == [] or doc.warnings
