from generate_sample_data import write_silent_mp4
from integrity_engine.extractors.base import extract
from integrity_engine.pipeline import discover, run_pipeline


def test_zero_byte_is_distinguished(tmp_path):
    path = tmp_path / "zero.pdf"
    path.write_bytes(b"")
    doc = extract(path, tmp_path)
    assert doc.status == "empty" and doc.error == "zero_byte_file"


def test_bad_docx_does_not_raise(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_text("not a docx")
    doc = extract(path, tmp_path)
    assert doc.status == "failed" and ("Unreadable DOCX" in doc.error or "requires python-docx" in doc.error)


def test_corrupt_pdf_does_not_raise(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4 not actually a pdf")
    doc = extract(path, tmp_path)
    assert doc.status in {"failed", "empty"}
    assert doc.error or doc.warnings


def test_silent_mp4_does_not_crash(tmp_path):
    path = tmp_path / "silent.mp4"
    write_silent_mp4(path)
    doc = extract(path, tmp_path)
    assert doc.status in {"processed", "failed"}
    assert not any("Lumen Harbor" in seg.text for seg in doc.segments)


def test_missing_file_status(tmp_path):
    path = tmp_path / "gone.pdf"
    doc = extract(path, tmp_path)
    assert doc.status == "missing"


def test_pipeline_continues_after_corrupt_file(tmp_path):
    from generate_sample_data import write_docx, write_pdf

    write_pdf(tmp_path / "ok.pdf", "A distinctive original sentence about cedar boats.")
    (tmp_path / "zero.pdf").write_bytes(b"")
    write_docx(tmp_path / "ok.docx", "A different original sentence about lantern maps.")
    _docs, reports, summary = run_pipeline(tmp_path, tmp_path / "out", overrides={"offline": True})
    assert summary["discovered_documents"] == 3
    assert summary["empty_or_unextractable"] >= 1
    assert "ok.pdf" in reports and "ok.docx" in reports


def test_empty_directory_is_graceful(tmp_path):
    files = discover(tmp_path)
    assert files == []
