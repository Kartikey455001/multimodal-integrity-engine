import subprocess
import sys
from pathlib import Path

from generate_sample_data import write_docx, write_pdf
from integrity_engine.pipeline import discover, run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_discovery_order_and_output_exclusion(tmp_path):
    (tmp_path / "z.pdf").write_bytes(b"x")
    (tmp_path / "a.docx").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp4").write_bytes(b"x")
    out = tmp_path / "reports"
    out.mkdir()
    (out / "fake.pdf").write_bytes(b"x")
    assert [p.name for p in discover(tmp_path, out)] == ["a.docx", "b.mp4", "z.pdf"]


def test_cli_help():
    result = subprocess.run([sys.executable, str(ROOT / "run_integrity_check.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--input_dir" in result.stdout
    assert "--peer_threshold" in result.stdout


def test_cli_empty_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_integrity_check.py"),
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--offline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "No supported" in result.stdout


def test_deterministic_repeat(tmp_path):
    write_pdf(tmp_path / "a.pdf", "Copied distinctive harbor lockers sentence for tests.")
    write_docx(tmp_path / "b.docx", "Copied distinctive harbor lockers sentence for tests.")
    first = run_pipeline(tmp_path, tmp_path / "out1", overrides={"offline": True})[1]
    second = run_pipeline(tmp_path, tmp_path / "out2", overrides={"offline": True})[1]
    assert first["a.pdf"]["scores"] == second["a.pdf"]["scores"]
    assert first["a.pdf"]["peer_matches"][0]["matched_file"] == "b.docx"


def test_self_match_not_reported(tmp_path):
    write_pdf(tmp_path / "solo.pdf", "Unique velora mist gardens never copied here.")
    _docs, reports, _summary = run_pipeline(tmp_path, tmp_path / "out", overrides={"offline": True})
    assert reports["solo.pdf"]["peer_matches"] == []
