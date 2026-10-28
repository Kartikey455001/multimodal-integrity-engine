"""Environment validation script.

Run before using the engine to verify required and optional dependencies.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def _check(name: str, import_name: str | None = None, version_attr: str = "__version__") -> tuple[bool, str]:
    module = import_name or name
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, version_attr, "unknown")
        return True, str(ver)
    except ImportError:
        return False, "NOT INSTALLED"


def _cmd(cmd: str) -> tuple[bool, str]:
    path = shutil.which(cmd)
    if not path:
        return False, "NOT ON PATH"
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            input="",
        )
        first_line = (result.stdout or result.stderr or "").splitlines()
        return True, first_line[0].strip() if first_line else "ok"
    except subprocess.TimeoutExpired:
        return True, "found (version check timed out)"
    except Exception as exc:
        return True, f"found (version check failed: {exc})"


def main() -> int:
    print(f"Python: {sys.version}")
    print()

    checks = [
        ("PyMuPDF", "pymupdf", True),
        ("python-docx", "docx", True),
        ("pydantic", "pydantic", True),
        ("PyYAML", "yaml", True),
        ("numpy", "numpy", True),
        ("scikit-learn", "sklearn", True),
        ("pytest", "pytest", True),
        ("av (PyAV)", "av", False),
        ("sentence-transformers", "sentence_transformers", False),
        ("faster-whisper", "faster_whisper", False),
    ]

    all_required_ok = True
    for display_name, import_name, required in checks:
        ok, version = _check(display_name, import_name)
        tag = "[REQUIRED]" if required else "[optional]"
        status = "OK" if ok else "MISSING"
        if required and not ok:
            all_required_ok = False
        print(f"  {tag} {display_name}: {status} ({version})")

    print()
    print("External tools:")
    for tool in ("ffmpeg", "ffprobe", "espeak-ng", "espeak"):
        ok, info = _cmd(tool)
        req = "required for video" if tool in ("ffmpeg", "ffprobe") else "optional TTS"
        print(f"  {tool} ({req}): {'FOUND' if ok else 'MISSING'} - {info}")

    print()
    corpus = Path(__file__).resolve().parents[1] / "reference_corpus" / "references.json"
    print(f"  Reference corpus: {'OK' if corpus.exists() else 'MISSING'} - {corpus}")

    print()
    if all_required_ok:
        print("All required dependencies are present. Run `python generate_sample_data.py` to create fixtures.")
        return 0
    else:
        print(
            "One or more required dependencies are missing. "
            "Run: python -m pip install -r requirements.txt"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
