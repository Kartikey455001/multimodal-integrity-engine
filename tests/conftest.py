import sys
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path(request):
    """Use an isolated workspace temp dir (the host temp directory can be sandboxed)."""
    path = ROOT / ".test-tmp" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
