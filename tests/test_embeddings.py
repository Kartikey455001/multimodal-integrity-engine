import numpy as np
import pytest

from integrity_engine.embeddings.encoder import cosine_matrix
from integrity_engine.embeddings.index import ExactIndex
from integrity_engine.exceptions import OptionalDependencyError


def test_cosine_and_dimensions():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_matrix(matrix, matrix)[0, 0] == pytest.approx(1)
    with pytest.raises(ValueError):
        cosine_matrix(matrix, np.ones((1, 3)))


def test_exact_index_ranks_self_highest():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = ExactIndex(vectors)
    result = index.search(vectors[0:1], k=2)
    assert result[0][0][0] == 0
    assert result[0][0][1] == pytest.approx(1.0)


def test_encoder_raises_on_bad_model(monkeypatch):
    """Verify that a missing model raises OptionalDependencyError without network access.

    We monkeypatch the SentenceTransformer constructor to raise immediately so
    the test does not attempt a real download.
    """
    from integrity_engine.embeddings import encoder as enc_module

    def _mock_st(model_name, device="cpu"):
        raise OSError(f"No such model: {model_name}")

    # Patch the SentenceTransformer import inside the encoder module.
    import sys
    import types

    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _mock_st
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    # Clear model cache so test is isolated.
    enc_module.SentenceEncoder._models.clear()

    from integrity_engine.embeddings.encoder import SentenceEncoder

    encoder = SentenceEncoder(model_name="definitely-missing-model")
    with pytest.raises(OptionalDependencyError) as exc_info:
        encoder.encode(["hello"])
    assert "Could not load" in str(exc_info.value) or "sentence-transformers" in str(exc_info.value)
