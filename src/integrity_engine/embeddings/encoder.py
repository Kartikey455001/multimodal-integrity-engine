"""Lazy sentence-transformer encoder with optional disk cache. No fake semantic vectors."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ..exceptions import OptionalDependencyError

LOG = logging.getLogger(__name__)


class SentenceEncoder:
    _models: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        cached = [self._read_cache(text) for text in texts]
        missing_idx = [i for i, vec in enumerate(cached) if vec is None]
        if missing_idx:
            model = self._load()
            missing_texts = [texts[i] if texts[i].strip() else " " for i in missing_idx]
            vec = np.asarray(
                model.encode(
                    missing_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            if vec.ndim != 2 or not np.isfinite(vec).all():
                raise ValueError("Encoder returned invalid vectors")
            for local, source_i in enumerate(missing_idx):
                cached[source_i] = vec[local]
                self._write_cache(texts[source_i], vec[local])
        matrix = np.vstack(cached).astype(np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Encoder returned invalid vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (matrix / norms).astype(np.float32)

    def _load(self):
        if self._model is not None:
            return self._model
        key = (self.model_name, self.device)
        if key in SentenceEncoder._models:
            self._model = SentenceEncoder._models[key]
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "Semantic embeddings require sentence-transformers. "
                "Install the ml extra (`pip install sentence-transformers`) and allow a one-time model download. "
                "Until then the engine uses a documented lexical overlap matcher, not paraphrase detection."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise OptionalDependencyError(
                f"Could not load embedding model '{self.model_name}': {exc}. "
                "Use --offline for lexical matching, or download all-MiniLM-L6-v2 while online."
            ) from exc
        SentenceEncoder._models[key] = self._model
        LOG.info("Loaded embedding model %s on %s", self.model_name, self.device)
        return self._model

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _read_cache(self, text: str) -> np.ndarray | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        try:
            vec = np.load(path)
            if vec.ndim == 1 and np.isfinite(vec).all():
                return vec.astype(np.float32)
        except (OSError, ValueError):
            return None
        return None

    def _write_cache(self, text: str, vec: np.ndarray) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vec.astype(np.float32))
            meta = path.with_suffix(".json")
            if not meta.exists():
                meta.write_text(json.dumps({"model": self.model_name}), encoding="utf-8")
        except OSError:
            LOG.debug("Could not write embedding cache for a chunk")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Embedding dimensions do not match")
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.clip(an @ bn.T, -1, 1)
