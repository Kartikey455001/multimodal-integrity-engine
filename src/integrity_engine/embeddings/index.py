"""Exact normalized NumPy similarity backend (replaceable later with FAISS/pgvector)."""

from __future__ import annotations

import numpy as np

from .encoder import cosine_matrix


class ExactIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int = 5) -> list[list[tuple[int, float]]]:
        scores = cosine_matrix(query, self.vectors)
        ranked: list[list[tuple[int, float]]] = []
        for row in scores:
            order = sorted(enumerate(row.tolist()), key=lambda x: (-float(x[1]), x[0]))[:k]
            ranked.append([(idx, float(score)) for idx, score in order])
        return ranked
