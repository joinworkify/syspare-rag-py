"""Vector similarity helpers for retrieval."""

from __future__ import annotations

from typing import Any, Union

import numpy as np

VectorLike = Union[list[float], np.ndarray]


def _to_vector(value: VectorLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def cosine_similarity(a: VectorLike, b: VectorLike) -> float:
    """L2-normalized cosine similarity in [-1, 1]. Returns 0.0 for zero vectors."""
    vec_a = _to_vector(a)
    vec_b = _to_vector(b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def row_cosine_similarity(row: Any, column_name: str, query_vector: VectorLike) -> float:
    """Cosine similarity between a DataFrame row embedding and a query vector."""
    stored = row[column_name]
    if stored is None:
        return 0.0
    return round(cosine_similarity(stored, query_vector), 4)
