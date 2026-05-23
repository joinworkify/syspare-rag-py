import numpy as np
import pytest

from syspare_rag.retrieval.similarity import cosine_similarity, row_cosine_similarity


def test_cosine_similarity_identical_vectors():
    vec = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_row_cosine_similarity():
    class Row:
        def __init__(self, embedding):
            self._embedding = embedding

        def __getitem__(self, key):
            if key == "vec":
                return self._embedding
            raise KeyError(key)

    row = Row([1.0, 0.0])
    score = row_cosine_similarity(row, "vec", [1.0, 0.0])
    assert score == pytest.approx(1.0)
