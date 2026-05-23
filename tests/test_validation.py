import pandas as pd
import pytest

from syspare_rag.indexing.validation import (
    assert_uniform_dimension,
    column_dimensions,
    inferred_image_pixel_dim,
    validate_image_index,
    validate_text_index,
)


def _df_with_vectors(vectors):
    return pd.DataFrame({"text_embedding_chunk": vectors})


def test_column_dimensions_counts_lengths():
    df = _df_with_vectors([[0.0] * 768, [0.0] * 768, [0.0] * 1408])
    counts = column_dimensions(df, "text_embedding_chunk")
    assert counts == {768: 2, 1408: 1}


def test_assert_uniform_dimension_passes_when_consistent():
    df = _df_with_vectors([[0.0] * 768, [0.1] * 768])
    assert_uniform_dimension(df, "text_embedding_chunk", expected=768)


def test_assert_uniform_dimension_raises_on_mixed_dims():
    df = _df_with_vectors([[0.0] * 768, [0.0] * 1408])
    with pytest.raises(ValueError, match="mixed embedding dimensions"):
        assert_uniform_dimension(df, "text_embedding_chunk", expected=768)


def test_assert_uniform_dimension_ignores_missing_column():
    df = pd.DataFrame({"other": [1, 2, 3]})
    assert_uniform_dimension(df, "missing_col", expected=768)


def test_validate_text_index_catches_p0_1_regression():
    df = pd.DataFrame(
        {
            "text_embedding_chunk": [[0.0] * 768, [0.0] * 1408],
            "extraction_method": ["native", "ocr"],
        }
    )
    with pytest.raises(ValueError):
        validate_text_index(df, text_embedding_dim=768)


def test_validate_image_index_with_explicit_dim():
    df = pd.DataFrame(
        {
            "text_embedding_from_image_description": [[0.0] * 768] * 3,
            "mm_embedding_from_img_only": [[0.0] * 128] * 3,
        }
    )
    validate_image_index(df, text_embedding_dim=768, image_embedding_dim=128)
    with pytest.raises(ValueError):
        validate_image_index(df, text_embedding_dim=768, image_embedding_dim=1408)


def test_validate_image_index_infers_dim_when_not_provided():
    df = pd.DataFrame(
        {
            "text_embedding_from_image_description": [[0.0] * 768] * 2,
            "mm_embedding_from_img_only": [[0.0] * 128, [0.0] * 1408],
        }
    )
    with pytest.raises(ValueError, match="mixed dimensions"):
        validate_image_index(df, text_embedding_dim=768, image_embedding_dim=None)


def test_inferred_image_pixel_dim():
    df = pd.DataFrame({"mm_embedding_from_img_only": [[0.0] * 128] * 5})
    assert inferred_image_pixel_dim(df) == 128
    empty = pd.DataFrame({"mm_embedding_from_img_only": []})
    assert inferred_image_pixel_dim(empty) is None
