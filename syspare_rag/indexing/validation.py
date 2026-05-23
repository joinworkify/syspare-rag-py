"""Index-time validation helpers.

Catches the P0-1 class of bugs (mixed embedding dimensions in the same column)
at index-build and cache-load time, instead of silently degrading retrieval.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _vector_len(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def column_dimensions(df: pd.DataFrame, column: str) -> Counter:
    """Return a Counter mapping vector length -> row count for an embedding column."""
    if column not in df.columns:
        return Counter()
    lengths = [l for l in (_vector_len(v) for v in df[column]) if l is not None]
    return Counter(lengths)


def assert_uniform_dimension(
    df: pd.DataFrame,
    column: str,
    expected: int,
    *,
    label: str = "",
) -> None:
    """Raise ValueError if any row in `column` has a vector of dim != expected.

    Empty/missing vectors are ignored (caller decides how to handle them).
    """
    dims = column_dimensions(df, column)
    if not dims:
        return
    bad = {d: c for d, c in dims.items() if d != expected}
    if bad:
        prefix = f"{label}: " if label else ""
        raise ValueError(
            f"{prefix}column '{column}' has mixed embedding dimensions; "
            f"expected {expected}, found {dict(bad)}. "
            "Rebuild the cache to fix this."
        )


def validate_text_index(
    df: pd.DataFrame,
    *,
    text_embedding_dim: int,
    chunk_column: str = "text_embedding_chunk",
    page_column: str = "text_embedding_page",
) -> None:
    """Validate dimensions of text embedding columns in a text metadata DataFrame."""
    assert_uniform_dimension(df, chunk_column, text_embedding_dim, label="text_metadata_df")
    if page_column in df.columns:
        assert_uniform_dimension(df, page_column, text_embedding_dim, label="text_metadata_df")


def validate_image_index(
    df: pd.DataFrame,
    *,
    text_embedding_dim: int,
    image_embedding_dim: Optional[int] = None,
    caption_column: str = "text_embedding_from_image_description",
    pixel_column: str = "mm_embedding_from_img_only",
) -> None:
    """Validate dimensions of image embedding columns in an image metadata DataFrame.

    `image_embedding_dim` may be None — image pixel dim is sometimes inferred from
    cache to support legacy 128d caches alongside newer 1408d builds. If provided,
    it is enforced.
    """
    assert_uniform_dimension(df, caption_column, text_embedding_dim, label="image_metadata_df")

    if image_embedding_dim is not None:
        assert_uniform_dimension(df, pixel_column, image_embedding_dim, label="image_metadata_df")
    else:
        # Without an expected dim, still require that all rows agree with each other.
        dims = column_dimensions(df, pixel_column)
        if len(dims) > 1:
            raise ValueError(
                f"image_metadata_df: column '{pixel_column}' has mixed dimensions "
                f"{dict(dims)}. Rebuild the cache to fix this."
            )


def inferred_image_pixel_dim(
    df: pd.DataFrame,
    pixel_column: str = "mm_embedding_from_img_only",
) -> Optional[int]:
    """Return the embedding dim found in the cache, or None if column is missing/empty."""
    dims = column_dimensions(df, pixel_column)
    if not dims:
        return None
    return max(dims, key=dims.get)
