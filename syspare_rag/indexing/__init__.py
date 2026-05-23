"""Index-time embedding helpers."""

from syspare_rag.indexing.embedder import (
    TEXT_EMBEDDING_DIMENSION,
    VertexImageEmbedder,
    VertexTextEmbedder,
    embed_image,
    embed_text,
    get_image_embedder,
    get_text_embedder,
    validate_embedding_dimension,
)
from syspare_rag.indexing.validation import (
    assert_uniform_dimension,
    column_dimensions,
    inferred_image_pixel_dim,
    validate_image_index,
    validate_text_index,
)

__all__ = [
    "TEXT_EMBEDDING_DIMENSION",
    "VertexImageEmbedder",
    "VertexTextEmbedder",
    "assert_uniform_dimension",
    "column_dimensions",
    "embed_image",
    "embed_text",
    "get_image_embedder",
    "get_text_embedder",
    "inferred_image_pixel_dim",
    "validate_embedding_dimension",
    "validate_image_index",
    "validate_text_index",
]
