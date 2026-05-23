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

__all__ = [
    "TEXT_EMBEDDING_DIMENSION",
    "VertexImageEmbedder",
    "VertexTextEmbedder",
    "embed_image",
    "embed_text",
    "get_image_embedder",
    "get_text_embedder",
    "validate_embedding_dimension",
]
