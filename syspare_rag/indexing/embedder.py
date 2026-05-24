"""Vertex AI embedding wrappers with a single text embedding space."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as VisionImage
from vertexai.vision_models import MultiModalEmbeddingModel

from syspare_rag._retry import retry_call
from syspare_rag.config import (
    IMAGE_EMBEDDING_DIMENSION_DEFAULT,
    IMAGE_EMBEDDING_MODEL,
    TEXT_EMBEDDING_DIMENSION,
    TEXT_EMBEDDING_MODEL,
)

Vector = Union[List[float], np.ndarray]

_text_embedder: Optional["VertexTextEmbedder"] = None
_image_embedder: Optional["VertexImageEmbedder"] = None


def validate_embedding_dimension(
    vector: Vector,
    expected: int,
    *,
    label: str = "embedding",
) -> None:
    length = len(vector)  # type: ignore[arg-type]
    if length != expected:
        raise ValueError(
            f"{label} dimension mismatch: expected {expected}, got {length}"
        )


class VertexTextEmbedder:
    """Embeds text with text-embedding-004 (768d) for all text retrieval paths."""

    model_name = TEXT_EMBEDDING_MODEL
    dimension = TEXT_EMBEDDING_DIMENSION

    def __init__(self, model_name: str = TEXT_EMBEDDING_MODEL) -> None:
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        embeddings = retry_call(
            self._model.get_embeddings,
            [text],
            label="text-embedding",
        )
        vector = list(embeddings[0].values)
        validate_embedding_dimension(vector, self.dimension, label="text embedding")
        return vector

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = retry_call(
            self._model.get_embeddings,
            list(texts),
            label="text-embedding-batch",
        )
        vectors: List[List[float]] = []
        for embedding in embeddings:
            vector = list(embedding.values)
            validate_embedding_dimension(vector, self.dimension, label="text embedding")
            vectors.append(vector)
        return vectors


class VertexImageEmbedder:
    """Embeds image pixels with multimodalembedding (configurable dimension)."""

    model_name = IMAGE_EMBEDDING_MODEL

    def __init__(
        self,
        model_name: str = IMAGE_EMBEDDING_MODEL,
        default_dimension: int = IMAGE_EMBEDDING_DIMENSION_DEFAULT,
    ) -> None:
        self._model = MultiModalEmbeddingModel.from_pretrained(model_name)
        self.default_dimension = default_dimension

    def embed_image(
        self,
        image_uri: str,
        *,
        embedding_size: Optional[int] = None,
        contextual_text: Optional[str] = None,
    ) -> List[float]:
        size = embedding_size or self.default_dimension
        image = VisionImage.load_from_file(image_uri)
        embeddings = retry_call(
            self._model.get_embeddings,
            image=image,
            contextual_text=contextual_text,
            dimension=size,
            label="image-embedding",
        )
        return list(embeddings.image_embedding)


def get_text_embedder() -> VertexTextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = VertexTextEmbedder()
    return _text_embedder


def get_image_embedder(
    *,
    model_name: str = IMAGE_EMBEDDING_MODEL,
    default_dimension: int = IMAGE_EMBEDDING_DIMENSION_DEFAULT,
) -> VertexImageEmbedder:
    global _image_embedder
    if _image_embedder is None:
        _image_embedder = VertexImageEmbedder(
            model_name=model_name,
            default_dimension=default_dimension,
        )
    return _image_embedder


def embed_text(text: str) -> List[float]:
    return get_text_embedder().embed_text(text)


def embed_image(
    image_uri: str,
    *,
    embedding_size: Optional[int] = None,
    contextual_text: Optional[str] = None,
) -> List[float]:
    return get_image_embedder(default_dimension=embedding_size or IMAGE_EMBEDDING_DIMENSION_DEFAULT).embed_image(
        image_uri,
        embedding_size=embedding_size,
        contextual_text=contextual_text,
    )
