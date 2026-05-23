"""Central configuration for the RAG index layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# Vertex text embeddings used for chunks, captions, queries, and OCR text.
TEXT_EMBEDDING_MODEL = "text-embedding-004"
TEXT_EMBEDDING_DIMENSION = 768

# Vertex multimodal embeddings used for image pixel vectors only.
IMAGE_EMBEDDING_MODEL = "multimodalembedding@001"
IMAGE_EMBEDDING_DIMENSION_DEFAULT = 1408


@dataclass
class PathSettings:
    pdf_folder: str = "./data/"
    cache_dir: str = "./cache"
    image_dir: str = "./cache/images"


@dataclass
class RagConfig:
    project_id: str
    location: str = "us-central1"

    # Gemini generation
    model_name: str = "gemini-2.0-flash"

    # Text embeddings (chunks, captions, queries, OCR)
    text_embedding_model: str = TEXT_EMBEDDING_MODEL
    text_embedding_dimension: int = TEXT_EMBEDDING_DIMENSION

    # Image pixel embeddings
    embedding_size: int = IMAGE_EMBEDDING_DIMENSION_DEFAULT
    embedding_model_name: str = IMAGE_EMBEDDING_MODEL

    # Extraction
    image_save_dir: str = "images"
    image_description_prompt: str = (
        "Explain what is going on in the image.\n"
        "If it's a table, extract all elements of the table.\n"
        "If it's a graph, explain the findings in the graph.\n"
        "Do not include any numbers that are not mentioned in the image.\n"
    )

    # OCR fallback
    enable_ocr_fallback: bool = True
    ocr_min_chars: int = 40
    ocr_dpi: int = 200
    ocr_lang: str = "eng"
    ocr_chunk_chars: int = 1200
    ocr_chunk_overlap: int = 150

    paths: PathSettings = field(default_factory=PathSettings)


def load_rag_config_from_env(
    *,
    project_id: str | None = None,
    location: str | None = None,
) -> RagConfig:
    """Build RagConfig from environment variables (Render / local .env)."""
    paths = PathSettings(
        pdf_folder=env("PDF_FOLDER", "./data/"),
        cache_dir=env("CACHE_DIR", "./cache"),
        image_dir=env("IMAGE_DIR", "./cache/images"),
    )
    return RagConfig(
        project_id=project_id or env("PROJECT_ID", "fortunaii"),
        location=location or env("LOCATION", "us-central1"),
        image_save_dir=paths.image_dir,
        ocr_lang=env("OCR_LANG", "mya+eng"),
        paths=paths,
    )
