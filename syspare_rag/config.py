"""Central configuration for the RAG index layer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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


# -----------------------------
# Per-manual configuration
# -----------------------------
@dataclass
class ManualConfig:
    """A single indexed manual with its own PDF folder, cache, and image dir.

    Local layout convention:
        manuals/<manual_id>/pdf/
        manuals/<manual_id>/cache/
        manuals/<manual_id>/cache/images/

    Matching S3 layout convention:
        {S3_RAG_PREFIX}/manuals/<manual_id>/pdf/
        {S3_RAG_PREFIX}/manuals/<manual_id>/cache/
    """

    manual_id: str
    display_name: str
    pdf_folder: str
    cache_dir: str
    image_dir: str
    ocr_lang: str = "eng"
    description: str = ""

    @property
    def s3_prefix_segment(self) -> str:
        """Path segment used under {S3_RAG_PREFIX}/manuals/<segment>/."""
        return self.manual_id


class ManualRegistry:
    """In-memory registry of all available manuals."""

    def __init__(
        self, manuals: List[ManualConfig], default_id: Optional[str] = None
    ) -> None:
        if not manuals:
            raise ValueError("ManualRegistry requires at least one ManualConfig")
        self._manuals: Dict[str, ManualConfig] = {m.manual_id: m for m in manuals}
        if default_id is not None and default_id not in self._manuals:
            raise ValueError(
                f"default_id={default_id!r} not in manuals {list(self._manuals)}"
            )
        self._default_id = default_id or manuals[0].manual_id

    def list(self) -> List[ManualConfig]:
        return list(self._manuals.values())

    def get(self, manual_id: Optional[str]) -> ManualConfig:
        """Return ManualConfig for manual_id, or default when None/empty."""
        if not manual_id:
            return self._manuals[self._default_id]
        if manual_id not in self._manuals:
            raise KeyError(
                f"Unknown manual_id={manual_id!r}; available: {list(self._manuals)}"
            )
        return self._manuals[manual_id]

    @property
    def default(self) -> ManualConfig:
        return self._manuals[self._default_id]

    @property
    def default_id(self) -> str:
        return self._default_id


_DEFAULT_MANUALS = [
    {
        "manual_id": "YM358_operation",
        "display_name": "YM358 Operation Manual",
        "pdf_folder": "manuals/YM358_operation/pdf",
        "cache_dir": "manuals/YM358_operation/cache",
        "image_dir": "manuals/YM358_operation/cache/images",
        "ocr_lang": "eng",
        "description": "Operator-facing manual for YM351R/YM358R tractors.",
    },
    {
        "manual_id": "YM358_service",
        "display_name": "YM358 Service Manual",
        "pdf_folder": "manuals/YM358_service/pdf",
        "cache_dir": "manuals/YM358_service/cache",
        "image_dir": "manuals/YM358_service/cache/images",
        "ocr_lang": "eng",
        "description": "Scanned service/repair manual; requires OCR for most pages.",
    },
    {
        "manual_id": "AW82_service",
        "display_name": "AW82 Service Manual",
        "pdf_folder": "manuals/AW82_service/pdf",
        "cache_dir": "manuals/AW82_service/cache",
        "image_dir": "manuals/AW82_service/cache/images",
        "ocr_lang": "eng",
        "description": "Service manual for AW82.",
    },
    {
        "manual_id": "YHCH_service",
        "display_name": "YHCH Service Manual",
        "pdf_folder": "manuals/YHCH_service/pdf",
        "cache_dir": "manuals/YHCH_service/cache",
        "image_dir": "manuals/YHCH_service/cache/images",
        "ocr_lang": "eng",
        "description": "Service manual for YHCH.",
    },
    {
        "manual_id": "YH_operation",
        "display_name": "YH Operation Manual",
        "pdf_folder": "manuals/YH_operation/pdf",
        "cache_dir": "manuals/YH_operation/cache",
        "image_dir": "manuals/YH_operation/cache/images",
        "ocr_lang": "eng",
        "description": "Operator-facing manual for YH",
    },
]


def load_manual_registry_from_env() -> ManualRegistry:
    """Build a ManualRegistry, optionally overridden by env vars.

    Override mechanism:
      - MANUALS_JSON: path to a JSON file with [{manual_id, display_name, pdf_folder,
        cache_dir, image_dir, ocr_lang, description}, ...]
      - DEFAULT_MANUAL_ID: id to mark as default (else first entry).
    """
    manuals_json = env("MANUALS_JSON")
    if manuals_json:
        path = Path(manuals_json)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read MANUALS_JSON={manuals_json}: {exc}")
    else:
        payload = _DEFAULT_MANUALS

    manuals = [
        ManualConfig(
            manual_id=str(m["manual_id"]),
            display_name=str(m.get("display_name", m["manual_id"])),
            pdf_folder=str(m["pdf_folder"]),
            cache_dir=str(m["cache_dir"]),
            image_dir=str(m.get("image_dir", str(Path(m["cache_dir"]) / "images"))),
            ocr_lang=str(m.get("ocr_lang", env("OCR_LANG", "eng"))),
            description=str(m.get("description", "")),
        )
        for m in payload
    ]
    default_id = env("DEFAULT_MANUAL_ID") or None
    return ManualRegistry(manuals, default_id=default_id)
