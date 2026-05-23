"""Rebuild the RAG cache from PDFs in PDF_FOLDER.

Use after fixing index-layer bugs (e.g. P0-1) to ensure the on-disk cache is
consistent with current code.

Usage:
    uv run python scripts/rebuild_cache.py [--pdf-folder ./data] [--cache-dir ./cache]
                                           [--image-dir ./cache/images]
                                           [--ocr-lang mya+eng]
                                           [--no-ocr]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from vertexai.generative_models import GenerationConfig  # noqa: E402

from pipeline import MultimodalRAGPipeline  # noqa: E402
from syspare_rag.config import load_rag_config_from_env  # noqa: E402


def _resolve_credentials() -> None:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        return
    p = Path(creds)
    if not p.is_absolute():
        p = _REPO_ROOT / creds
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-folder", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--ocr-lang", default=None)
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback (useful when Tesseract is unavailable).",
    )
    args = parser.parse_args()

    _resolve_credentials()

    cfg = load_rag_config_from_env()
    if args.pdf_folder:
        cfg.paths.pdf_folder = args.pdf_folder
    if args.cache_dir:
        cfg.paths.cache_dir = args.cache_dir
    if args.image_dir:
        cfg.paths.image_dir = args.image_dir
        cfg.image_save_dir = args.image_dir
    if args.ocr_lang:
        cfg.ocr_lang = args.ocr_lang
    if args.no_ocr:
        cfg.enable_ocr_fallback = False

    pdf_folder = cfg.paths.pdf_folder
    cache_dir = cfg.paths.cache_dir
    image_dir = cfg.image_save_dir

    print(f"Rebuilding cache:")
    print(f"  pdf_folder = {pdf_folder}")
    print(f"  cache_dir  = {cache_dir}")
    print(f"  image_dir  = {image_dir}")
    print(f"  ocr        = {cfg.enable_ocr_fallback} ({cfg.ocr_lang})")
    print(f"  text_embed = {cfg.text_embedding_model} ({cfg.text_embedding_dimension}d)")
    print(f"  img_embed  = {cfg.embedding_model_name} ({cfg.embedding_size}d)")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    rag = MultimodalRAGPipeline(cfg)
    rag.build_metadata(
        pdf_folder_path=pdf_folder,
        cache_dir=cache_dir,
        force_rebuild=True,
        generation_config=GenerationConfig(temperature=0.2),
        ocr_fallback=cfg.enable_ocr_fallback,
        image_save_dir=image_dir,
    )

    text_rows = len(rag.text_metadata_df) if rag.text_metadata_df is not None else 0
    image_rows = len(rag.image_metadata_df) if rag.image_metadata_df is not None else 0
    print(f"\nRebuild complete: {text_rows} text chunk(s), {image_rows} image(s).")
    print(f"Run: uv run python scripts/audit_cache.py --cache-dir {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
