"""Rebuild the RAG cache from PDFs in PDF_FOLDER.

Use after fixing index-layer bugs (e.g. P0-1) to ensure the on-disk cache is
consistent with current code.

Usage:
    # Rebuild a registered manual (preferred):
    uv run python scripts/rebuild_cache.py --manual-id YM358_service

    # Or override paths directly (legacy):
    uv run python scripts/rebuild_cache.py [--pdf-folder ./data] [--cache-dir ./cache]
                                           [--image-dir ./cache/images]
                                           [--ocr-lang mya+eng]
                                           [--no-ocr]

If both --manual-id and explicit paths are provided, the manual sets defaults
and the explicit flags override on top.
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
from syspare_rag.config import (  # noqa: E402
    load_manual_registry_from_db,
    load_rag_config_from_env,
)


def _resolve_credentials() -> None:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        return
    p = Path(creds)
    if not p.is_absolute():
        p = _REPO_ROOT / creds
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manual-id",
        default=None,
        help="Manual id from the ManualRegistry (e.g. YM358_operation, YM358_service).",
    )
    parser.add_argument("--pdf-folder", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--ocr-lang", default=None)
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback (useful when Tesseract is unavailable).",
    )
    parser.add_argument(
        "--list-manuals",
        action="store_true",
        help="Print registered manuals and exit (no rebuild).",
    )
    args = parser.parse_args()

    _resolve_credentials()

    if args.list_manuals:
        registry = load_manual_registry_from_db()
        print("Registered manuals:")
        for m in registry.list():
            print(
                f"  - {m.manual_id} ({m.display_name}) "
                f"[pdf={m.pdf_folder}, cache={m.cache_dir}, ocr_lang={m.ocr_lang}]"
            )
        print(f"Default: {registry.default_id}")
        return 0

    cfg = load_rag_config_from_env()

    manual = None
    if args.manual_id:
        registry = load_manual_registry_from_db()
        try:
            manual = registry.get(args.manual_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Run with --list-manuals to see available ids.", file=sys.stderr)
            return 2
        cfg.paths.pdf_folder = manual.pdf_folder
        cfg.paths.cache_dir = manual.cache_dir
        cfg.paths.image_dir = manual.image_dir
        cfg.image_save_dir = manual.image_dir
        cfg.ocr_lang = manual.ocr_lang or cfg.ocr_lang

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

    label = f"manual={manual.manual_id}" if manual else "manual=<none>"
    print(f"Rebuilding cache: {label}")
    print(f"  pdf_folder = {pdf_folder}")
    print(f"  cache_dir  = {cache_dir}")
    print(f"  image_dir  = {image_dir}")
    print(f"  ocr        = {cfg.enable_ocr_fallback} ({cfg.ocr_lang})")
    print(f"  text_embed = {cfg.text_embedding_model} ({cfg.text_embedding_dimension}d)")
    print(f"  img_embed  = {cfg.embedding_model_name} ({cfg.embedding_size}d)")

    if not Path(pdf_folder).exists():
        print(f"Error: pdf_folder {pdf_folder} does not exist.", file=sys.stderr)
        return 2

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
