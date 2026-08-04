#!/usr/bin/env python3
"""Add a new manual to the registry (syspare_rag_manuals table).

Usage:
    uv run python scripts/add_manual.py <manual_id> [options]

Examples:
    uv run python scripts/add_manual.py AW82_service
    uv run python scripts/add_manual.py AW82_service --name "AW82 Service Manual" --lang mya+eng
    uv run python scripts/add_manual.py AW82_service --build   # create dirs + add row + build index
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve project root (scripts/ is one level below root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from syspare_rag import db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a new manual to the registry")
    parser.add_argument("manual_id", help="Unique manual ID (e.g. AW82_service)")
    parser.add_argument("--name", help="Display name (default: manual_id with underscores replaced)")
    parser.add_argument("--lang", default="eng", help="OCR language (default: eng)")
    parser.add_argument("--desc", default="", help="Short description")
    parser.add_argument("--org", default=None, help="Organization id (omit for a global/shared manual)")
    parser.add_argument(
        "--build",
        action="store_true",
        help="After adding the row, trigger build-cache via API (server must be running)",
    )
    parser.add_argument("--port", default="8000", help="Server port for --build (default: 8000)")
    args = parser.parse_args()

    manual_id: str = args.manual_id
    display_name: str = args.name or manual_id.replace("_", " ").title()

    existing = {m["manual_id"] for m in db.list_manuals()}
    if manual_id in existing:
        print(f"[skip] '{manual_id}' already in syspare_rag_manuals")
    else:
        db.insert_manual(
            manual_id=manual_id,
            display_name=display_name,
            pdf_folder=f"manuals/{manual_id}/pdf",
            cache_dir=f"manuals/{manual_id}/cache",
            image_dir=f"manuals/{manual_id}/cache/images",
            ocr_lang=args.lang,
            description=args.desc,
            organization_id=args.org,
        )
        print(f"[added] '{manual_id}' -> syspare_rag_manuals")

    # Create directory structure
    for subdir in ("pdf", "cache", "cache/images"):
        d = ROOT / "manuals" / manual_id / subdir
        d.mkdir(parents=True, exist_ok=True)
    print(f"[dirs]  manuals/{manual_id}/{{pdf,cache,cache/images}} ready")

    # Optionally trigger build
    if args.build:
        import urllib.request

        # The running server's in-memory registry only sees a new DB row after
        # /api/sync-registry or the periodic 60s refresh -- force it now so build-cache
        # doesn't 404 on a manual_id it hasn't loaded yet.
        sync_url = f"http://localhost:{args.port}/api/sync-registry"
        try:
            urllib.request.urlopen(urllib.request.Request(sync_url, method="POST"), timeout=10)
        except Exception as e:
            print(f"[sync-registry] failed: {e} (is server running?)")

        url = f"http://localhost:{args.port}/api/build-cache?manual_id={manual_id}"
        print(f"[build] POST {url}")
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(resp.read().decode())
        except Exception as e:
            print(f"[build] failed: {e} (is server running?)")


if __name__ == "__main__":
    main()
