"""One-time migration: manuals.json (or _DEFAULT_MANUALS) -> syspare_rag_manuals Postgres table.

Idempotent -- safe to re-run (upsert by manual_id). Marks whichever entry matches
DEFAULT_MANUAL_ID as is_default, same fallback order load_manual_registry_from_env() used.

Usage:
    DATABASE_URL=... MANUALS_JSON=manuals/manuals.json uv run python scripts/migrate_manuals_to_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from syspare_rag import db
from syspare_rag.config import env, load_manual_registry_from_env


def main() -> None:
    registry = load_manual_registry_from_env()
    default_id = env("DEFAULT_MANUAL_ID") or registry.default_id

    manuals = registry.list()
    print(f"Migrating {len(manuals)} manual(s) from manuals.json/_DEFAULT_MANUALS -> syspare_rag_manuals...")

    for manual in manuals:
        db.upsert_manual(
            manual_id=manual.manual_id,
            display_name=manual.display_name,
            pdf_folder=manual.pdf_folder,
            cache_dir=manual.cache_dir,
            image_dir=manual.image_dir,
            ocr_lang=manual.ocr_lang,
            description=manual.description,
            organization_id=manual.organization_id,
        )
        marker = " (default)" if manual.manual_id == default_id else ""
        print(f"  upserted {manual.manual_id}{marker}")

    if default_id:
        db.set_default(default_id)
        print(f"Marked default manual: {default_id}")

    rows = db.list_manuals()
    print(f"Done. syspare_rag_manuals now has {len(rows)} row(s).")


if __name__ == "__main__":
    main()
