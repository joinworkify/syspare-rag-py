"""Postgres connection to the shared Supabase project's syspare_rag_manuals table -- the
DB-backed replacement for manuals.json. Plain SQL, no ORM, matching the rest of this codebase's
style (and workify's own migration-driven schema on the same table).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from psycopg_pool import ConnectionPool

_pool: Optional[ConnectionPool] = None

_COLUMNS = (
    "manual_id, display_name, pdf_folder, cache_dir, image_dir, "
    "ocr_lang, description, organization_id, is_default"
)


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL is not set -- required for the manual registry (syspare_rag_manuals)."
            )
        # autocommit: every write here is a single statement, no multi-statement transactions
        # needed, so autocommit avoids every call site having to remember conn.commit().
        _pool = ConnectionPool(dsn, min_size=1, max_size=5, kwargs={"autocommit": True}, open=True)
    return _pool


def _row_to_dict(cols: List[str], row: tuple) -> Dict[str, Any]:
    return dict(zip(cols, row))


def list_manuals() -> List[Dict[str, Any]]:
    """Every manual row, unfiltered -- ManualRegistry applies org visibility in-memory (see
    ManualConfig.visible_to()); this just loads everything once at startup or on refresh."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM syspare_rag_manuals ORDER BY created_at")
            cols = [d.name for d in cur.description]
            return [_row_to_dict(cols, row) for row in cur.fetchall()]


def insert_manual(
    manual_id: str,
    display_name: str,
    pdf_folder: str,
    cache_dir: str,
    image_dir: str,
    ocr_lang: str = "eng",
    description: str = "",
    organization_id: Optional[str] = None,
) -> None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO syspare_rag_manuals
                    (manual_id, display_name, pdf_folder, cache_dir, image_dir,
                     ocr_lang, description, organization_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    manual_id,
                    display_name,
                    pdf_folder,
                    cache_dir,
                    image_dir,
                    ocr_lang,
                    description,
                    organization_id,
                ),
            )


def upsert_manual(
    manual_id: str,
    display_name: str,
    pdf_folder: str,
    cache_dir: str,
    image_dir: str,
    ocr_lang: str = "eng",
    description: str = "",
    organization_id: Optional[str] = None,
) -> None:
    """Insert-or-update by manual_id -- used by the one-time manuals.json migration script,
    which needs to be safely re-runnable."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO syspare_rag_manuals
                    (manual_id, display_name, pdf_folder, cache_dir, image_dir,
                     ocr_lang, description, organization_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (manual_id) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    pdf_folder = EXCLUDED.pdf_folder,
                    cache_dir = EXCLUDED.cache_dir,
                    image_dir = EXCLUDED.image_dir,
                    ocr_lang = EXCLUDED.ocr_lang,
                    description = EXCLUDED.description,
                    organization_id = EXCLUDED.organization_id,
                    updated_at = now()
                """,
                (
                    manual_id,
                    display_name,
                    pdf_folder,
                    cache_dir,
                    image_dir,
                    ocr_lang,
                    description,
                    organization_id,
                ),
            )


def delete_manual(manual_id: str) -> None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM syspare_rag_manuals WHERE manual_id = %s", (manual_id,))


def set_default(manual_id: str) -> None:
    """Marks exactly one manual as default, clearing the flag on every other row in one
    statement so there's never a moment with zero or multiple defaults set."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE syspare_rag_manuals SET is_default = (manual_id = %s)",
                (manual_id,),
            )
