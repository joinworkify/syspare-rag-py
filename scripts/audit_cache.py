"""Audit a RAG cache for embedding dimension consistency and norms.

Usage:
    uv run python scripts/audit_cache.py [--cache-dir ./cache]

Reports:
  - Row counts per (column, extraction_method)
  - Embedding dimension distribution per column
  - L2-norm summary stats (helps confirm whether dot product == cosine)
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List

import numpy as np
import pandas as pd

TEXT_EMBED_COLS = ("text_embedding_chunk", "text_embedding_page")
IMAGE_EMBED_COLS = (
    "text_embedding_from_image_description",
    "mm_embedding_from_img_only",
)


def _vector_len(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _norms(values: Iterable) -> List[float]:
    norms: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float64)
        except Exception:
            continue
        if arr.size == 0:
            continue
        norms.append(float(np.linalg.norm(arr)))
    return norms


def _describe_norms(label: str, norms: List[float]) -> None:
    if not norms:
        print(f"  {label}: no vectors")
        return
    print(
        f"  {label}: n={len(norms)} "
        f"min={min(norms):.4f} mean={mean(norms):.4f} "
        f"median={median(norms):.4f} max={max(norms):.4f}"
    )


def _describe_dims(label: str, lengths: List[int | None]) -> None:
    filtered = [l for l in lengths if l is not None]
    counter = Counter(filtered)
    none_count = len(lengths) - len(filtered)
    print(f"  {label}: total_rows={len(lengths)} missing={none_count}")
    for dim, count in sorted(counter.items()):
        print(f"    dim={dim}: {count} row(s)")


def audit_text_df(df: pd.DataFrame) -> None:
    print(f"\n[text_metadata_df] rows={len(df)} columns={list(df.columns)}")
    if "extraction_method" in df.columns:
        method_counts = df["extraction_method"].fillna("native").value_counts().to_dict()
        print(f"  extraction_method counts: {method_counts}")

    for col in TEXT_EMBED_COLS:
        if col not in df.columns:
            continue
        print(f"\n  -- column: {col}")
        lengths = df[col].map(_vector_len).tolist()
        _describe_dims("dim distribution", lengths)
        norms = _norms(df[col].tolist())
        _describe_norms("L2 norms", norms)

        if "extraction_method" in df.columns:
            for method, sub in df.groupby(df["extraction_method"].fillna("native")):
                sub_lengths = sub[col].map(_vector_len).tolist()
                _describe_dims(f"dim distribution [{method}]", sub_lengths)
                sub_norms = _norms(sub[col].tolist())
                _describe_norms(f"L2 norms [{method}]", sub_norms)


def audit_image_df(df: pd.DataFrame) -> None:
    print(f"\n[image_metadata_df] rows={len(df)} columns={list(df.columns)}")
    for col in IMAGE_EMBED_COLS:
        if col not in df.columns:
            continue
        print(f"\n  -- column: {col}")
        lengths = df[col].map(_vector_len).tolist()
        _describe_dims("dim distribution", lengths)
        norms = _norms(df[col].tolist())
        _describe_norms("L2 norms", norms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CACHE_DIR", "./cache"),
        help="Path to RAG cache directory (default: env CACHE_DIR or ./cache)",
    )
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    text_pkl = cache / "text_metadata_df.pkl"
    image_pkl = cache / "image_metadata_df.pkl"

    if not text_pkl.exists() or not image_pkl.exists():
        print(f"Cache not found at {cache}")
        return 1

    text_df: pd.DataFrame = pd.read_pickle(text_pkl)
    image_df: pd.DataFrame = pd.read_pickle(image_pkl)

    print(f"Auditing cache at: {cache.resolve()}")
    audit_text_df(text_df)
    audit_image_df(image_df)

    print("\nInterpretation hints:")
    print("  - L2 norms near 1.0 => vectors are already unit-normalized")
    print("    (dot product == cosine in that case).")
    print("  - Mixed dimensions in the SAME column => P0-1 bug present;")
    print("    rebuild required to make those rows retrievable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
