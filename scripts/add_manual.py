#!/usr/bin/env python3
"""Add a new manual to the registry.

Usage:
    uv run python scripts/add_manual.py <manual_id> [options]

Examples:
    uv run python scripts/add_manual.py AW82_service
    uv run python scripts/add_manual.py AW82_service --name "AW82 Service Manual" --lang mya+eng
    uv run python scripts/add_manual.py AW82_service --build   # create dirs + add config + build index
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve project root (scripts/ is one level below root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENV_FILE = ROOT / ".env"
DEFAULT_JSON_PATH = ROOT / "manuals" / "manuals.json"


def _read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


def _write_env_key(key: str, value: str) -> None:
    text = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"# {key}="):
            lines[i] = f"{key}={value}"
            ENV_FILE.write_text("\n".join(lines) + "\n")
            return
    with ENV_FILE.open("a") as f:
        f.write(f"\n{key}={value}\n")


def _load_manuals(json_path: Path) -> list[dict]:
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    # Bootstrap from _DEFAULT_MANUALS in config.py
    from syspare_rag.config import _DEFAULT_MANUALS
    return list(_DEFAULT_MANUALS)


def _save_manuals(json_path: Path, manuals: list[dict]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(manuals, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a new manual to the registry")
    parser.add_argument("manual_id", help="Unique manual ID (e.g. AW82_service)")
    parser.add_argument("--name", help="Display name (default: manual_id with underscores replaced)")
    parser.add_argument("--lang", default="eng", help="OCR language (default: eng)")
    parser.add_argument("--desc", default="", help="Short description")
    parser.add_argument(
        "--build",
        action="store_true",
        help="After adding config, trigger build-cache via API (server must be running)",
    )
    parser.add_argument("--port", default="8000", help="Server port for --build (default: 8000)")
    args = parser.parse_args()

    manual_id: str = args.manual_id
    display_name: str = args.name or manual_id.replace("_", " ").title()

    # Resolve MANUALS_JSON path
    env = _read_env()
    manuals_json_path = Path(env["MANUALS_JSON"]) if "MANUALS_JSON" in env else DEFAULT_JSON_PATH
    if not manuals_json_path.is_absolute():
        manuals_json_path = ROOT / manuals_json_path

    # Load existing manuals
    manuals = _load_manuals(manuals_json_path)

    # Check duplicate
    if any(m["manual_id"] == manual_id for m in manuals):
        print(f"[skip] '{manual_id}' already in registry ({manuals_json_path})")
    else:
        entry = {
            "manual_id": manual_id,
            "display_name": display_name,
            "pdf_folder": f"manuals/{manual_id}/pdf",
            "cache_dir": f"manuals/{manual_id}/cache",
            "image_dir": f"manuals/{manual_id}/cache/images",
            "ocr_lang": args.lang,
            "description": args.desc,
        }
        manuals.append(entry)
        _save_manuals(manuals_json_path, manuals)
        print(f"[added] '{manual_id}' → {manuals_json_path}")

    # Ensure MANUALS_JSON is set in .env
    if "MANUALS_JSON" not in env:
        rel = manuals_json_path.relative_to(ROOT)
        _write_env_key("MANUALS_JSON", str(rel))
        print(f"[.env]  MANUALS_JSON={rel}")

    # Create directory structure
    for subdir in ("pdf", "cache", "cache/images"):
        d = ROOT / "manuals" / manual_id / subdir
        d.mkdir(parents=True, exist_ok=True)
    print(f"[dirs]  manuals/{manual_id}/{{pdf,cache,cache/images}} ready")

    # Optionally trigger build
    if args.build:
        import urllib.request
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
