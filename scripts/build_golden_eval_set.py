"""Build golden RAG regression fixtures from feedback CSV exports.

Examples:
    uv run python scripts/build_golden_eval_set.py
    uv run python scripts/build_golden_eval_set.py --csv path/to/logs_rag_chat.csv

The historical CSV is one row per assistant response. It has no stable
conversation/session key, so this extractor treats prompt-answer pairs as the
canonical regression unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "AI Service Response Feedback - logs_rag_chat.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "eval" / "golden_rag" / "golden_pairs.jsonl"
DEFAULT_STATS = REPO_ROOT / "artifacts" / "eval" / "golden_rag" / "golden_stats.json"


def _load_json_cell(value: str) -> List[Dict[str, Any]]:
    raw = (value or "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else []


def _feedback_kind(feedback: str) -> str:
    raw = (feedback or "").strip()
    low = raw.lower()
    if not raw:
        return "blank"
    if low in {"perfect", "perect", "percect"}:
        return "perfect"
    if "exceptionally perfect" in low:
        return "perfect"
    if low.startswith(("perfect", "perect", "percect")):
        return "perfect_with_note"
    if any(
        marker in low
        for marker in [
            "mistook",
            "unrelevant",
            "wrong",
            "not useful",
            "did not understand",
            "did not undersatand",
            "can't find",
            "opearation manual",
        ]
    ):
        return "bad"
    if any(
        marker in low
        for marker in [
            "=",
            "should understand",
            "should note",
            "translate",
            "same word",
            "track roller",
            "oil seal",
            "threshing drum",
            "crawler",
            "feeder",
        ]
    ):
        return "terminology_issue"
    return "other"


def _infer_manual_id(texts: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> Optional[str]:
    docs = [
        str(item.get("doc") or "").lower()
        for item in [*texts, *images]
        if item.get("doc")
    ]
    joined = " ".join(docs)
    if "ym358a" in joined:
        return "YM358_operation"
    if "ym358s" in joined:
        return "YM358_service"
    if "ch" in joined:
        return "CH_manual"
    if "yh" in joined:
        return "YH_operation"
    return None


def _doc_names(texts: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> List[str]:
    names = {
        str(item.get("doc")).strip()
        for item in [*texts, *images]
        if str(item.get("doc") or "").strip()
    }
    return sorted(names)


def _extract_expected_terms(feedback: str) -> List[str]:
    """Extract lightweight glossary hints from human feedback notes.

    These are advisory hints for reports; LLM judging still receives the full
    feedback note because Myanmar glossary corrections are often contextual.
    """
    terms: List[str] = []
    for _, rhs in re.findall(r"([^=,;]+)=([^,.;\\n]+)", feedback or ""):
        candidate = rhs.strip()
        if candidate:
            terms.append(candidate)
    for quoted in re.findall(r"as ([^,.]+)", feedback or "", flags=re.IGNORECASE):
        candidate = quoted.strip()
        if candidate:
            terms.append(candidate)
    return sorted(set(terms))


def _make_record(row: Dict[str, str], kind: str) -> Optional[Dict[str, Any]]:
    texts = _load_json_cell(row.get("texts", ""))
    images = _load_json_cell(row.get("images", ""))
    feedback = (row.get("Recommended Response") or "").strip()

    if kind == "perfect":
        tier = "tier_a"
        record_type = "golden_pair"
        mandatory = True
    elif kind == "perfect_with_note":
        tier = "tier_b"
        record_type = "golden_pair"
        mandatory = False
    elif kind == "terminology_issue":
        tier = "tier_c"
        record_type = "terminology_constraint"
        mandatory = False
    elif kind == "bad":
        tier = "negative"
        record_type = "negative_example"
        mandatory = False
    else:
        return None

    return {
        "id": row.get("id", ""),
        "slug": row.get("slug", ""),
        "tier": tier,
        "record_type": record_type,
        "mandatory": mandatory,
        "question": row.get("question", ""),
        "accepted_answer": row.get("answer", ""),
        "feedback": feedback,
        "expected_terms": _extract_expected_terms(feedback),
        "texts": texts,
        "images": images,
        "doc_names": _doc_names(texts, images),
        "manual_id": _infer_manual_id(texts, images),
        "created_at": row.get("created_at", ""),
        "user_name": row.get("user_name", ""),
    }


def read_feedback_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_golden_set(csv_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_feedback_csv(csv_path)
    kinds = Counter(_feedback_kind(row.get("Recommended Response", "")) for row in rows)
    records = [
        record
        for row in rows
        if (record := _make_record(row, _feedback_kind(row.get("Recommended Response", ""))))
    ]
    record_counts = Counter(record["tier"] for record in records)
    stats = {
        "source_csv": str(csv_path),
        "total_rows": len(rows),
        "feedback_kind_counts": dict(kinds),
        "exported_record_counts": dict(record_counts),
        "notes": {
            "tier_a": "Clean perfect prompt-answer pairs for mandatory regression gates.",
            "tier_b": "Accepted answers with human notes; advisory unless notes are encoded as hard requirements.",
            "tier_c": "Terminology/translation constraints; judge against the feedback note rather than old answer similarity.",
            "negative": "Known bad/context-failure examples; not used for golden answer matching.",
        },
    }
    return records, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Feedback CSV export path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL fixture path.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS, help="Output stats JSON path.")
    args = parser.parse_args()

    records, stats = build_golden_set(args.csv)
    write_jsonl(records, args.output)
    write_json(stats, args.stats)

    print(f"Wrote {len(records)} golden eval records to {args.output}")
    print(json.dumps(stats["exported_record_counts"], ensure_ascii=False, sort_keys=True))
    print(f"Wrote stats to {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
