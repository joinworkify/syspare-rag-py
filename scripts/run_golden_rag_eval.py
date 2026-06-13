"""Run golden prompt-answer regression evals against the current RAG server.

Examples:
    uv run python scripts/build_golden_eval_set.py
    uv run python scripts/run_golden_rag_eval.py --limit 5 --skip-judge
    uv run python scripts/run_golden_rag_eval.py --tiers tier_a tier_b

By default this script calls a local FastAPI server at http://127.0.0.1:8000.
Start it with:
    uv run uvicorn rag_server:app --reload --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "artifacts" / "eval" / "golden_rag" / "golden_pairs.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "eval" / "golden_rag" / "runs"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

PASSING_OVERALL_SCORE = 4


JUDGE_JSON_SCHEMA = {
    "pass": "boolean",
    "overall_score": "integer 1-5",
    "factual_regression": "boolean",
    "safety_regression": "boolean",
    "terminology_regression": "boolean",
    "missing_key_points": "array of strings",
    "new_unsupported_claims": "array of strings",
    "rationale": "string",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                records.append(json.loads(raw))
    return records


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _post_json(url: str, payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body[:1000]}") from exc


def run_current_answer(
    record: Dict[str, Any],
    *,
    server_url: str,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "session_id": f"golden-eval-{record.get('id')}",
        "question": record["question"],
        "history": [],
        "manual_id": record.get("manual_id"),
        "answer_language": "auto",
        "log_response": False,
    }
    data = _post_json(f"{server_url.rstrip('/')}/api/chat", payload, timeout=timeout)
    return {
        "answer": data.get("answer", ""),
        "texts": data.get("texts", []),
        "images": data.get("images", []),
        "manual_id": data.get("manual_id"),
        "log_id": data.get("log_id"),
        "slug": data.get("slug"),
    }


def _configure_vertex() -> None:
    load_dotenv(REPO_ROOT / ".env")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds:
        cred_path = Path(creds)
        if not cred_path.is_absolute():
            cred_path = REPO_ROOT / cred_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path.resolve())

    import vertexai

    vertexai.init(
        project=os.environ.get("PROJECT_ID", "fortunaii"),
        location=os.environ.get("LOCATION", "us-central1"),
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start : end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


def _coerce_judge_result(payload: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    score = int(payload.get("overall_score") or 0)
    score = max(1, min(5, score))
    result = {
        "pass": bool(payload.get("pass")),
        "overall_score": score,
        "factual_regression": bool(payload.get("factual_regression")),
        "safety_regression": bool(payload.get("safety_regression")),
        "terminology_regression": bool(payload.get("terminology_regression")),
        "missing_key_points": list(payload.get("missing_key_points") or []),
        "new_unsupported_claims": list(payload.get("new_unsupported_claims") or []),
        "rationale": str(payload.get("rationale") or "").strip(),
    }
    if record.get("mandatory"):
        result["pass"] = (
            result["overall_score"] >= PASSING_OVERALL_SCORE
            and not result["factual_regression"]
            and not result["safety_regression"]
            and not result["terminology_regression"]
        )
    return result


def build_judge_prompt(record: Dict[str, Any], new_answer: str) -> str:
    if record.get("record_type") == "terminology_constraint":
        task = (
            "Evaluate whether the new answer respects the human terminology or translation "
            "feedback. Do not require it to match the old answer if the old answer was flawed."
        )
    else:
        task = (
            "Compare the old accepted answer and the new answer for the same farmer prompt. "
            "The old answer was marked as correct by a human reviewer."
        )

    return textwrap.dedent(
        f"""
        You are a strict evaluator for a tractor manual RAG system.

        Task:
        {task}

        Passing criteria:
        - Overall score must be at least {PASSING_OVERALL_SCORE} for mandatory golden pairs.
        - No critical factual regression.
        - No safety regression.
        - No required terminology/translation regression.
        - The new answer may be phrased differently if it preserves the same practical guidance.

        Return ONLY valid JSON with this schema:
        {json.dumps(JUDGE_JSON_SCHEMA, ensure_ascii=False, indent=2)}

        Record metadata:
        - id: {record.get("id")}
        - tier: {record.get("tier")}
        - record_type: {record.get("record_type")}
        - manual_id: {record.get("manual_id")}

        Farmer prompt:
        {record.get("question", "")}

        Human feedback note:
        {record.get("feedback", "")}

        Expected terms or glossary hints:
        {json.dumps(record.get("expected_terms") or [], ensure_ascii=False)}

        Old accepted answer:
        \"\"\"
        {record.get("accepted_answer", "")}
        \"\"\"

        New candidate answer:
        \"\"\"
        {new_answer}
        \"\"\"
        """
    ).strip()


def judge_answer(
    record: Dict[str, Any],
    new_answer: str,
    *,
    model: Any,
    generation_config: Any,
) -> Dict[str, Any]:
    raw = model.generate_content(
        build_judge_prompt(record, new_answer),
        generation_config=generation_config,
    )
    text = getattr(raw, "text", "") or str(raw)
    return _coerce_judge_result(_parse_json_object(text), record)


def deterministic_term_check(record: Dict[str, Any], new_answer: str) -> Dict[str, Any]:
    terms = [str(term).strip() for term in record.get("expected_terms") or [] if str(term).strip()]
    if not terms:
        return {"checked": False, "missing_terms": []}
    low_answer = new_answer.lower()
    missing = [term for term in terms if term.lower() not in low_answer]
    return {"checked": True, "missing_terms": missing}


def fallback_judge(record: Dict[str, Any], new_answer: str) -> Dict[str, Any]:
    """Deterministic fallback for dry runs; not a replacement for LLM judging."""
    term_result = deterministic_term_check(record, new_answer)
    terminology_regression = bool(term_result.get("missing_terms"))
    return {
        "pass": not terminology_regression,
        "overall_score": 4 if not terminology_regression else 2,
        "factual_regression": False,
        "safety_regression": False,
        "terminology_regression": terminology_regression,
        "missing_key_points": [],
        "new_unsupported_claims": [],
        "rationale": "LLM judge skipped; deterministic terminology fallback only.",
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    mandatory = [r for r in results if r["record"].get("mandatory")]
    judged = [r for r in results if r.get("judge")]
    failed = [r for r in judged if not r["judge"].get("pass")]
    mandatory_failed = [
        r for r in judged if r["record"].get("mandatory") and not r["judge"].get("pass")
    ]
    return {
        "total": len(results),
        "mandatory_total": len(mandatory),
        "judged_total": len(judged),
        "failed_total": len(failed),
        "mandatory_failed_total": len(mandatory_failed),
        "pass": not mandatory_failed,
        "tiers": dict(Counter(r["record"].get("tier") for r in results)),
        "failures_by_tier": dict(
            Counter(r["record"].get("tier") for r in failed)
        ),
    }


def write_markdown_report(results: List[Dict[str, Any]], summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Golden RAG Regression Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Total records: {summary['total']}",
        f"- Mandatory records: {summary['mandatory_total']}",
        f"- Judged records: {summary['judged_total']}",
        f"- Failed records: {summary['failed_total']}",
        f"- Mandatory failures: {summary['mandatory_failed_total']}",
        f"- Gate pass: {summary['pass']}",
        "",
        "## Failures",
        "",
    ]
    failures = [r for r in results if r.get("judge") and not r["judge"].get("pass")]
    if not failures:
        lines.append("No failures.")
    for result in failures:
        record = result["record"]
        judge = result["judge"]
        lines.extend(
            [
                f"### {record.get('id')} ({record.get('tier')})",
                "",
                f"Prompt: {record.get('question')}",
                "",
                f"Score: {judge.get('overall_score')}",
                f"Rationale: {judge.get('rationale')}",
                "",
                f"Missing key points: {', '.join(judge.get('missing_key_points') or []) or 'None'}",
                f"Unsupported claims: {', '.join(judge.get('new_unsupported_claims') or []) or 'None'}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_records(records: List[Dict[str, Any]], tiers: List[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    selected = [record for record in records if record.get("tier") in set(tiers)]
    return selected[:limit] if limit else selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-jsonl", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tiers", nargs="+", default=["tier_a", "tier_b", "tier_c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-judge", action="store_true", help="Skip live LLM judge and use deterministic fallback.")
    parser.add_argument("--judge-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()

    records = select_records(read_jsonl(args.golden_jsonl), args.tiers, args.limit)
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    judge_model = None
    generation_config = None
    if not args.skip_judge:
        _configure_vertex()
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        judge_model = GenerativeModel(args.judge_model)
        generation_config = GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
        )

    results: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] {record.get('tier')} {record.get('id')}")
        started = time.time()
        result: Dict[str, Any] = {"record": record, "error": None}
        try:
            current = run_current_answer(record, server_url=args.server_url, timeout=args.timeout)
            result["current"] = current
            term_check = deterministic_term_check(record, current["answer"])
            result["term_check"] = term_check
            if args.skip_judge:
                result["judge"] = fallback_judge(record, current["answer"])
            else:
                assert judge_model is not None and generation_config is not None
                result["judge"] = judge_answer(
                    record,
                    current["answer"],
                    model=judge_model,
                    generation_config=generation_config,
                )
        except Exception as exc:
            result["error"] = str(exc)
            result["judge"] = {
                "pass": False,
                "overall_score": 1,
                "factual_regression": True,
                "safety_regression": False,
                "terminology_regression": False,
                "missing_key_points": [],
                "new_unsupported_claims": [],
                "rationale": f"Eval execution failed: {exc}",
            }
        result["elapsed_seconds"] = round(time.time() - started, 3)
        results.append(result)

    summary = summarize_results(results)
    write_json({"summary": summary, "results": results}, run_dir / "results.json")
    write_markdown_report(results, summary, run_dir / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote report to {run_dir / 'report.md'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
