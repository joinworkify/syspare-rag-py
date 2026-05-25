"""First-phase eval tools for the YM358 operation manual.

Examples:
    uv run python scripts/eval_ym358_operation.py audit-extraction
    uv run python scripts/eval_ym358_operation.py generate-questions
    uv run python scripts/eval_ym358_operation.py run-retrieval
    uv run python scripts/eval_ym358_operation.py run-answers --limit 10
    uv run python scripts/eval_ym358_operation.py generate-general-qa

The script uses the existing Vertex AI setup for retrieval/answer calls. The
fixture audit and deterministic question generation do not require live LLM
calls, which makes them cheap to inspect before running full evals.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
from dotenv import load_dotenv
from vertexai.generative_models import GenerationConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

load_dotenv(_REPO_ROOT / ".env")

from pipeline import MultimodalRAGPipeline  # noqa: E402
from syspare_rag.config import load_manual_registry_from_env, load_rag_config_from_env  # noqa: E402
from syspare_rag.evaluation import (  # noqa: E402
    DEFAULT_TROUBLESHOOTING_FIXTURE,
    audit_extraction,
    classify_failure,
    generate_general_qa_from_text_df,
    generate_troubleshooting_questions,
    load_troubleshooting_fixture,
    read_jsonl,
    read_pdf_pages,
    score_answer,
    score_retrieval,
    select_general_qa_sources,
    write_jsonl,
)
from syspare_rag.evaluation.troubleshooting import load_cache_text_pages  # noqa: E402
from utils import get_gemini_response  # noqa: E402

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "eval" / "ym358_operation"
DEFAULT_QUESTIONS = DEFAULT_OUTPUT_DIR / "troubleshooting_questions.jsonl"
DEFAULT_GENERAL_QA = DEFAULT_OUTPUT_DIR / "general_qa_questions.jsonl"
DEFAULT_RETRIEVAL = DEFAULT_OUTPUT_DIR / "retrieval_results.json"
DEFAULT_ANSWERS = DEFAULT_OUTPUT_DIR / "answer_results.json"


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    """Serialize pandas/numpy/cache values emitted by retrieval matches."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return str(value)


def _load_fixture(path: str | Path) -> Dict[str, Any]:
    return load_troubleshooting_fixture(path)


def _manual_paths(manual_id: str = "YM358_operation") -> Dict[str, str]:
    registry = load_manual_registry_from_env()
    manual = registry.get(manual_id)
    return {
        "manual_id": manual.manual_id,
        "pdf_folder": manual.pdf_folder,
        "cache_dir": manual.cache_dir,
        "image_dir": manual.image_dir,
        "ocr_lang": manual.ocr_lang,
    }


def _default_pdf_path() -> Path:
    paths = _manual_paths()
    return _REPO_ROOT / paths["pdf_folder"] / "ym358a.pdf"


def _default_cache_dir() -> Path:
    paths = _manual_paths()
    return _REPO_ROOT / paths["cache_dir"]


def _build_rag(cache_dir: str | Path) -> MultimodalRAGPipeline:
    paths = _manual_paths()
    cfg = load_rag_config_from_env()
    cfg.paths.pdf_folder = paths["pdf_folder"]
    cfg.paths.cache_dir = str(cache_dir)
    cfg.paths.image_dir = paths["image_dir"]
    cfg.image_save_dir = paths["image_dir"]
    cfg.ocr_lang = paths["ocr_lang"]

    rag = MultimodalRAGPipeline(cfg)
    if not rag.load_cache(str(cache_dir), rebuild_image_objects=False):
        raise FileNotFoundError(
            f"Cache not found at {cache_dir}. Sync/build YM358_operation cache before retrieval eval."
        )
    return rag


def _summarize_retrieval(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}
    return {
        "count": len(results),
        "top1_page_hit_rate": round(
            sum(1 for r in results if r["retrieval_score"]["top1_page_hit"]) / len(results),
            3,
        ),
        "topk_page_hit_rate": round(
            sum(1 for r in results if r["retrieval_score"]["topk_page_hit"]) / len(results),
            3,
        ),
        "section16_hit_rate": round(
            sum(1 for r in results if r["retrieval_score"]["section16_hit"]) / len(results),
            3,
        ),
        "avg_evidence_term_recall": round(
            sum(float(r["retrieval_score"]["evidence_term_recall"]) for r in results)
            / len(results),
            3,
        ),
    }


def _summarize_answers(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}
    return {
        "count": len(results),
        "pass_rate": round(sum(1 for r in results if r["answer_score"]["passes"]) / len(results), 3),
        "avg_answer_score": round(
            sum(float(r["answer_score"]["score"]) for r in results) / len(results),
            3,
        ),
        "failure_counts": _counts(r["failure_type"] for r in results if r["failure_type"]),
    }


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output with light markdown cleanup."""
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
        raise ValueError("model output is not a JSON object")
    return parsed


def _generate_vertex_general_qa(
    model: Any,
    source: Mapping[str, Any],
    *,
    temperature: float,
) -> Dict[str, Any]:
    source_text = str(source["source_text"])
    prompt = (
        "You are creating one evaluation question-answer pair for a tractor RAG system.\n"
        "Use ONLY the source excerpt. Do not mention page numbers or the manual excerpt.\n"
        "Generate a natural question a farmer or mechanic might ask, and a concise gold answer "
        "that is fully supported by the excerpt.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        '{ "question": string, "gold_answer": string }\n\n'
        f"Source excerpt:\n\"\"\"\n{source_text[:1800]}\n\"\"\""
    )
    raw = get_gemini_response(
        model,
        model_input=prompt,
        stream=False,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
    )
    parsed = _parse_json_object(str(raw))
    question = str(parsed.get("question", "")).strip()
    gold_answer = str(parsed.get("gold_answer", "")).strip()
    if not question or not gold_answer:
        raise ValueError(f"incomplete model QA output: {parsed}")
    return {
        "id": f"{source['id']}__vertex",
        "manual_id": "YM358_operation",
        "kind": "general_qa",
        "variant_type": "vertex_reverse",
        "question": question,
        "gold_answer": gold_answer,
        "source_pages": source["source_pages"],
        "must_include": _required_terms_from_answer(gold_answer) or source["must_include"],
        "dealer_boundary": False,
        "source_excerpt": source_text[:900],
    }


def _required_terms_from_answer(answer: str, *, limit: int = 4) -> List[str]:
    stopwords = {
        "about",
        "after",
        "before",
        "check",
        "from",
        "should",
        "that",
        "the",
        "then",
        "this",
        "tractor",
        "when",
        "with",
        "your",
    }
    terms: List[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9/-]{3,}", answer):
        normalized = word.lower()
        if normalized in stopwords:
            continue
        if normalized not in [t.lower() for t in terms]:
            terms.append(word)
        if len(terms) >= limit:
            break
    return terms


def cmd_audit_extraction(args: argparse.Namespace) -> int:
    fixture = _load_fixture(args.fixture)
    page_numbers = sorted({p for case in fixture["cases"] for p in case["source_pages"]})
    if args.source == "cache":
        page_texts = load_cache_text_pages(args.cache_dir)
    else:
        page_texts = read_pdf_pages(args.pdf_path, page_numbers)

    report = audit_extraction(fixture, page_texts)
    report["source"] = args.source
    report["pdf_path"] = str(args.pdf_path) if args.source == "pdf" else None
    report["cache_dir"] = str(args.cache_dir) if args.source == "cache" else None
    _write_json(report, args.output)
    print(
        "Extraction audit:",
        f"cases={report['case_count']}",
        f"all_terms={report['cases_with_all_terms']}",
        f"avg_recall={report['average_term_recall']}",
        f"output={args.output}",
    )
    return 0


def cmd_generate_questions(args: argparse.Namespace) -> int:
    fixture = _load_fixture(args.fixture)
    questions = generate_troubleshooting_questions(
        fixture,
        variants_per_case=args.variants_per_case,
    )
    write_jsonl(questions, args.output)
    print(f"Generated {len(questions)} troubleshooting questions -> {args.output}")
    return 0


def cmd_run_retrieval(args: argparse.Namespace) -> int:
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    rag = _build_rag(args.cache_dir)
    results: List[Dict[str, Any]] = []
    for question in questions:
        matches = rag.search_text(question["question"], top_n=args.top_n, chunk_text=True)
        retrieval_score = score_retrieval(question, matches, top_k=args.top_n)
        results.append(
            {
                "question": question,
                "retrieval_score": retrieval_score,
                "matches": matches,
            }
        )
    payload = {"summary": _summarize_retrieval(results), "results": results}
    _write_json(payload, args.output)
    print(f"Retrieval eval: {payload['summary']} -> {args.output}")
    return 0


def cmd_run_answers(args: argparse.Namespace) -> int:
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    rag = _build_rag(args.cache_dir)
    results: List[Dict[str, Any]] = []
    for question in questions:
        answer = rag.answer_text_query(
            question["question"],
            top_n=args.top_n,
            temperature=args.temperature,
            stream=False,
            answer_language=args.answer_language,
        )
        retrieval_score = score_retrieval(question, answer["matches"], top_k=args.top_n)
        answer_score = score_answer(question, str(answer["response"]))
        failure_type = None
        if not retrieval_score["topk_page_hit"] or not answer_score["passes"]:
            failure_type = classify_failure(retrieval_score, answer_score)
        results.append(
            {
                "question": question,
                "retrieval_score": retrieval_score,
                "answer_score": answer_score,
                "failure_type": failure_type,
                "answer": answer["response"],
                "matches": answer["matches"],
            }
        )
    payload = {"summary": _summarize_answers(results), "results": results}
    _write_json(payload, args.output)
    print(f"Answer eval: {payload['summary']} -> {args.output}")
    return 0


def cmd_generate_general_qa(args: argparse.Namespace) -> int:
    text_pkl = Path(args.cache_dir) / "text_metadata_df.pkl"
    if not text_pkl.exists():
        raise FileNotFoundError(
            f"Cache not found at {text_pkl}. Sync/build YM358_operation cache first."
        )
    df: pd.DataFrame = pd.read_pickle(text_pkl)
    if args.mode == "vertex":
        sources = select_general_qa_sources(
            df,
            count=args.count,
            min_page=args.min_page,
            seed=args.seed,
        )
        rag = _build_rag(args.cache_dir)
        questions = [
            _generate_vertex_general_qa(
                rag.text_model,
                source,
                temperature=args.temperature,
            )
            for source in sources
        ]
    else:
        questions = generate_general_qa_from_text_df(
            df,
            count=args.count,
            min_page=args.min_page,
            seed=args.seed,
        )
    write_jsonl(questions, args.output)
    print(f"Generated {len(questions)} general QA seeds ({args.mode}) -> {args.output}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    audit_args = argparse.Namespace(
        fixture=args.fixture,
        pdf_path=args.pdf_path,
        cache_dir=args.cache_dir,
        source="pdf",
        output=DEFAULT_OUTPUT_DIR / "extraction_audit_pdf.json",
    )
    cmd_audit_extraction(audit_args)

    question_args = argparse.Namespace(
        fixture=args.fixture,
        variants_per_case=args.variants_per_case,
        output=args.questions,
    )
    cmd_generate_questions(question_args)

    cache_dir = Path(args.cache_dir)
    if (cache_dir / "text_metadata_df.pkl").exists() and (cache_dir / "image_metadata_df.pkl").exists():
        cmd_generate_general_qa(
            argparse.Namespace(
                cache_dir=cache_dir,
                count=args.general_count,
                min_page=args.min_page,
                seed=args.seed,
                mode=args.general_mode,
                temperature=args.temperature,
                output=args.general_output,
            )
        )
        if args.include_retrieval:
            cmd_run_retrieval(
                argparse.Namespace(
                    questions=args.questions,
                    cache_dir=cache_dir,
                    output=args.retrieval_output,
                    top_n=args.top_n,
                    limit=args.limit,
                )
            )
        if args.include_answers:
            cmd_run_answers(
                argparse.Namespace(
                    questions=args.questions,
                    cache_dir=cache_dir,
                    output=args.answer_output,
                    top_n=args.top_n,
                    limit=args.limit,
                    temperature=args.temperature,
                    answer_language=args.answer_language,
                )
            )
    else:
        print(f"Cache not found at {cache_dir}; skipped general QA, retrieval, and answer eval.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    common_fixture = argparse.ArgumentParser(add_help=False)
    common_fixture.add_argument("--fixture", default=DEFAULT_TROUBLESHOOTING_FIXTURE)

    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit-extraction", parents=[common_fixture])
    audit.add_argument("--source", choices=["pdf", "cache"], default="pdf")
    audit.add_argument("--pdf-path", default=_default_pdf_path())
    audit.add_argument("--cache-dir", default=_default_cache_dir())
    audit.add_argument("--output", default=DEFAULT_OUTPUT_DIR / "extraction_audit_pdf.json")
    audit.set_defaults(func=cmd_audit_extraction)

    gen = sub.add_parser("generate-questions", parents=[common_fixture])
    gen.add_argument("--variants-per-case", type=int, default=3)
    gen.add_argument("--output", default=DEFAULT_QUESTIONS)
    gen.set_defaults(func=cmd_generate_questions)

    retrieval = sub.add_parser("run-retrieval")
    retrieval.add_argument("--questions", default=DEFAULT_QUESTIONS)
    retrieval.add_argument("--cache-dir", default=_default_cache_dir())
    retrieval.add_argument("--output", default=DEFAULT_RETRIEVAL)
    retrieval.add_argument("--top-n", type=int, default=5)
    retrieval.add_argument("--limit", type=int, default=0)
    retrieval.set_defaults(func=cmd_run_retrieval)

    answers = sub.add_parser("run-answers")
    answers.add_argument("--questions", default=DEFAULT_QUESTIONS)
    answers.add_argument("--cache-dir", default=_default_cache_dir())
    answers.add_argument("--output", default=DEFAULT_ANSWERS)
    answers.add_argument("--top-n", type=int, default=5)
    answers.add_argument("--limit", type=int, default=0)
    answers.add_argument("--temperature", type=float, default=0.1)
    answers.add_argument("--answer-language", default="en")
    answers.set_defaults(func=cmd_run_answers)

    general = sub.add_parser("generate-general-qa")
    general.add_argument("--cache-dir", default=_default_cache_dir())
    general.add_argument("--count", type=int, default=25)
    general.add_argument("--min-page", type=int, default=40)
    general.add_argument("--seed", type=int, default=7)
    general.add_argument("--mode", choices=["deterministic", "vertex"], default="vertex")
    general.add_argument("--temperature", type=float, default=0.2)
    general.add_argument("--output", default=DEFAULT_GENERAL_QA)
    general.set_defaults(func=cmd_generate_general_qa)

    all_cmd = sub.add_parser("all", parents=[common_fixture])
    all_cmd.add_argument("--pdf-path", default=_default_pdf_path())
    all_cmd.add_argument("--cache-dir", default=_default_cache_dir())
    all_cmd.add_argument("--variants-per-case", type=int, default=3)
    all_cmd.add_argument("--questions", default=DEFAULT_QUESTIONS)
    all_cmd.add_argument("--general-output", default=DEFAULT_GENERAL_QA)
    all_cmd.add_argument("--retrieval-output", default=DEFAULT_RETRIEVAL)
    all_cmd.add_argument("--answer-output", default=DEFAULT_ANSWERS)
    all_cmd.add_argument("--general-count", type=int, default=25)
    all_cmd.add_argument("--min-page", type=int, default=40)
    all_cmd.add_argument("--seed", type=int, default=7)
    all_cmd.add_argument("--general-mode", choices=["deterministic", "vertex"], default="vertex")
    all_cmd.add_argument("--top-n", type=int, default=5)
    all_cmd.add_argument("--limit", type=int, default=0)
    all_cmd.add_argument("--temperature", type=float, default=0.1)
    all_cmd.add_argument("--answer-language", default="en")
    all_cmd.add_argument("--include-retrieval", action="store_true")
    all_cmd.add_argument("--include-answers", action="store_true")
    all_cmd.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
