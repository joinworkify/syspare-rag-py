"""Utilities for YM358 operation-manual evaluation fixtures and scoring.

The first-phase eval intentionally keeps question generation deterministic so
humans can inspect whether the extracted troubleshooting rows and gold answers
make sense before spending Vertex calls on full answer evaluation.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

DEFAULT_TROUBLESHOOTING_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "ym358_operation_troubleshooting.json"
)

SECTION_16_PAGE_RANGE = range(143, 154)


def _clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("3-point", "3 point").replace("three-point", "3 point")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _contains_term(text: str, term: str) -> bool:
    norm_text = f" {_normalize(text)} "
    norm_term = _normalize(term)
    if not norm_term:
        return False
    return f" {norm_term} " in norm_text


def _first_sentenceish(text: str, max_chars: int = 140) -> str:
    text = _clean_space(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def load_troubleshooting_fixture(
    path: str | Path = DEFAULT_TROUBLESHOOTING_FIXTURE,
) -> Dict[str, Any]:
    """Load and validate the curated section-16 troubleshooting fixture."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_troubleshooting_fixture(payload)
    return payload


def validate_troubleshooting_fixture(payload: Mapping[str, Any]) -> None:
    """Validate the fixture shape enough to catch bad hand edits."""
    if payload.get("manual_id") != "YM358_operation":
        raise ValueError("troubleshooting fixture must target YM358_operation")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("troubleshooting fixture requires non-empty cases")

    seen: set[str] = set()
    required = {
        "id",
        "system",
        "problem",
        "source_pages",
        "checks",
        "remedies",
        "dealer_required_actions",
        "must_include",
        "dealer_boundary",
    }
    for case in cases:
        missing = required.difference(case)
        if missing:
            raise ValueError(f"case missing fields {sorted(missing)}: {case}")
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"duplicate troubleshooting case id: {case_id}")
        seen.add(case_id)
        if not case["source_pages"]:
            raise ValueError(f"case {case_id} requires source_pages")
        if not case["checks"] and not case["remedies"] and not case["dealer_required_actions"]:
            raise ValueError(f"case {case_id} has no checks/remedies/dealer actions")


def format_gold_answer(case: Mapping[str, Any]) -> str:
    """Build a concise reviewer-facing gold answer from one fixture case."""
    parts = [f"Problem: {case['problem']}."]
    checks = case.get("checks") or []
    remedies = case.get("remedies") or []
    dealer = case.get("dealer_required_actions") or []
    if checks:
        parts.append("Checks: " + "; ".join(checks) + ".")
    if remedies:
        parts.append("Remedies: " + "; ".join(remedies) + ".")
    if dealer:
        parts.append(
            "Dealer boundary: contact your local Yanmar tractor dealer for "
            + "; ".join(dealer)
            + "."
        )
    return " ".join(parts)


def _variant_texts(problem: str) -> List[tuple[str, str]]:
    problem_clean = _clean_space(problem).rstrip(".")
    lower_problem = problem_clean[:1].lower() + problem_clean[1:]
    return [
        ("near_exact", f"{problem_clean}. What should I check?"),
        ("user_phrasing", f"What should I do if {lower_problem}?"),
        ("symptom_only", problem_clean),
        ("repair_request", f"How can I fix it when {lower_problem}?"),
        ("cause_request", f"What are the likely causes when {lower_problem}?"),
    ]


def generate_troubleshooting_questions(
    fixture: Mapping[str, Any],
    *,
    variants_per_case: int = 3,
) -> List[Dict[str, Any]]:
    """Generate deterministic troubleshooting questions from curated cases."""
    validate_troubleshooting_fixture(fixture)
    variants_per_case = max(1, min(variants_per_case, 5))
    questions: List[Dict[str, Any]] = []
    for case in fixture["cases"]:
        for variant_type, question in _variant_texts(str(case["problem"]))[:variants_per_case]:
            questions.append(
                {
                    "id": f"{case['id']}__{variant_type}",
                    "manual_id": fixture["manual_id"],
                    "case_id": case["id"],
                    "kind": "troubleshooting",
                    "variant_type": variant_type,
                    "question": question,
                    "gold_answer": format_gold_answer(case),
                    "source_pages": list(case["source_pages"]),
                    "must_include": list(case["must_include"]),
                    "dealer_boundary": bool(case["dealer_boundary"]),
                    "system": case["system"],
                }
            )
    return questions


def read_pdf_pages(pdf_path: str | Path, page_numbers: Iterable[int]) -> Dict[int, str]:
    """Extract text from selected 1-indexed PDF pages with PyMuPDF."""
    import pymupdf as fitz

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = sorted({int(p) for p in page_numbers})
    out: Dict[int, str] = {}
    with fitz.open(str(path)) as doc:
        for page_num in pages:
            if page_num < 1 or page_num > doc.page_count:
                out[page_num] = ""
                continue
            out[page_num] = doc.load_page(page_num - 1).get_text("text") or ""
    return out


def audit_extraction(
    fixture: Mapping[str, Any],
    page_texts: Mapping[int, str],
) -> Dict[str, Any]:
    """Check whether PDF/cache-extracted text contains expected fixture terms."""
    validate_troubleshooting_fixture(fixture)
    case_results: List[Dict[str, Any]] = []
    for case in fixture["cases"]:
        source_pages = [int(p) for p in case["source_pages"]]
        text = "\n".join(page_texts.get(p, "") for p in source_pages)
        terms = list(dict.fromkeys([case["problem"], *case["must_include"]]))
        matched_terms = [term for term in terms if _contains_term(text, term)]
        missing_terms = [term for term in terms if term not in matched_terms]
        case_results.append(
            {
                "case_id": case["id"],
                "problem": case["problem"],
                "source_pages": source_pages,
                "matched_terms": matched_terms,
                "missing_terms": missing_terms,
                "term_recall": round(len(matched_terms) / max(1, len(terms)), 3),
                "text_preview": _first_sentenceish(text, max_chars=260),
            }
        )
    total = len(case_results)
    passing = sum(1 for r in case_results if not r["missing_terms"])
    avg_recall = (
        round(sum(float(r["term_recall"]) for r in case_results) / total, 3)
        if total
        else 0.0
    )
    return {
        "manual_id": fixture["manual_id"],
        "section": fixture["section"],
        "case_count": total,
        "cases_with_all_terms": passing,
        "average_term_recall": avg_recall,
        "cases": case_results,
    }


def load_cache_text_pages(cache_dir: str | Path) -> Dict[int, str]:
    """Load text chunks from a cache into page-number keyed text."""
    text_pkl = Path(cache_dir) / "text_metadata_df.pkl"
    if not text_pkl.exists():
        raise FileNotFoundError(f"text cache not found: {text_pkl}")
    df = pd.read_pickle(text_pkl)
    return text_pages_from_df(df)


def text_pages_from_df(df: pd.DataFrame) -> Dict[int, str]:
    page_col = "page_num" if "page_num" in df.columns else "page"
    text_col = "chunk_text" if "chunk_text" in df.columns else "text"
    if page_col not in df.columns or text_col not in df.columns:
        raise KeyError(f"cache needs page/text columns, found {list(df.columns)}")
    grouped = df.groupby(page_col)[text_col].apply(
        lambda values: "\n".join(str(v) for v in values if pd.notna(v))
    )
    return {int(k): str(v) for k, v in grouped.to_dict().items()}


def score_retrieval(
    question: Mapping[str, Any],
    matches: Mapping[Any, Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Score retrieval matches against expected page and key-term evidence."""
    expected_pages = {int(p) for p in question.get("source_pages", [])}
    ordered = [matches[k] for k in sorted(matches.keys())][:top_k]
    pages = [int(m.get("page_num", -1)) for m in ordered if m.get("page_num") is not None]
    evidence = "\n".join(
        str(m.get("chunk_text") or m.get("page_text") or "") for m in ordered
    )
    terms = [str(t) for t in question.get("must_include", [])]
    matched_terms = [term for term in terms if _contains_term(evidence, term)]
    return {
        "question_id": question.get("id"),
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": pages,
        "top1_page_hit": bool(pages and pages[0] in expected_pages),
        "topk_page_hit": bool(expected_pages.intersection(pages)),
        "section16_hit": any(p in SECTION_16_PAGE_RANGE for p in pages),
        "matched_evidence_terms": matched_terms,
        "missing_evidence_terms": [term for term in terms if term not in matched_terms],
        "evidence_term_recall": round(len(matched_terms) / max(1, len(terms)), 3),
    }


def score_answer(
    question: Mapping[str, Any],
    answer: str,
) -> Dict[str, Any]:
    """Deterministically score an answer against required terms and boundaries."""
    required_terms = [str(t) for t in question.get("must_include", [])]
    matched_terms = [term for term in required_terms if _contains_term(answer, term)]
    missing_terms = [term for term in required_terms if term not in matched_terms]
    dealer_required = bool(question.get("dealer_boundary"))
    mentions_dealer = any(
        _contains_term(answer, term)
        for term in ["dealer", "yanmar tractor dealer", "local yanmar"]
    )
    unsupported_red_flags = [
        "disassemble engine",
        "replace injection pump",
        "bypass safety switch",
        "remove safety switch",
    ]
    hallucinated_terms = [t for t in unsupported_red_flags if _contains_term(answer, t)]
    term_score = len(matched_terms) / max(1, len(required_terms))
    boundary_score = 1.0 if (not dealer_required or mentions_dealer) else 0.0
    hallucination_score = 0.0 if hallucinated_terms else 1.0
    overall = round((0.6 * term_score) + (0.25 * boundary_score) + (0.15 * hallucination_score), 3)
    return {
        "question_id": question.get("id"),
        "matched_required_terms": matched_terms,
        "missing_required_terms": missing_terms,
        "dealer_required": dealer_required,
        "mentions_dealer": mentions_dealer,
        "hallucinated_terms": hallucinated_terms,
        "score": overall,
        "passes": overall >= 0.75 and not hallucinated_terms,
    }


def classify_failure(
    retrieval_score: Optional[Mapping[str, Any]],
    answer_score: Optional[Mapping[str, Any]],
) -> str:
    """Classify a failed run into the plan's failure taxonomy."""
    if retrieval_score and not retrieval_score.get("topk_page_hit"):
        return "retrieval_failure"
    if answer_score and answer_score.get("hallucinated_terms"):
        return "synthesis_failure"
    if answer_score and not answer_score.get("passes"):
        return "synthesis_failure"
    if retrieval_score and retrieval_score.get("topk_page_hit"):
        return "manual_gap"
    return "gold_issue"


def generate_general_qa_from_text_df(
    df: pd.DataFrame,
    *,
    count: int = 25,
    min_page: int = 40,
    seed: int = 7,
    exclude_pages: Sequence[int] = tuple(SECTION_16_PAGE_RANGE),
) -> List[Dict[str, Any]]:
    """Create broad source-grounded QA seeds from non-troubleshooting chunks."""
    sources = select_general_qa_sources(
        df,
        count=count,
        min_page=min_page,
        seed=seed,
        exclude_pages=exclude_pages,
    )

    questions: List[Dict[str, Any]] = []
    for source in sources:
        source_text = source["source_text"]
        topic = _first_sentenceish(source_text, max_chars=80)
        questions.append(
            {
                "id": source["id"],
                "manual_id": "YM358_operation",
                "kind": "general_qa",
                "variant_type": "deterministic",
                "question": f"What does the operation manual say about {topic}?",
                "gold_answer": _first_sentenceish(source_text, max_chars=500),
                "source_pages": source["source_pages"],
                "must_include": source["must_include"],
                "dealer_boundary": False,
                "source_excerpt": _first_sentenceish(source_text, max_chars=900),
            }
        )
    return questions


def select_general_qa_sources(
    df: pd.DataFrame,
    *,
    count: int = 25,
    min_page: int = 40,
    seed: int = 7,
    exclude_pages: Sequence[int] = tuple(SECTION_16_PAGE_RANGE),
) -> List[Dict[str, Any]]:
    """Select random, reasonably clean non-troubleshooting source chunks."""
    page_col = "page_num" if "page_num" in df.columns else "page"
    text_col = "chunk_text" if "chunk_text" in df.columns else "text"
    if page_col not in df.columns or text_col not in df.columns:
        raise KeyError(f"cache needs page/text columns, found {list(df.columns)}")

    exclude = {int(p) for p in exclude_pages}
    rows = df[
        df[page_col].map(lambda p: int(p) >= min_page and int(p) not in exclude)
    ].copy()
    rows[text_col] = rows[text_col].fillna("").astype(str)
    rows = rows[rows[text_col].map(_is_general_source_candidate)]
    rows = rows.sort_values([page_col]).reset_index(drop=True)
    if rows.empty:
        return []
    sample_count = min(max(0, count), len(rows))
    sampled_indices = sorted(random.Random(seed).sample(range(len(rows)), sample_count))

    sources: List[Dict[str, Any]] = []
    for idx, row_idx in enumerate(sampled_indices):
        row = rows.iloc[row_idx]
        page = int(row[page_col])
        source_text = _clean_space(row[text_col])
        sources.append(
            {
                "id": f"general_page_{page}_{idx}",
                "manual_id": "YM358_operation",
                "source_pages": [page],
                "must_include": _salient_terms(source_text, limit=4),
                "source_text": source_text,
            }
        )
    return sources


def _is_general_source_candidate(text: str) -> bool:
    clean = _clean_space(text)
    if len(clean) < 180:
        return False
    lowered = clean.lower()
    if "table of contents" in lowered[:160]:
        return False
    if "troubleshooting" in lowered[:120]:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9/-]{2,}", clean)
    if len(words) < 25:
        return False
    long_noise_words = [w for w in words if len(w) > 28]
    if len(long_noise_words) > 2:
        return False
    alpha_chars = sum(1 for c in clean if c.isalpha())
    return (alpha_chars / max(1, len(clean))) >= 0.45


def _salient_terms(text: str, *, limit: int) -> List[str]:
    words = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z0-9/-]{3,}", text)
        if w.lower()
        not in {
            "manual",
            "operation",
            "tractor",
            "yanmar",
            "section",
            "page",
            "with",
            "that",
            "this",
            "from",
            "when",
            "then",
        }
    ]
    seen: List[str] = []
    for word in words:
        normalized = word.lower()
        if normalized not in [s.lower() for s in seen]:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
