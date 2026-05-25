"""Evaluation helpers for manual-grounded RAG quality checks."""

from syspare_rag.evaluation.troubleshooting import (
    DEFAULT_TROUBLESHOOTING_FIXTURE,
    audit_extraction,
    classify_failure,
    format_gold_answer,
    generate_general_qa_from_text_df,
    generate_troubleshooting_questions,
    load_troubleshooting_fixture,
    read_jsonl,
    read_pdf_pages,
    score_answer,
    score_retrieval,
    select_general_qa_sources,
    validate_troubleshooting_fixture,
    write_jsonl,
)

__all__ = [
    "DEFAULT_TROUBLESHOOTING_FIXTURE",
    "audit_extraction",
    "classify_failure",
    "format_gold_answer",
    "generate_general_qa_from_text_df",
    "generate_troubleshooting_questions",
    "load_troubleshooting_fixture",
    "read_jsonl",
    "read_pdf_pages",
    "score_answer",
    "score_retrieval",
    "select_general_qa_sources",
    "validate_troubleshooting_fixture",
    "write_jsonl",
]
