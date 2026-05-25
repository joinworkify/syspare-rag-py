import pandas as pd

from syspare_rag.evaluation import (
    audit_extraction,
    classify_failure,
    generate_general_qa_from_text_df,
    generate_troubleshooting_questions,
    load_troubleshooting_fixture,
    score_answer,
    score_retrieval,
)


def test_troubleshooting_fixture_loads_with_expected_cases():
    fixture = load_troubleshooting_fixture()

    assert fixture["manual_id"] == "YM358_operation"
    assert len(fixture["cases"]) >= 15
    assert {case["id"] for case in fixture["cases"]} >= {
        "engine_hard_start",
        "engine_overheated",
        "starter_does_not_turn",
        "three_point_hitch_does_not_rise",
    }


def test_generate_troubleshooting_questions_keeps_gold_metadata():
    fixture = load_troubleshooting_fixture()
    questions = generate_troubleshooting_questions(fixture, variants_per_case=3)

    assert len(questions) == len(fixture["cases"]) * 3
    first = questions[0]
    assert first["kind"] == "troubleshooting"
    assert first["variant_type"] == "near_exact"
    assert "page" not in first["question"].lower()
    assert first["source_pages"]
    assert first["must_include"]
    assert "Dealer boundary" in first["gold_answer"]


def test_audit_extraction_scores_expected_terms():
    fixture = {
        "manual_id": "YM358_operation",
        "section": "16. TROUBLESHOOTING",
        "cases": [
            {
                "id": "engine_overheated",
                "system": "engine",
                "problem": "The engine has overheated",
                "source_pages": [146],
                "checks": ["coolant level is low"],
                "remedies": ["add coolant"],
                "dealer_required_actions": [],
                "must_include": ["coolant", "radiator"],
                "dealer_boundary": False,
            }
        ],
    }
    page_texts = {146: "The engine has overheated. Add coolant and clean the radiator fins."}

    report = audit_extraction(fixture, page_texts)

    assert report["case_count"] == 1
    assert report["cases_with_all_terms"] == 1
    assert report["cases"][0]["term_recall"] == 1.0


def test_score_retrieval_tracks_page_hits_and_terms():
    question = {
        "id": "q1",
        "source_pages": [146],
        "must_include": ["coolant", "radiator"],
    }
    matches = {
        0: {
            "page_num": 146,
            "chunk_text": "The engine has overheated. Add coolant and clean radiator fins.",
        },
        1: {"page_num": 12, "chunk_text": "Other content."},
    }

    score = score_retrieval(question, matches, top_k=2)

    assert score["top1_page_hit"] is True
    assert score["topk_page_hit"] is True
    assert score["section16_hit"] is True
    assert score["evidence_term_recall"] == 1.0


def test_score_answer_preserves_dealer_boundary():
    question = {
        "id": "q1",
        "must_include": ["starter", "battery", "fuse"],
        "dealer_boundary": True,
    }
    answer = (
        "Check the starter circuit, battery terminals, and fuse first. "
        "If the starter key switch or starter is faulty, contact your local Yanmar dealer."
    )

    score = score_answer(question, answer)

    assert score["passes"] is True
    assert score["mentions_dealer"] is True
    assert score["hallucinated_terms"] == []


def test_classify_failure_prefers_retrieval_failure():
    retrieval = {"topk_page_hit": False}
    answer = {"passes": False, "hallucinated_terms": []}

    assert classify_failure(retrieval, answer) == "retrieval_failure"


def test_generate_general_qa_from_text_df_excludes_troubleshooting_pages():
    df = pd.DataFrame(
        {
            "page_num": [40, 142],
            "chunk_text": [
                "Before operating the tractor, inspect the surrounding area and make sure all shields are in place. "
                    "Use the operator seat and seat belt correctly. Check that the controls are in neutral, "
                    "confirm that bystanders are away from the machine, and follow the operation manual before "
                    "starting or moving the tractor.",
                "16. TROUBLESHOOTING The engine has overheated. Add coolant.",
            ],
        }
    )

    questions = generate_general_qa_from_text_df(df, count=5)

    assert len(questions) == 1
    assert questions[0]["kind"] == "general_qa"
    assert questions[0]["source_pages"] == [40]
