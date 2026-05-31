from pipeline import (
    INSUFFICIENT_CONTEXT_SENTINEL,
    has_insufficient_marker,
    strip_insufficient_marker,
)


def test_has_insufficient_marker_detects_sentinel():
    assert has_insufficient_marker(f"{INSUFFICIENT_CONTEXT_SENTINEL}\nContact your dealer.")
    assert has_insufficient_marker("insufficient_context: not enough info")


def test_has_insufficient_marker_false_for_normal_answer():
    assert not has_insufficient_marker("Check the oil level and tighten the bolt.")
    assert not has_insufficient_marker("")
    assert not has_insufficient_marker(None)


def test_strip_insufficient_marker_removes_token_and_punctuation():
    raw = f"{INSUFFICIENT_CONTEXT_SENTINEL}\nPlease contact your local Yanmar dealer."
    cleaned = strip_insufficient_marker(raw)
    assert INSUFFICIENT_CONTEXT_SENTINEL not in cleaned
    assert cleaned.startswith("Please contact")


def test_strip_insufficient_marker_keeps_normal_text():
    text = "Tighten to 25 Nm. [Image 1]"
    assert strip_insufficient_marker(text) == text
