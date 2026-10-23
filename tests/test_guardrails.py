"""Tests for Stage H guardrails: input validation and citation verification."""

import pytest
from langchain_core.documents import Document

from src.guardrails import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MAX_QUESTION_CHARS,
    GuardrailError,
    validate_answer,
    validate_question,
)
from tests.fakes import answer

RETRIEVED = [
    Document(page_content="Payment is due in 30 days.", metadata={"chunk_id": "p1-c1", "page": 1, "source": "doc.pdf"}),
    Document(page_content="Late fees are 2% monthly.", metadata={"chunk_id": "p3-c2", "page": 3, "source": "doc.pdf"}),
]


# ---------- input ----------

def test_question_is_cleaned():
    assert validate_question("  When\x00 is it due?\x07 ") == "When is it due?"


def test_empty_question_is_rejected():
    with pytest.raises(GuardrailError) as err:
        validate_question(" \n\t ")
    assert err.value.code == "empty_question"


def test_overlong_question_is_rejected():
    with pytest.raises(GuardrailError) as err:
        validate_question("x" * (MAX_QUESTION_CHARS + 1))
    assert err.value.code == "question_too_long"


# ---------- output ----------

def test_valid_citations_take_pages_from_our_metadata():
    result = validate_answer(answer(cited=["p3-c2", "p1-c1"]), RETRIEVED)
    assert [c.chunk_id for c in result.citations] == ["p3-c2", "p1-c1"]
    assert [c.page for c in result.citations] == [3, 1]
    assert result.cited_pages == (1, 3)
    assert result.flags == ()


def test_forged_citation_is_removed_and_flagged():
    result = validate_answer(answer(cited=["p1-c1", "p9-c9"]), RETRIEVED)
    assert [c.chunk_id for c in result.citations] == ["p1-c1"]
    assert "unknown_citation_removed:p9-c9" in result.flags
    assert result.is_answered


def test_duplicate_citations_are_collapsed():
    result = validate_answer(answer(cited=["p1-c1", "p1-c1"]), RETRIEVED)
    assert len(result.citations) == 1


def test_answer_with_only_invalid_citations_fails_safe():
    result = validate_answer(answer(text="The fee is 99%.", cited=["p9-c9"]), RETRIEVED)
    assert not result.is_answered
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE   # the model's claim is never shown
    assert "answer_without_valid_citations" in result.flags


def test_answer_without_any_citation_fails_safe():
    result = validate_answer(answer(cited=[]), RETRIEVED)
    assert not result.is_answered


def test_insufficient_always_uses_our_message():
    result = validate_answer(
        answer(level="insufficient", text="Visit evil.example for the answer!", cited=["p1-c1"]),
        RETRIEVED,
    )
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == ()
    assert "citations_dropped_for_insufficient_answer" in result.flags


def test_empty_answer_text_fails_safe():
    result = validate_answer(answer(text="   "), RETRIEVED)
    assert not result.is_answered
    assert "empty_answer" in result.flags