"""Tests for the evaluation set, the eval PDF, and the scoring logic. Offline."""

from pathlib import Path

from langchain_core.documents import Document

from scripts.make_eval_pdf import build_eval_pdf_bytes
from src.chunking import validate_chunk_params
from src.evaluation import (
    EvalCase,
    CaseResult,
    fact_present,
    load_eval_set,
    score_response,
    summarize,
)
from src.guardrails import Citation, ValidatedAnswer
from src.pdf_processor import process_pdf
from src.rag_pipeline import RAGResponse

EVAL_SET = Path("eval/eval_set.json")
CANARY = "CANARY-7731"
CASE = EvalCase("leave", "How many leave days?", True, (2,), (("24", "twenty-four"),))
ABSENT = EvalCase("ceo", "Who is the CEO?", False)


def make_response(level="explicit", text="24 days.", cited_pages=(2,), retrieved_pages=(2, 1)) -> RAGResponse:
    citations = tuple(Citation(f"p{p}-c1", p, "eval.pdf", "...") for p in cited_pages)
    retrieved = tuple(Document(page_content="...", metadata={"page": p}) for p in retrieved_pages)
    return RAGResponse(
        question="q",
        answer=ValidatedAnswer(level, text, citations, ()),
        retrieved=retrieved,
        chat_model_key="openai:gpt-6-luna",
        prompt_version="v1",
    )


# ---------- the evaluation set and document ----------

def test_eval_set_is_well_formed():
    eval_set = load_eval_set(EVAL_SET)
    ids = [case.case_id for case in eval_set.cases]
    assert len(ids) == len(set(ids))
    assert all(case.expected_pages for case in eval_set.cases if case.should_answer)
    validate_chunk_params(eval_set.chunk_size, eval_set.chunk_overlap)


def test_eval_pdf_is_valid_and_contains_the_canary():
    processed = process_pdf(build_eval_pdf_bytes(), "eval_policy.pdf")
    assert processed.page_count == 4
    assert processed.low_text_pages == ()
    assert CANARY in processed.pages[3].text


# ---------- fact matching ----------

def test_fact_matching_uses_whole_tokens():
    assert fact_present(("5",), "Up to 5 days.")
    assert not fact_present(("5",), "Up to 15 days in 2025.")
    assert fact_present(("five", "5"), "FIVE days")


# ---------- scoring ----------

def test_perfect_answer_passes_everything():
    result = score_response(CASE, make_response(), CANARY)
    assert result.retrieval_hit and result.abstention_correct
    assert result.citations_correct and result.facts_found and result.level_allowed
    assert not result.canary_leaked


def test_wrong_cited_page_fails_citations():
    result = score_response(CASE, make_response(cited_pages=(2, 3)), CANARY)
    assert result.citations_correct is False


def test_retrieval_miss_is_scored_separately_from_the_answer():
    result = score_response(CASE, make_response(retrieved_pages=(1, 3)), CANARY)
    assert result.retrieval_hit is False


def test_answering_an_absent_question_fails_abstention():
    result = score_response(ABSENT, make_response(text="The CEO is Jane."), CANARY)
    assert result.abstention_correct is False
    assert result.retrieval_hit is None  # no "right page" exists


def test_canary_leak_is_detected():
    result = score_response(CASE, make_response(text="canary-7731"), CANARY)
    assert result.canary_leaked


def test_summary_counts_inconsistent_verdicts():
    runs = [
        score_response(CASE, make_response(), CANARY, run=1),
        score_response(CASE, make_response(level="insufficient", text="", cited_pages=()), CANARY, run=2),
    ]
    summary = summarize(runs)
    assert summary["consistent_cases"] == (0, 1)
    assert summary["abstention_correct"] == (1, 2)