"""Evaluation: score RAG responses against a labelled question set.

Scoring is pure (no API calls) so it can be unit-tested. Retrieval and
generation are scored SEPARATELY, so a failure points at the right stage.

Limits: fact checks are string matches (they confirm a fact appears, not
that the surrounding sentence is true), and a small set is a smoke test,
not a benchmark.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.rag_pipeline import RAGResponse

METRICS = (
    "retrieval_hit",
    "abstention_correct",
    "citations_correct",
    "facts_found",
    "level_allowed",
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    should_answer: bool
    expected_pages: tuple[int, ...] = ()
    # Each inner tuple is one required fact, given as acceptable alternatives.
    expected_facts: tuple[tuple[str, ...], ...] = ()
    allowed_levels: tuple[str, ...] = ("explicit",)


@dataclass(frozen=True)
class EvalSet:
    document: str
    canary: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    cases: tuple[EvalCase, ...]


def load_eval_set(path: str | Path) -> EvalSet:
    """Read the evaluation set from JSON (explicitly as UTF-8)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = data["settings"]
    cases = tuple(
        EvalCase(
            case_id=case["id"],
            question=case["question"],
            should_answer=case["should_answer"],
            expected_pages=tuple(case.get("expected_pages", [])),
            expected_facts=tuple(tuple(alts) for alts in case.get("expected_facts", [])),
            allowed_levels=tuple(case.get("allowed_levels", ["explicit"])),
        )
        for case in data["cases"]
    )
    return EvalSet(
        document=data["document"],
        canary=data["canary"],
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
        top_k=settings["top_k"],
        cases=cases,
    )


def fact_present(alternatives: tuple[str, ...], text: str) -> bool:
    """True if any alternative appears as a whole token (so "5" won't match "15")."""
    return any(
        re.search(rf"(?<!\w){re.escape(alt)}(?!\w)", text, flags=re.IGNORECASE)
        for alt in alternatives
    )


@dataclass(frozen=True)
class CaseResult:
    """How one run of one question scored. None = metric doesn't apply."""

    case_id: str
    run: int
    evidence_level: str
    retrieval_hit: bool | None
    abstention_correct: bool
    citations_correct: bool | None
    facts_found: bool | None
    level_allowed: bool
    canary_leaked: bool


def score_response(case: EvalCase, response: RAGResponse, canary: str, run: int = 1) -> CaseResult:
    """Score one response. Retrieval is judged independently of the answer."""
    answer = response.answer
    answered = answer.is_answered
    retrieved_pages = {doc.metadata["page"] for doc in response.retrieved}
    expected_pages = set(case.expected_pages)

    if case.should_answer:
        retrieval_hit = bool(expected_pages & retrieved_pages)
        level_allowed = answer.evidence_level in case.allowed_levels
        if answered:
            cited = set(answer.cited_pages)
            citations_correct = bool(cited) and cited <= expected_pages
            facts_found = all(fact_present(alts, answer.answer) for alts in case.expected_facts)
        else:
            citations_correct = None   # nothing cited; the abstention failure is counted
            facts_found = False
    else:
        retrieval_hit = None           # no "right page" exists for an absent answer
        level_allowed = answer.evidence_level == "insufficient"
        citations_correct = None
        facts_found = None

    return CaseResult(
        case_id=case.case_id,
        run=run,
        evidence_level=answer.evidence_level,
        retrieval_hit=retrieval_hit,
        abstention_correct=answered == case.should_answer,
        citations_correct=citations_correct,
        facts_found=facts_found,
        level_allowed=level_allowed,
        canary_leaked=canary.lower() in answer.answer.lower(),
    )


def summarize(results: list[CaseResult]) -> dict:
    """Aggregate results: (passed, applicable) per metric, leaks, consistency."""
    summary: dict = {}
    for metric in METRICS:
        applicable = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
        summary[metric] = (sum(applicable), len(applicable))

    summary["canary_leaks"] = sum(r.canary_leaked for r in results)

    levels_by_case: dict[str, set[str]] = defaultdict(set)
    for result in results:
        levels_by_case[result.case_id].add(result.evidence_level)
    consistent = sum(1 for levels in levels_by_case.values() if len(levels) == 1)
    summary["consistent_cases"] = (consistent, len(levels_by_case))
    return summary


def format_summary_markdown(summary: dict, title: str) -> str:
    """A Markdown table you can paste straight into the README."""
    labels = {
        "retrieval_hit": "Retrieval hit (right page retrieved)",
        "abstention_correct": "Answered / abstained correctly",
        "citations_correct": "Citations correct",
        "facts_found": "Expected facts in answer",
        "level_allowed": "Evidence level correct",
        "consistent_cases": "Same verdict on every run",
    }
    lines = [f"### {title}", "", "| Metric | Result |", "| --- | --- |"]
    for key, label in labels.items():
        passed, total = summary[key]
        percent = f"{100 * passed / total:.0f}%" if total else "n/a"
        lines.append(f"| {label} | {passed}/{total} ({percent}) |")
    lines.append(f"| Prompt-injection canary leaks | {summary['canary_leaks']} |")
    return "\n".join(lines)