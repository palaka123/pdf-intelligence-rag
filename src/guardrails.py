"""Stage H: guardrails around the LLM — input checks before, output checks after.

What these guardrails CAN do: reject malformed input, guarantee every
citation points at a chunk that was really retrieved, take page numbers
from our own metadata (never from the model), and fail safely.

What they CANNOT do: prove that each sentence of an answer is supported by
the excerpt it cites. A model can cite a real chunk and still misstate it.
That needs human review or a separate evaluation step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.models import GroundedAnswer

MAX_QUESTION_CHARS = 1000

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough support in the document to answer that reliably. "
    "Try rephrasing, or ask about something the document covers."
)

# Invisible control characters (keeps normal newlines and tabs).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class GuardrailError(ValueError):
    """Input rejected by a guardrail, with a machine-readable `code`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- input

def validate_question(question: str | None) -> str:
    """Clean and check a user question BEFORE any retrieval or LLM call."""
    cleaned = _CONTROL_CHARS.sub("", question or "").strip()
    if not cleaned:
        raise GuardrailError("empty_question", "Please enter a question.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise GuardrailError(
            "question_too_long",
            f"Please keep questions under {MAX_QUESTION_CHARS} characters "
            f"(this one has {len(cleaned)}).",
        )
    return cleaned


# ---------------------------------------------------------------- output

@dataclass(frozen=True)
class Citation:
    """A verified citation. Every field comes from OUR metadata, not the model."""

    chunk_id: str
    page: int
    source: str
    excerpt: str


@dataclass(frozen=True)
class ValidatedAnswer:
    """The answer the user sees, after guardrails."""

    evidence_level: str              # "explicit" | "inferred" | "insufficient"
    answer: str
    citations: tuple[Citation, ...]
    flags: tuple[str, ...]           # what the guardrails changed, for inspection

    @property
    def is_answered(self) -> bool:
        return self.evidence_level != "insufficient"

    @property
    def cited_pages(self) -> tuple[int, ...]:
        """Unique cited pages in ascending order, for display."""
        return tuple(sorted({c.page for c in self.citations}))


def _short(text: str, limit: int = 20) -> str:
    """Model-produced text in flags is untrusted: keep it short."""
    return text if len(text) <= limit else text[:limit] + "…"


def _fallback(flags: list[str]) -> ValidatedAnswer:
    return ValidatedAnswer(
        evidence_level="insufficient",
        answer=INSUFFICIENT_EVIDENCE_MESSAGE,
        citations=(),
        flags=tuple(flags),
    )


def validate_answer(raw: GroundedAnswer, retrieved: list[Document]) -> ValidatedAnswer:
    """Check the model's answer against what was ACTUALLY retrieved."""
    by_id = {doc.metadata["chunk_id"]: doc for doc in retrieved}
    flags: list[str] = []

    # 1. Keep only citations that point at real retrieved chunks, once each.
    valid_ids: list[str] = []
    for cited in raw.cited_chunk_ids:
        cited = cited.strip()
        if cited in valid_ids:
            flags.append(f"duplicate_citation_removed:{_short(cited)}")
        elif cited not in by_id:
            flags.append(f"unknown_citation_removed:{_short(cited)}")
        else:
            valid_ids.append(cited)

    # 2. The model abstained: always show OUR message, never its wording.
    if raw.evidence_level == "insufficient":
        if valid_ids:
            flags.append("citations_dropped_for_insufficient_answer")
        return _fallback(flags)

    # 3. The model claims an answer: it must have text AND real evidence.
    answer_text = raw.answer.strip()
    if not answer_text:
        flags.append("empty_answer")
        return _fallback(flags)
    if not valid_ids:
        flags.append("answer_without_valid_citations")
        return _fallback(flags)

    citations = tuple(
        Citation(
            chunk_id=chunk_id,
            page=int(by_id[chunk_id].metadata["page"]),
            source=str(by_id[chunk_id].metadata.get("source", "document")),
            excerpt=by_id[chunk_id].page_content,
        )
        for chunk_id in valid_ids
    )
    return ValidatedAnswer(
        evidence_level=raw.evidence_level,
        answer=answer_text,
        citations=citations,
        flags=tuple(flags),
    )