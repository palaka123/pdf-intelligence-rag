"""Tests for display safety and exports."""

from langchain_core.documents import Document

from src.guardrails import Citation, ValidatedAnswer
from src.pdf_processor import process_pdf
from src.presentation import build_chat_markdown, build_extracted_text, escape_markdown
from src.rag_pipeline import RAGResponse
from tests.conftest import page_text


def response(question="When is it due?", text="Within 30 days.") -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=ValidatedAnswer(
            evidence_level="explicit",
            answer=text,
            citations=(Citation("p2-c1", 2, "doc.pdf", "Due in 30 days."),),
            flags=(),
        ),
        retrieved=(Document(page_content="Due in 30 days.", metadata={}),),
        chat_model_key="openai:gpt-6-luna",
        prompt_version="v1",
    )


def test_escape_defuses_markdown_images_and_links():
    escaped = escape_markdown("![x](https://evil.example/?leak=1) [click](https://evil.example)")
    assert "![" not in escaped and "](" not in escaped
    assert "\\!\\[" in escaped


def test_escape_keeps_plain_text_and_line_breaks():
    assert escape_markdown("Hello world\nLine two") == "Hello world  \nLine two"


def test_extracted_text_has_page_headers_and_is_utf8(make_pdf):
    processed = process_pdf(make_pdf([page_text(1), page_text(2)]), "r.pdf")
    text = build_extracted_text(processed).decode("utf-8")
    assert "===== Page 1 =====" in text and "===== Page 2 =====" in text
    assert "Page 2 marker" in text


def test_chat_export_includes_questions_answers_and_pages():
    markdown = build_chat_markdown([response()], "doc.pdf").decode("utf-8")
    assert "## Q1. When is it due?" in markdown
    assert "Within 30 days\\." in markdown           # escaped answer text
    assert "page 2 (`p2-c1`)" in markdown
    assert "verify important facts" in markdown       # the disclaimer is present


def test_chat_export_escapes_untrusted_content():
    markdown = build_chat_markdown([response(text="![](https://evil.example)")], "doc.pdf").decode("utf-8")
    assert "![" not in markdown