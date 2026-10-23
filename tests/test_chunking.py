"""Tests for Stage C: cleaning and chunking."""

import pytest
from langchain_core.documents import Document

from src.chunking import chunk_documents, clean_text
from src.pdf_processor import process_pdf
from tests.conftest import page_text


def page_doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": "t.pdf", "page": page})


# A long single-page text: 60 short, uniquely numbered sentences (~1,300 chars).
LONG_TEXT = " ".join(f"This is sentence {i:03d}." for i in range(1, 61))


# ---------- cleaning ----------

def test_clean_text_collapses_spaces_and_strips_lines():
    assert clean_text("  Hello    world  \n\t indented ") == "Hello world\nindented"


def test_clean_text_keeps_one_paragraph_break():
    assert clean_text("Para one.\n\n\n\n\nPara two.") == "Para one.\n\nPara two."


def test_clean_text_normalizes_pdf_typography():
    # "\ufb01" is the "ﬁ" ligature; "\u00a0" is a non-breaking space.
    assert clean_text("\ufb01nance\u00a0team") == "finance team"


# ---------- parameters ----------

@pytest.mark.parametrize(
    "size, overlap",
    [(50, 10), (800, -1), (800, 800), (800, 900)],
)
def test_invalid_chunk_params_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([page_doc(LONG_TEXT)], chunk_size=size, chunk_overlap=overlap)


# ---------- chunking behaviour ----------

def test_chunks_respect_size_limit():
    chunks = chunk_documents([page_doc(LONG_TEXT)], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = chunk_documents([page_doc(LONG_TEXT)], chunk_size=200, chunk_overlap=50)
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current.page_content.split(".")[0]
        assert first_sentence in previous.page_content


def test_zero_overlap_means_no_repetition():
    chunks = chunk_documents([page_doc(LONG_TEXT)], chunk_size=200, chunk_overlap=0)
    joined_length = sum(len(c.page_content) for c in chunks)
    assert joined_length <= len(LONG_TEXT)


def test_metadata_and_stable_chunk_ids():
    docs = [page_doc(LONG_TEXT, page=1), page_doc("Short second page text.", page=2)]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids[0] == "p1-c1"
    assert ids[-1] == "p2-c1"
    assert len(ids) == len(set(ids))  # unique
    assert all(c.metadata["source"] == "t.pdf" for c in chunks)

    # Same input -> same IDs (citations and eval sets depend on this).
    again = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert [c.metadata["chunk_id"] for c in again] == ids


def test_page_that_is_empty_after_cleaning_produces_no_chunks():
    chunks = chunk_documents([page_doc("   \n\n\t  "), page_doc("Real text here.", page=2)])
    assert [c.metadata["page"] for c in chunks] == [2]


# ---------- integration with Stage A/B ----------

def test_chunks_never_mix_pages(make_pdf):
    processed = process_pdf(make_pdf([page_text(1), page_text(2), page_text(3)]), "r.pdf")
    chunks = chunk_documents(processed.to_documents())

    for chunk in chunks:
        page = chunk.metadata["page"]
        assert f"Page {page} marker" in chunk.page_content
        other_pages = {1, 2, 3} - {page}
        assert not any(f"Page {p} marker" in chunk.page_content for p in other_pages)