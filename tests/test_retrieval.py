"""Tests for Stage F: the LangChain retriever adapter."""

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.retrieval import DocumentRetriever
from src.vector_store import DocumentIndex
from tests.conftest import KeywordEmbeddings

MODEL = "local:bge-small-en-v1.5"

CHUNKS = [
    Document(
        page_content="Invoice payment is due within thirty days of receipt.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_id": "p1-c1"},
    ),
    Document(
        page_content="Apples and oranges are sold at the farmers market.",
        metadata={"source": "doc.pdf", "page": 2, "chunk_id": "p2-c1"},
    ),
    Document(
        page_content="The office closes early on public holidays.",
        metadata={"source": "doc.pdf", "page": 3, "chunk_id": "p3-c1"},
    ),
]


@pytest.fixture
def index() -> DocumentIndex:
    return DocumentIndex.build(CHUNKS, KeywordEmbeddings(), MODEL, "hash-a")


def test_invoke_returns_documents_with_score_and_rank(index):
    docs = DocumentRetriever(index=index, k=2).invoke("When is the invoice payment due?")

    assert len(docs) == 2
    assert docs[0].metadata["chunk_id"] == "p1-c1"
    assert docs[0].metadata["rank"] == 1
    assert docs[0].metadata["score"] >= docs[1].metadata["score"]
    assert docs[0].metadata["page"] == 1  # original metadata preserved


def test_stored_chunks_are_never_modified(index):
    # If scores leaked into the stored chunks, a follow-up question would
    # see the previous question's score.
    retriever = DocumentRetriever(index=index, k=3)
    retriever.invoke("invoice payment")
    retriever.invoke("office holidays")

    for stored in index.chunks:
        assert "score" not in stored.metadata
        assert "rank" not in stored.metadata


def test_default_k_comes_from_config(index):
    from src.config import RETRIEVAL

    assert DocumentRetriever(index=index).k == RETRIEVAL.top_k


def test_invalid_k_is_rejected_at_creation(index):
    with pytest.raises(ValidationError):
        DocumentRetriever(index=index, k=0)