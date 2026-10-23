"""Tests for Stage E: FAISS indexing, retrieval, isolation, and safety."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings import EmbeddingConfigError
from src.vector_store import DocumentIndex, VectorStoreError
from tests.conftest import KeywordEmbeddings

MODEL = "local:bge-small-en-v1.5"  # 384 dims, matching KeywordEmbeddings' default


def chunk(text: str, page: int, source: str = "doc.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_id": f"p{page}-c1"},
    )


CHUNKS = [
    chunk("Invoice payment is due within thirty days of receipt.", 1),
    chunk("Apples and oranges are sold at the farmers market.", 2),
    chunk("The office closes early on public holidays.", 3),
]


def build(chunks=CHUNKS, document_hash="hash-a", embeddings=None) -> DocumentIndex:
    return DocumentIndex.build(chunks, embeddings or KeywordEmbeddings(), MODEL, document_hash)


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise RuntimeError("provider exploded")

    def embed_query(self, text):
        raise RuntimeError("provider exploded")


# ---------- retrieval quality ----------

def test_most_relevant_chunk_ranks_first():
    results = build().search("When is the invoice payment due?", k=3)
    assert results[0].chunk_id == "p1-c1"
    assert results[0].page == 1


def test_results_are_ranked_and_carry_metadata():
    results = build().search("office holidays", k=3)
    assert [r.rank for r in results] == [1, 2, 3]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # best first
    assert all(r.chunk.metadata["source"] == "doc.pdf" for r in results)


def test_identical_text_scores_one():
    # Proves normalization: identical vectors have cosine similarity 1.0.
    results = build().search(CHUNKS[1].page_content, k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_index_is_clamped():
    assert len(build().search("anything at all", k=10)) == 3


# ---------- validation ----------

def test_empty_chunk_list_is_rejected():
    with pytest.raises(VectorStoreError) as err:
        build(chunks=[])
    assert err.value.code == "no_chunks"


def test_empty_query_is_rejected():
    with pytest.raises(VectorStoreError) as err:
        build().search("   ", k=3)
    assert err.value.code == "empty_query"


def test_invalid_k_is_rejected():
    with pytest.raises(VectorStoreError) as err:
        build().search("invoice", k=0)
    assert err.value.code == "invalid_k"


def test_wrong_vector_size_is_rejected():
    with pytest.raises(VectorStoreError) as err:
        build(embeddings=KeywordEmbeddings(dimensions=1536))  # model expects 384
    assert err.value.code == "dimension_mismatch"


def test_embedding_failure_during_build_gets_safe_error():
    with pytest.raises(EmbeddingConfigError) as err:
        build(embeddings=FailingEmbeddings())
    assert err.value.code == "provider_error"
    assert "exploded" not in str(err.value)  # raw provider text never leaks


# ---------- isolation ----------

def test_is_current_for_detects_document_or_model_change():
    index = build(document_hash="hash-a")
    assert index.is_current_for("hash-a", MODEL)
    assert not index.is_current_for("hash-b", MODEL)                  # new document
    assert not index.is_current_for("hash-a", "openai:text-embedding-3-small")  # new model


def test_indexes_for_different_documents_never_mix():
    index_a = build(document_hash="hash-a")
    tempting_b = [chunk("Invoice payment terms for supplier invoice payment.", 1, source="b.pdf")]
    build(chunks=tempting_b, document_hash="hash-b")

    # Document B matches the query better, but must never appear in A's results.
    results = index_a.search("invoice payment", k=3)
    assert all(r.chunk.metadata["source"] == "doc.pdf" for r in results)