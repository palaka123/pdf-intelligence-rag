"""End-to-end pipeline tests: real PDF processing, chunking and FAISS;
fake embeddings and a fake LLM so everything runs offline and free."""

import pytest

from src.guardrails import GuardrailError
from src.rag_pipeline import answer_question, index_document
from tests.conftest import KeywordEmbeddings, page_text
from tests.fakes import FakeStructuredLLM, answer

MODEL = "local:bge-small-en-v1.5"
CHAT = "openai:gpt-6-luna"


@pytest.fixture
def indexed(make_pdf):
    pdf = make_pdf([page_text(1), "Invoices are payable within thirty days.", page_text(3)])
    return index_document(pdf, "report.pdf", KeywordEmbeddings(), MODEL)


def test_index_document_tracks_everything_that_shapes_the_index(indexed):
    file_hash = indexed.processed.file_hash
    assert indexed.index.size == 3
    assert indexed.is_current_for(file_hash, MODEL, 800, 120)
    assert not indexed.is_current_for("other-file", MODEL, 800, 120)
    assert not indexed.is_current_for(file_hash, "openai:text-embedding-3-small", 800, 120)
    assert not indexed.is_current_for(file_hash, MODEL, 400, 120)  # chunk setting changed


def test_end_to_end_answer_cites_the_right_page(indexed):
    llm = FakeStructuredLLM(answer(text="Within thirty days.", cited=["p2-c1"]))
    response = answer_question("When are invoices payable?", indexed, llm, CHAT)

    assert response.answer.is_answered
    assert response.answer.cited_pages == (2,)
    assert response.retrieved[0].metadata["chunk_id"] == "p2-c1"  # retrieval found it too
    assert response.prompt_version == "v1"


def test_forged_citation_never_reaches_the_user(indexed):
    llm = FakeStructuredLLM(answer(text="Trust me.", cited=["p9-c9"]))
    response = answer_question("When are invoices payable?", indexed, llm, CHAT)

    assert not response.answer.is_answered
    assert response.answer.citations == ()


def test_invalid_question_is_stopped_before_the_llm(indexed):
    llm = FakeStructuredLLM(answer())
    with pytest.raises(GuardrailError):
        answer_question("   ", indexed, llm, CHAT)
    assert llm.calls == []  # no paid call was made


def test_retrieved_chunks_carry_scores_for_inspection(indexed):
    llm = FakeStructuredLLM(answer(cited=["p2-c1"]))
    response = answer_question("invoices payable", indexed, llm, CHAT)

    ranks = [doc.metadata["rank"] for doc in response.retrieved]
    assert ranks == sorted(ranks)
    assert all("score" in doc.metadata for doc in response.retrieved)