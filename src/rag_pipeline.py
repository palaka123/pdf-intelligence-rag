"""End-to-end pipeline: the two functions the UI calls.

index_document():  PDF bytes -> validated, chunked, embedded, indexed.
answer_question(): question -> retrieve -> generate -> validate -> answer.

Every error raised here subclasses ValueError and carries a safe message,
so the UI can catch ValueError and show the message directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.chunking import chunk_documents
from src.config import CHUNKING, RETRIEVAL
from src.generation import generate_grounded_answer
from src.guardrails import ValidatedAnswer, validate_answer, validate_question
from src.pdf_processor import ProcessedPDF, process_pdf
from src.prompts import PROMPT_VERSION
from src.retrieval import DocumentRetriever
from src.vector_store import DocumentIndex


@dataclass(frozen=True)
class IndexedDocument:
    """A processed PDF plus its index, and the settings that built them."""

    processed: ProcessedPDF
    index: DocumentIndex
    chunk_size: int
    chunk_overlap: int

    def is_current_for(
        self, file_hash: str, embedding_model_key: str, chunk_size: int, chunk_overlap: int
    ) -> bool:
        """True if NOTHING that shapes the index has changed.

        A new file, embedding model, OR chunk setting means different chunks
        (and different chunk ids), so the index must be rebuilt.
        """
        return (
            self.index.is_current_for(file_hash, embedding_model_key)
            and self.chunk_size == chunk_size
            and self.chunk_overlap == chunk_overlap
        )


@dataclass(frozen=True)
class RAGResponse:
    """Everything about one answered question, for display and inspection."""

    question: str
    answer: ValidatedAnswer
    retrieved: tuple[Document, ...]  # with "score" and "rank" metadata
    chat_model_key: str
    prompt_version: str


def index_document(
    file_bytes: bytes,
    filename: str,
    embeddings: Embeddings,
    embedding_model_key: str,
    chunk_size: int = CHUNKING.chunk_size,
    chunk_overlap: int = CHUNKING.chunk_overlap,
) -> IndexedDocument:
    """Validate, extract, chunk, embed and index one PDF. Embeds each chunk once."""
    processed = process_pdf(file_bytes, filename)
    chunks = chunk_documents(processed.to_documents(), chunk_size, chunk_overlap)
    index = DocumentIndex.build(chunks, embeddings, embedding_model_key, processed.file_hash)
    return IndexedDocument(
        processed=processed,
        index=index,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def answer_question(
    question: str,
    indexed: IndexedDocument,
    llm: BaseChatModel,
    chat_model_key: str,
    k: int = RETRIEVAL.top_k,
) -> RAGResponse:
    """Answer one question about an indexed document, with verified citations."""
    clean_question = validate_question(question)              # input guardrail
    retrieved = DocumentRetriever(index=indexed.index, k=k).invoke(clean_question)
    raw_answer = generate_grounded_answer(llm, clean_question, retrieved)
    validated = validate_answer(raw_answer, retrieved)        # output guardrail
    return RAGResponse(
        question=clean_question,
        answer=validated,
        retrieved=tuple(retrieved),
        chat_model_key=chat_model_key,
        prompt_version=PROMPT_VERSION,
    )