"""Stage E: a FAISS vector index scoped to ONE document and ONE embedding model.

Design rules:
- One index per document (keyed by the PDF's SHA-256 hash), so chunks from
  different documents can never mix.
- The index keeps the embedding model that built it and uses that same
  model for every query, so a model mismatch is impossible by construction.
- Indexes live only in memory. There is deliberately no save/load: common
  FAISS persistence relies on pickle files, and loading a pickle can run
  arbitrary code. Rebuilding a 5-page index takes seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings import (
    EmbeddingConfigError,
    describe_provider_error,
    get_embedding_spec,
)


class VectorStoreError(ValueError):
    """An indexing or search problem, with a machine-readable `code`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and how similar it was to the question."""

    rank: int        # 1 = most similar
    score: float     # cosine similarity — comparable ONLY within one model
    chunk: Document

    @property
    def chunk_id(self) -> str:
        return self.chunk.metadata["chunk_id"]

    @property
    def page(self) -> int:
        return self.chunk.metadata["page"]


def _to_normalized_matrix(vectors: list[list[float]], dimensions: int) -> np.ndarray:
    """Convert vectors to the float32 matrix FAISS needs, scaled to length 1.

    After normalization, FAISS's inner-product search equals cosine
    similarity. Also verifies every vector has the expected length.
    """
    matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise VectorStoreError(
            "dimension_mismatch",
            f"Expected vectors of {dimensions} numbers, got shape {matrix.shape}.",
        )
    faiss.normalize_L2(matrix)  # in place: each row now has length 1
    return matrix


class DocumentIndex:
    """A searchable index for one document, built with one embedding model.

    Create it with DocumentIndex.build(...), not the constructor.
    """

    def __init__(
        self,
        *,
        faiss_index: faiss.Index,
        chunks: tuple[Document, ...],
        embeddings: Embeddings,
        embedding_model_key: str,
        document_hash: str,
    ) -> None:
        self._index = faiss_index
        self._chunks = chunks          # position i <-> FAISS vector i
        self._embeddings = embeddings  # the SAME model is used for queries
        self.embedding_model_key = embedding_model_key
        self.document_hash = document_hash

    @classmethod
    def build(
        cls,
        chunks: list[Document],
        embeddings: Embeddings,
        embedding_model_key: str,
        document_hash: str,
    ) -> DocumentIndex:
        """Embed every chunk once and load the vectors into a FAISS index.

        Raises:
            VectorStoreError: No chunks, or vectors of the wrong size.
            EmbeddingConfigError: The embedding provider failed (safe message).
        """
        if not chunks:
            raise VectorStoreError("no_chunks", "There are no text chunks to index.")

        spec = get_embedding_spec(embedding_model_key)

        try:
            raw_vectors = embeddings.embed_documents([c.page_content for c in chunks])
        except Exception as exc:  # noqa: BLE001 — translate to a safe message
            code, message = describe_provider_error(exc)
            raise EmbeddingConfigError(code, message) from exc

        matrix = _to_normalized_matrix(raw_vectors, spec.dimensions)
        if matrix.shape[0] != len(chunks):
            raise VectorStoreError(
                "count_mismatch",
                f"Got {matrix.shape[0]} vectors for {len(chunks)} chunks.",
            )

        faiss_index = faiss.IndexFlatIP(spec.dimensions)  # exact search
        faiss_index.add(matrix)

        return cls(
            faiss_index=faiss_index,
            chunks=tuple(chunks),
            embeddings=embeddings,
            embedding_model_key=embedding_model_key,
            document_hash=document_hash,
        )

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return self._index.ntotal

    @property
    def chunks(self) -> tuple[Document, ...]:
        """All indexed chunks, in index order (read-only)."""
        return self._chunks

    def is_current_for(self, document_hash: str, embedding_model_key: str) -> bool:
        """True if this index matches the current document AND model setting.

        The UI uses this to decide whether to rebuild after the user uploads
        a different file or picks a different embedding model.
        """
        return (
            self.document_hash == document_hash
            and self.embedding_model_key == embedding_model_key
        )

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return the k chunks most similar to the query, best first.

        Note: this ALWAYS returns up to k results, even if none of them is
        actually relevant. Judging relevance is a later stage's job.
        """
        query = query.strip()
        if not query:
            raise VectorStoreError("empty_query", "The question is empty.")
        if k < 1:
            raise VectorStoreError("invalid_k", f"k must be at least 1 (got {k}).")
        k = min(k, self.size)

        spec = get_embedding_spec(self.embedding_model_key)
        try:
            raw_query_vector = self._embeddings.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            code, message = describe_provider_error(exc)
            raise EmbeddingConfigError(code, message) from exc

        query_matrix = _to_normalized_matrix([raw_query_vector], spec.dimensions)
        scores, positions = self._index.search(query_matrix, k)

        return [
            SearchResult(rank=rank, score=float(score), chunk=self._chunks[int(position)])
            for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1)
            if position != -1  # FAISS uses -1 for "no result"
        ]