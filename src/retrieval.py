"""Stage F: expose a DocumentIndex through LangChain's standard retriever interface.

A LangChain retriever is anything that maps a question to a list of
Documents. Implementing the interface lets this index plug into any
LangChain component that expects a retriever, and gives us `.invoke()`.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.config import RETRIEVAL
from src.vector_store import DocumentIndex


class DocumentRetriever(BaseRetriever):
    """Retrieves the top-k chunks of ONE document for a question.

    Each returned Document is a copy whose metadata also contains:
        "score": cosine similarity (comparable only within one model)
        "rank":  1 = most similar
    """

    # BaseRetriever is a Pydantic model; DocumentIndex is our own class,
    # so Pydantic must be told to accept it as a field type.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: DocumentIndex
    k: int = Field(default=RETRIEVAL.top_k, ge=1)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Called by .invoke(); returns copies, never the stored chunks."""
        results = self.index.search(query, k=self.k)
        return [
            Document(
                page_content=result.chunk.page_content,
                metadata={
                    **result.chunk.metadata,  # copy: originals stay untouched
                    "score": result.score,
                    "rank": result.rank,
                },
            )
            for result in results
        ]