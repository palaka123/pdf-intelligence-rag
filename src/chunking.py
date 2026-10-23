"""Stage C: clean extracted text and split it into overlapping chunks.

Input:  per-page LangChain Documents (from ProcessedPDF.to_documents()).
Output: smaller Documents, each carrying its source, page, and a stable
        chunk_id such as "p2-c3" that citations will refer to.
"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNKING

# Below this, chunks are too small to carry meaningful context.
MIN_CHUNK_SIZE = 100

# Most meaningful boundary first; fall back only when a piece is too long.
# "" (split anywhere) is the last resort that guarantees the size limit.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RUNS_OF_SPACES = re.compile(r"[ \t\f\v]+")
_THREE_OR_MORE_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize PDF text without destroying its structure.

    - Unicode NFKC: ligatures ("ﬁ" -> "fi"), non-breaking and other odd
      spaces -> normal space, full-width characters -> normal ones.
    - Removes null characters some PDFs contain.
    - Collapses runs of spaces/tabs and strips each line.
    - Keeps single line breaks and paragraph breaks; reduces 3+ newlines
      to one paragraph break.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_RUNS_OF_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _THREE_OR_MORE_NEWLINES.sub("\n\n", text)
    return text.strip()


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Reject chunking settings that cannot work. Raises ValueError."""
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(
            f"Chunk size must be at least {MIN_CHUNK_SIZE} characters "
            f"(got {chunk_size})."
        )
    if chunk_overlap < 0:
        raise ValueError(f"Chunk overlap cannot be negative (got {chunk_overlap}).")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Chunk overlap ({chunk_overlap}) must be smaller than chunk "
            f"size ({chunk_size})."
        )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNKING.chunk_size,
    chunk_overlap: int = CHUNKING.chunk_overlap,
) -> list[Document]:
    """Clean and split per-page Documents into chunks with stable IDs.

    Each page is split on its own, so a chunk never spans two pages and
    its page citation is always exact.

    Args:
        documents: Per-page Documents whose metadata includes "page".
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters repeated between neighbouring chunks.

    Returns:
        Chunk Documents. Metadata keeps the page's "source" and "page" and
        adds "chunk_id" ("p2-c3"), "chunk_index" (1-based, per page),
        "start_index" (offset within the cleaned page text) and "char_count".
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",   # keep ". " with the sentence it ends
        add_start_index=True,   # record where each chunk starts in the page
    )

    chunks: list[Document] = []
    for page_doc in documents:
        cleaned = clean_text(page_doc.page_content)
        if not cleaned:
            continue  # nothing left after cleaning: no chunks for this page

        page = page_doc.metadata["page"]
        pieces = splitter.create_documents([cleaned], metadatas=[dict(page_doc.metadata)])

        for index, piece in enumerate(pieces, start=1):
            chunks.append(
                Document(
                    page_content=piece.page_content,
                    metadata={
                        **piece.metadata,
                        "chunk_id": f"p{page}-c{index}",
                        "chunk_index": index,
                        "char_count": len(piece.page_content),
                    },
                )
            )
    return chunks