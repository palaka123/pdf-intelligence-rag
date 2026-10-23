"""Central configuration for PDF Intelligence.

Every tunable number lives here so that the validator, chunker, UI and
tests all read the same values. Change a limit once, here, and it
changes everywhere.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PDFLimits:
    """Limits enforced BEFORE any expensive processing or API call."""

    max_pages: int = 5
    max_file_size_mb: int = 10
    # A page with fewer extracted characters than this is treated as
    # having no usable text (likely scanned/image-only).
    min_chars_per_page: int = 20

    @property
    def max_file_size_bytes(self) -> int:
        """File-size limit converted to bytes, for comparing with len(file_bytes)."""
        return self.max_file_size_mb * 1024 * 1024


@dataclass(frozen=True)
class ChunkingDefaults:
    """Starting values for the text splitter — experimental, not 'optimal'."""

    chunk_size: int = 800      # characters, not tokens
    chunk_overlap: int = 120   # characters shared between neighbouring chunks


@dataclass(frozen=True)
class RetrievalDefaults:
    """How many chunks to retrieve per question."""

    top_k: int = 4


# Single shared instances the rest of the app imports.
PDF_LIMITS = PDFLimits()
CHUNKING = ChunkingDefaults()
RETRIEVAL = RetrievalDefaults()