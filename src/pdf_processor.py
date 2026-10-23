"""Stage A + B: validate an uploaded PDF and extract its text page by page.

Everything here runs BEFORE any paid API call, so bad files are rejected
cheaply. This module knows nothing about Streamlit: it takes raw bytes and
returns plain Python objects, which keeps it easy to test and reuse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PureWindowsPath

import pymupdf
from langchain_core.documents import Document

from src.config import PDF_LIMITS, PDFLimits

# Every real PDF contains the signature "%PDF-" near the start. The PDF spec
# tolerates a little junk before it, so we search the first 1024 bytes
# instead of insisting it sits at byte 0.
PDF_SIGNATURE = b"%PDF-"
SIGNATURE_SEARCH_WINDOW = 1024


class PDFValidationError(ValueError):
    """Raised when an upload cannot be processed.

    Attributes:
        code: Stable, machine-readable reason (e.g. "encrypted"). Tests and
            UI logic check this instead of matching message text.
    The exception message itself is written for the end user.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PageText:
    """The extracted text of one page."""

    page_number: int  # 1-based, the way humans and citations count pages
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class ProcessedPDF:
    """A validated PDF and its per-page text. Read-only once created."""

    filename: str
    file_hash: str                   # SHA-256 fingerprint of the exact bytes
    page_count: int
    pages: tuple[PageText, ...]
    low_text_pages: tuple[int, ...]  # pages with little/no extractable text

    @property
    def text_pages(self) -> tuple[PageText, ...]:
        """Pages that have enough text to be worth chunking."""
        low = set(self.low_text_pages)
        return tuple(p for p in self.pages if p.page_number not in low)

    def to_documents(self) -> list[Document]:
        """Convert usable pages into LangChain Documents for the chunker.

        The metadata travels with every chunk cut from this page, which is
        what later lets us cite "page 3" truthfully.
        """
        return [
            Document(
                page_content=page.text,
                metadata={"source": self.filename, "page": page.page_number},
            )
            for page in self.text_pages
        ]


def compute_file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest: same bytes in, same fingerprint out."""
    return hashlib.sha256(file_bytes).hexdigest()


def safe_filename(filename: str) -> str:
    """Strip any folder path so only the bare file name is kept.

    PureWindowsPath understands both "/" and "\\" separators, so this works
    the same on Windows and on Streamlit Cloud's Linux servers.
    """
    return PureWindowsPath(filename).name or "document.pdf"


def _check_raw_bytes(file_bytes: bytes, limits: PDFLimits) -> None:
    """Cheap checks that need no PDF parsing at all."""
    if not file_bytes:
        raise PDFValidationError("empty_file", "The uploaded file is empty.")

    if len(file_bytes) > limits.max_file_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise PDFValidationError(
            "too_large",
            f"This file is {size_mb:.1f} MB. The limit is "
            f"{limits.max_file_size_mb} MB.",
        )

    if PDF_SIGNATURE not in file_bytes[:SIGNATURE_SEARCH_WINDOW]:
        raise PDFValidationError(
            "not_pdf",
            "This file is not a PDF. Its contents don't match the PDF "
            "format, even if its name ends in .pdf.",
        )


def process_pdf(
    file_bytes: bytes,
    filename: str,
    limits: PDFLimits = PDF_LIMITS,
) -> ProcessedPDF:
    """Validate an uploaded PDF and extract its text, page by page.

    Args:
        file_bytes: The raw contents of the uploaded file.
        filename: The name the user's file had (used for display/citations).
        limits: Validation limits; defaults to the app config. Tests pass
            custom limits to exercise rejections cheaply.

    Returns:
        A ProcessedPDF with per-page text and page numbers.

    Raises:
        PDFValidationError: If the file cannot or should not be processed.
    """
    # 1-3. Free checks first: empty, too large, not actually a PDF.
    _check_raw_bytes(file_bytes, limits)

    # 4. Try to parse it. FileDataError is a subclass of RuntimeError; we
    #    name both to make the intent explicit.
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise PDFValidationError(
            "corrupted",
            "This file could not be opened as a PDF. It may be damaged or "
            "incomplete.",
        ) from exc

    # `with` guarantees the document is closed even if a check below fails.
    with doc:
        # 5. Password-protected? (Some PDFs are "encrypted" with an empty
        #    password and open normally; needs_pass is only True when a
        #    password is actually required.)
        if doc.needs_pass:
            raise PDFValidationError(
                "encrypted",
                "This PDF is password-protected. Please upload an unlocked copy.",
            )

        page_count = doc.page_count
        if page_count == 0:
            raise PDFValidationError("no_pages", "This PDF contains no pages.")
        if page_count > limits.max_pages:
            raise PDFValidationError(
                "too_many_pages",
                f"This PDF has {page_count} pages. The limit is "
                f"{limits.max_pages}.",
            )

        # 6. Extract text. sort=True reorders text blocks into reading order
        #    (top-to-bottom, left-to-right). A damaged page can still fail
        #    here, so this is guarded too.
        try:
            pages = tuple(
                PageText(
                    page_number=index + 1,  # PyMuPDF counts from 0; humans from 1
                    text=page.get_text("text", sort=True).strip(),
                )
                for index, page in enumerate(doc)
            )
        except RuntimeError as exc:
            raise PDFValidationError(
                "corrupted",
                "Text could not be read from one or more pages. The file may "
                "be damaged.",
            ) from exc

    # Pages with (almost) no text are usually scanned images.
    low_text_pages = tuple(
        p.page_number for p in pages if p.char_count < limits.min_chars_per_page
    )
    if len(low_text_pages) == page_count:
        raise PDFValidationError(
            "no_text",
            "No readable text was found. This usually means the PDF is "
            "scanned (pages saved as images). Scanned PDFs need OCR, which "
            "this app does not support yet.",
        )

    return ProcessedPDF(
        filename=safe_filename(filename),
        file_hash=compute_file_hash(file_bytes),
        page_count=page_count,
        pages=pages,
        low_text_pages=low_text_pages,
    )