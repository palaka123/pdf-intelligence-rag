"""Shared test helpers. pytest loads this file automatically."""

import re
import zlib

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings


# ---------- PDF helpers (Stage A/B) ----------

def build_pdf(page_texts: list[str], user_password: str | None = None) -> bytes:
    """Build a PDF in memory: one page per string; "" makes a blank page.

    A blank page behaves exactly like a scanned page for text extraction
    (no text layer), so it lets us test scanned-PDF handling for free.
    """
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)

    if user_password:
        data = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_password,
            owner_pw=user_password + "-owner",
        )
    else:
        data = doc.tobytes()
    doc.close()
    return data


def page_text(n: int) -> str:
    """Realistic page text with a unique marker we can search for."""
    return f"Page {n} marker.\nThis page contains enough text to pass validation."


@pytest.fixture
def make_pdf():
    """Give tests access to the PDF builder."""
    return build_pdf


# ---------- Embedding helpers (Stage E) ----------

class KeywordEmbeddings(Embeddings):
    """Offline fake: a bag-of-words vector, so texts sharing words are similar.

    Each word is hashed into one of `dimensions` slots and counted.
    zlib.crc32 is used (not hash()) because it gives the same result every
    run, keeping tests deterministic.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)