"""Shared test helpers. pytest loads this file automatically."""

import pymupdf
import pytest


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