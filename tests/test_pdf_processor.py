"""Tests for Stage A (validation) and Stage B (extraction)."""

import pytest

from src.config import PDFLimits
from src.pdf_processor import PDFValidationError, process_pdf
from tests.conftest import page_text


def test_valid_pdf_extracts_pages_with_metadata(make_pdf):
    result = process_pdf(make_pdf([page_text(1), page_text(2)]), "report.pdf")

    assert result.page_count == 2
    assert result.low_text_pages == ()
    assert "Page 2 marker" in result.pages[1].text
    assert result.pages[1].page_number == 2  # 1-based, not 0-based

    docs = result.to_documents()
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert all(d.metadata["source"] == "report.pdf" for d in docs)


def test_rejects_empty_bytes():
    with pytest.raises(PDFValidationError) as err:
        process_pdf(b"", "empty.pdf")
    assert err.value.code == "empty_file"


def test_rejects_non_pdf_content_even_with_pdf_name():
    with pytest.raises(PDFValidationError) as err:
        process_pdf(b"Hello, I am a text file pretending to be a PDF.", "fake.pdf")
    assert err.value.code == "not_pdf"


def test_rejects_oversized_file(make_pdf):
    tiny_limit = PDFLimits(max_file_size_mb=0)  # any non-empty file is too big
    with pytest.raises(PDFValidationError) as err:
        process_pdf(make_pdf([page_text(1)]), "big.pdf", limits=tiny_limit)
    assert err.value.code == "too_large"


def test_rejects_more_than_max_pages(make_pdf):
    six_pages = make_pdf([page_text(n) for n in range(1, 7)])
    with pytest.raises(PDFValidationError) as err:
        process_pdf(six_pages, "long.pdf")
    assert err.value.code == "too_many_pages"


def test_accepts_exactly_max_pages(make_pdf):
    # Boundary test: 5 pages must pass. Off-by-one bugs live at limits.
    five_pages = make_pdf([page_text(n) for n in range(1, 6)])
    assert process_pdf(five_pages, "five.pdf").page_count == 5


def test_rejects_password_protected(make_pdf):
    locked = make_pdf([page_text(1)], user_password="secret")
    with pytest.raises(PDFValidationError) as err:
        process_pdf(locked, "locked.pdf")
    assert err.value.code == "encrypted"


def test_rejects_corrupted_pdf():
    garbage = b"%PDF-1.7\n" + b"this is not a valid pdf body " * 50
    with pytest.raises(PDFValidationError) as err:
        process_pdf(garbage, "broken.pdf")
    # MuPDF may fail to open it, or "repair" it into zero pages — both are
    # correct rejections.
    assert err.value.code in {"corrupted", "no_pages"}


def test_rejects_pdf_with_no_text(make_pdf):
    scanned_like = make_pdf(["", ""])
    with pytest.raises(PDFValidationError) as err:
        process_pdf(scanned_like, "scan.pdf")
    assert err.value.code == "no_text"


def test_mixed_pdf_flags_low_text_pages_and_skips_them(make_pdf):
    mixed = make_pdf([page_text(1), "", page_text(3)])
    result = process_pdf(mixed, "mixed.pdf")

    assert result.low_text_pages == (2,)
    assert [d.metadata["page"] for d in result.to_documents()] == [1, 3]


def test_filename_path_is_stripped(make_pdf):
    result = process_pdf(make_pdf([page_text(1)]), "..\\..\\secrets\\report.pdf")
    assert result.filename == "report.pdf"


def test_same_bytes_give_same_hash(make_pdf):
    data = make_pdf([page_text(1)])
    assert process_pdf(data, "a.pdf").file_hash == process_pdf(data, "b.pdf").file_hash