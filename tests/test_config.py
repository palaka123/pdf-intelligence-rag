"""Smoke test: proves pytest can import our package and reads the right defaults."""

import dataclasses

import pytest

from src.config import CHUNKING, PDF_LIMITS, RETRIEVAL


def test_pdf_limits_defaults():
    assert PDF_LIMITS.max_pages == 5
    assert PDF_LIMITS.max_file_size_bytes == 10 * 1024 * 1024


def test_overlap_smaller_than_chunk():
    # Overlap >= chunk size would make the splitter loop or produce duplicates.
    assert CHUNKING.chunk_overlap < CHUNKING.chunk_size


def test_retrieval_top_k_positive():
    assert RETRIEVAL.top_k > 0


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PDF_LIMITS.max_pages = 50  # type: ignore[misc]