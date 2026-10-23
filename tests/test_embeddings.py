"""Tests for Stage D: embedding factory, connection check, safe errors.

All tests run offline and free: no real API calls are made.
"""

import httpx
import openai
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from src.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfigError,
    check_embeddings,
    create_embeddings,
)

FAKE_KEY = "sk-test-not-a-real-key-12345"


class FailingEmbeddings(Embeddings):
    """An embeddings object whose every call raises a chosen exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def embed_documents(self, texts):
        raise self.exc

    def embed_query(self, text):
        raise self.exc


def openai_error(error_class, status: int) -> Exception:
    """Build a real OpenAI SDK exception, the way the SDK itself does."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, request=request)
    # Simulates a provider message that leaks part of the key.
    return error_class(f"Incorrect API key provided: {FAKE_KEY}", response=response, body=None)


# ---------- factory ----------

def test_missing_api_key_is_rejected():
    with pytest.raises(EmbeddingConfigError) as err:
        create_embeddings(DEFAULT_EMBEDDING_MODEL, api_key="   ")
    assert err.value.code == "missing_api_key"


def test_unsupported_model_is_rejected():
    with pytest.raises(EmbeddingConfigError) as err:
        create_embeddings("openai:does-not-exist", api_key=FAKE_KEY)
    assert err.value.code == "unsupported_model"


def test_api_key_is_hidden_in_object_repr():
    embeddings = create_embeddings(DEFAULT_EMBEDDING_MODEL, api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(embeddings)
    assert FAKE_KEY not in str(embeddings)


# ---------- connection check ----------

def test_check_passes_with_matching_dimensions():
    check_embeddings(DeterministicFakeEmbedding(size=1536), DEFAULT_EMBEDDING_MODEL)


def test_check_detects_dimension_mismatch():
    with pytest.raises(EmbeddingConfigError) as err:
        check_embeddings(DeterministicFakeEmbedding(size=384), DEFAULT_EMBEDDING_MODEL)
    assert err.value.code == "dimension_mismatch"


@pytest.mark.parametrize(
    "error_class, status, expected_code",
    [
        (openai.AuthenticationError, 401, "invalid_api_key"),
        (openai.PermissionDeniedError, 403, "permission_denied"),
        (openai.NotFoundError, 404, "model_not_found"),
        (openai.RateLimitError, 429, "rate_limited"),
    ],
)
def test_provider_errors_get_safe_messages(error_class, status, expected_code):
    failing = FailingEmbeddings(openai_error(error_class, status))
    with pytest.raises(EmbeddingConfigError) as err:
        check_embeddings(failing, DEFAULT_EMBEDDING_MODEL)

    assert err.value.code == expected_code
    # The provider's text contained the key; our message must not.
    assert FAKE_KEY not in str(err.value)


def test_connection_error_is_recognised():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    failing = FailingEmbeddings(openai.APIConnectionError(request=request))
    with pytest.raises(EmbeddingConfigError) as err:
        check_embeddings(failing, DEFAULT_EMBEDDING_MODEL)
    assert err.value.code == "connection_error"


def test_unknown_errors_get_generic_message():
    failing = FailingEmbeddings(RuntimeError(f"internal detail {FAKE_KEY}"))
    with pytest.raises(EmbeddingConfigError) as err:
        check_embeddings(failing, DEFAULT_EMBEDDING_MODEL)
    assert err.value.code == "provider_error"
    assert FAKE_KEY not in str(err.value)