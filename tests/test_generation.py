"""Tests for Stage H generation: structured call, parsing failures, safe errors."""

import httpx
import openai
import pytest
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from src.generation import GENERIC_LLM_ERROR, GenerationError, generate_grounded_answer
from tests.fakes import FakeStructuredLLM, answer

FAKE_KEY = "sk-test-not-a-real-key-12345"
DOCS = [Document(page_content="Payment is due in 30 days.", metadata={"chunk_id": "p1-c1", "page": 1, "source": "doc.pdf"})]


def auth_error() -> openai.AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.AuthenticationError(
        f"Incorrect API key provided: {FAKE_KEY}",
        response=httpx.Response(401, request=request),
        body=None,
    )


def test_returns_parsed_answer_using_strict_schema():
    llm = FakeStructuredLLM(answer())
    result = generate_grounded_answer(llm, "When is payment due?", DOCS)

    assert result.cited_chunk_ids == ["p1-c1"]
    assert llm.structured_kwargs == {"method": "json_schema", "strict": True, "include_raw": True}


def test_sends_grounded_prompt():
    llm = FakeStructuredLLM(answer())
    generate_grounded_answer(llm, "When is payment due?", DOCS)

    messages = llm.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert "When is payment due?" in messages[1].content
    assert 'id="p1-c1"' in messages[1].content


def test_parsing_failure_becomes_malformed_output():
    llm = FakeStructuredLLM(parsing_error=ValueError("not JSON"))
    with pytest.raises(GenerationError) as err:
        generate_grounded_answer(llm, "q", DOCS)
    assert err.value.code == "malformed_output"


def test_auth_error_is_safe():
    llm = FakeStructuredLLM(error=auth_error())
    with pytest.raises(GenerationError) as err:
        generate_grounded_answer(llm, "q", DOCS)
    assert err.value.code == "invalid_api_key"
    assert FAKE_KEY not in str(err.value)


def test_wrapped_auth_error_is_still_recognised():
    # Simulates a library wrapping the provider's error in its own type.
    wrapper = RuntimeError("wrapped by a library")
    wrapper.__cause__ = auth_error()
    llm = FakeStructuredLLM(error=wrapper)

    with pytest.raises(GenerationError) as err:
        generate_grounded_answer(llm, "q", DOCS)
    assert err.value.code == "invalid_api_key"


def test_unknown_error_gets_llm_specific_message():
    llm = FakeStructuredLLM(error=RuntimeError(f"boom {FAKE_KEY}"))
    with pytest.raises(GenerationError) as err:
        generate_grounded_answer(llm, "q", DOCS)
    assert err.value.code == "provider_error"
    assert str(err.value) == GENERIC_LLM_ERROR