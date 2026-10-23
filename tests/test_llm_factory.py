"""Tests for Stage G2: chat model factory and schema compatibility. Offline."""

import pytest

from src.llm_factory import (
    CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    FACTUAL_TEMPERATURE,
    LLMConfigError,
    create_chat_model,
)
from src.models import GroundedAnswer

FAKE_KEY = "sk-test-not-a-real-key-12345"


def test_default_model_is_in_catalog():
    assert DEFAULT_CHAT_MODEL in CHAT_MODELS


def test_missing_api_key_is_rejected():
    with pytest.raises(LLMConfigError) as err:
        create_chat_model(DEFAULT_CHAT_MODEL, api_key="")
    assert err.value.code == "missing_api_key"


def test_unsupported_model_is_rejected():
    with pytest.raises(LLMConfigError) as err:
        create_chat_model("openai:gpt-imaginary", api_key=FAKE_KEY)
    assert err.value.code == "unsupported_model"


def test_api_key_is_hidden_in_object_repr():
    llm = create_chat_model(DEFAULT_CHAT_MODEL, api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(llm)


@pytest.mark.parametrize("model_key", list(CHAT_MODELS))
def test_temperature_sent_only_when_supported(model_key):
    llm = create_chat_model(model_key, api_key=FAKE_KEY)
    expected = FACTUAL_TEMPERATURE if CHAT_MODELS[model_key].supports_temperature else None
    assert llm.temperature == expected


def test_answer_schema_has_no_optional_fields():
    # OpenAI strict structured output requires every field to be required.
    schema = GroundedAnswer.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])