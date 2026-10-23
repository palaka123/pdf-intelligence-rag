"""Stage G2: create chat models from a small catalog of verified models.

Mirrors embeddings.py: the catalog describes each model once, including
which optional features it actually supports. Capabilities are verified
empirically with scripts/probe_chat_models.py, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

# Used only for models confirmed to accept a temperature setting.
# Low temperature makes answers CONSISTENT, not CORRECT.
FACTUAL_TEMPERATURE = 0.0


class LLMConfigError(ValueError):
    """A chat-model configuration problem, with a machine-readable `code`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ChatModelSpec:
    """Everything the app needs to know about one supported chat model."""

    key: str
    provider: str
    model: str
    label: str
    # True only if temperature verifiably REACHES the API (see the probe).
    supports_temperature: bool = False
    use_responses_api: bool = False


# Capability flags verified on 2026-10-23 with scripts/probe_chat_models.py
# AND tests/test_llm_factory.py. Re-check before adding a model, after a
# provider announcement, or after upgrading langchain-openai.
CHAT_MODELS: dict[str, ChatModelSpec] = {
    "openai:gpt-6-luna": ChatModelSpec(
        key="openai:gpt-6-luna",
        provider="openai",
        model="gpt-6-luna",
        label="OpenAI · GPT-6 Luna (recommended: newest, lowest cost)",
        supports_temperature=False,  # API accepts only the default (1)
    ),
    "openai:gpt-4.1-mini": ChatModelSpec(
        key="openai:gpt-4.1-mini",
        provider="openai",
        model="gpt-4.1-mini",
        label="OpenAI · GPT-4.1 mini (older; most consistent answers)",
        supports_temperature=True,   # verified: temperature 0 reaches the API
    ),
    "openai:gpt-5.6-luna": ChatModelSpec(
        key="openai:gpt-5.6-luna",
        provider="openai",
        model="gpt-5.6-luna",
        label="OpenAI · GPT-5.6 Luna",
        supports_temperature=False,  # langchain-openai drops it for this model
    ),
}

DEFAULT_CHAT_MODEL = "openai:gpt-6-luna"


def get_chat_spec(model_key: str) -> ChatModelSpec:
    """Look up a supported chat model, or raise a clear error."""
    try:
        return CHAT_MODELS[model_key]
    except KeyError:
        raise LLMConfigError(
            "unsupported_model", f"'{model_key}' is not a supported chat model."
        ) from None


def create_chat_model(model_key: str, api_key: str | None) -> BaseChatModel:
    """Create a chat model. Makes no network call.

    Optional parameters are sent only if the catalog says the model
    supports them, so an unsupported setting can never cause an API error.

    Raises:
        LLMConfigError: Unsupported model or missing key.
    """
    spec = get_chat_spec(model_key)

    key = (api_key or "").strip()
    if not key:
        raise LLMConfigError("missing_api_key", f"{spec.label} needs an API key.")

    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": spec.model,
            "api_key": SecretStr(key),
            "max_retries": 2,
            "timeout": 60,  # answers take longer than embeddings
        }
        if spec.supports_temperature:
            kwargs["temperature"] = FACTUAL_TEMPERATURE
        if spec.use_responses_api:
            kwargs["use_responses_api"] = True
        return ChatOpenAI(**kwargs)

    raise LLMConfigError(
        "unsupported_provider", f"Provider '{spec.provider}' is not supported."
    )