"""Stage D: create embedding models behind one small, swappable interface.

The rest of the app only ever sees LangChain's `Embeddings` interface
(embed_documents / embed_query), so providers can be swapped without
touching chunking, indexing or retrieval code.

Two options:
- OpenAI (API): strong quality, small cost, text is sent to OpenAI.
- Local FastEmbed (BGE-small): free, private, runs on this machine's CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr


class EmbeddingConfigError(ValueError):
    """An embedding configuration or connection problem, with a safe message.

    `code` is machine-readable; the message is safe to show users and never
    contains provider error text (which can include fragments of API keys).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Everything the app needs to know about one supported embedding model."""

    key: str                     # stable identifier, e.g. "openai:text-embedding-3-small"
    provider: str
    model: str
    label: str                   # shown in the UI
    dimensions: int              # length of every vector this model produces
    requires_api_key: bool
    sends_text_externally: bool  # privacy: does document text leave this machine?


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "openai:text-embedding-3-small": EmbeddingModelSpec(
        key="openai:text-embedding-3-small",
        provider="openai",
        model="text-embedding-3-small",
        label="OpenAI · text-embedding-3-small",
        dimensions=1536,
        requires_api_key=True,
        sends_text_externally=True,
    ),
    "local:bge-small-en-v1.5": EmbeddingModelSpec(
        key="local:bge-small-en-v1.5",
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        label="Local · BGE small (free, private)",
        dimensions=384,
        requires_api_key=False,
        sends_text_externally=False,
    ),
}

DEFAULT_EMBEDDING_MODEL = "openai:text-embedding-3-small"


class FastEmbedLocalEmbeddings(Embeddings):
    """Adapter: makes a FastEmbed model fit LangChain's Embeddings interface.

    Implementing an interface means providing its methods — here, just two.
    BGE models embed questions and passages slightly differently, so each
    LangChain method routes to the matching FastEmbed method.
    """

    def __init__(self, model_name: str) -> None:
        # Imported here so the package is only required if this model is used.
        from fastembed import TextEmbedding

        # First use downloads the model and caches it; later uses load from cache.
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks. FastEmbed returns numpy arrays; LangChain expects lists."""
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed one question, using the model's query-specific handling."""
        return next(iter(self._model.query_embed(text))).tolist()


def get_embedding_spec(model_key: str) -> EmbeddingModelSpec:
    """Look up a supported model, or raise a clear error."""
    try:
        return EMBEDDING_MODELS[model_key]
    except KeyError:
        raise EmbeddingConfigError(
            "unsupported_model",
            f"'{model_key}' is not a supported embedding model.",
        ) from None


def create_embeddings(model_key: str, api_key: str | None = None) -> Embeddings:
    """Create an embedding model.

    API models make no network call here. Local models may download their
    weights the first time (a one-off, free download).

    Args:
        model_key: A key from EMBEDDING_MODELS.
        api_key: Provider key, if the model requires one. Passed in
            explicitly — this module never reads environment variables.

    Raises:
        EmbeddingConfigError: Unsupported model, missing key, missing
            package, or a local model that could not be loaded.
    """
    spec = get_embedding_spec(model_key)

    key = (api_key or "").strip()
    if spec.requires_api_key and not key:
        raise EmbeddingConfigError(
            "missing_api_key", f"{spec.label} needs an API key."
        )

    if spec.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=spec.model,
            api_key=SecretStr(key),  # prints as '**********', never the key
            max_retries=2,           # retry brief network hiccups, then give up
            timeout=30,              # seconds; never hang the app forever
        )

    if spec.provider == "fastembed":
        try:
            return FastEmbedLocalEmbeddings(spec.model)
        except ImportError:
            raise EmbeddingConfigError(
                "dependency_missing",
                "Local embeddings need the 'fastembed' package, which is "
                "not installed.",
            ) from None
        except Exception as exc:  # noqa: BLE001 — download/load failures
            raise EmbeddingConfigError(
                "model_unavailable",
                "The local embedding model could not be loaded. The first "
                "use needs an internet connection to download it.",
            ) from exc

    raise EmbeddingConfigError(
        "unsupported_provider", f"Provider '{spec.provider}' is not supported."
    )


def check_embeddings(embeddings: Embeddings, model_key: str) -> None:
    """Make one tiny, cheap call to prove the configuration really works.

    Catches bad keys, missing billing or network problems BEFORE we embed a
    whole document, and verifies the vector length matches the catalog.

    Raises:
        EmbeddingConfigError: With a safe, user-facing message.
    """
    spec = get_embedding_spec(model_key)
    try:
        vector = embeddings.embed_query("connection check")
    except Exception as exc:  # noqa: BLE001 — we translate every failure
        code, message = describe_provider_error(exc)
        raise EmbeddingConfigError(code, message) from exc

    if len(vector) != spec.dimensions:
        raise EmbeddingConfigError(
            "dimension_mismatch",
            f"Expected {spec.dimensions}-number vectors from {spec.label}, "
            f"got {len(vector)}. The model configuration is inconsistent.",
        )


def describe_provider_error(exc: Exception) -> tuple[str, str]:
    """Map a provider exception to (code, safe message).

    We match on exception TYPE and write our own messages. We deliberately
    never include str(exc): provider messages can contain partial keys.
    """
    try:
        import openai
    except ImportError:  # OpenAI SDK not installed: fall through to generic
        openai = None

    if openai is not None:
        if isinstance(exc, openai.AuthenticationError):
            return "invalid_api_key", (
                "The API key was rejected. Check that it was copied "
                "correctly and has not been revoked."
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return "permission_denied", (
                "This API key does not have access to the selected model."
            )
        if isinstance(exc, openai.RateLimitError):
            return "rate_limited", (
                "The provider refused the request: either too many requests "
                "were sent, or the account has no remaining credit. Wait a "
                "moment, or check the account's billing settings."
            )
        if isinstance(exc, openai.NotFoundError):
            return "model_not_found", (
                "The provider does not recognise this model name."
            )
        if isinstance(exc, openai.APIConnectionError):  # includes timeouts
            return "connection_error", (
                "Could not reach the provider. Check your internet connection."
            )

    return "provider_error", (
        "The embedding provider returned an unexpected error. Please try again."
    )