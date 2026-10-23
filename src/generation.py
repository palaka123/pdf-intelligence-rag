"""Stage H: ask the LLM for a structured, grounded answer."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.embeddings import describe_provider_error
from src.models import GroundedAnswer
from src.prompts import build_messages

GENERIC_LLM_ERROR = "The language model returned an unexpected error. Please try again."


class GenerationError(ValueError):
    """The LLM call failed or returned unusable output. Safe message, with `code`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _describe_with_cause_chain(exc: BaseException) -> tuple[str, str]:
    """Map an error to (code, safe message), looking inside wrapper exceptions.

    Libraries may wrap the provider's exception in their own type; the
    original is then available as __cause__ (or __context__).
    """
    current: BaseException | None = exc
    for _ in range(5):  # a few levels is plenty; avoids endless loops
        if current is None:
            break
        code, message = describe_provider_error(current)  # type: ignore[arg-type]
        if code != "provider_error":
            return code, message
        current = current.__cause__ or current.__context__
    return "provider_error", GENERIC_LLM_ERROR


def generate_grounded_answer(
    llm: BaseChatModel, question: str, docs: list[Document]
) -> GroundedAnswer:
    """Send the grounded prompt and parse the structured answer.

    Raises:
        GenerationError: Provider failure, or output that does not fit the
            schema ("malformed_output").
    """
    structured_llm = llm.with_structured_output(
        GroundedAnswer,
        method="json_schema",  # provider-enforced JSON schema
        strict=True,           # every field must be present and well-typed
        include_raw=True,      # return parse errors instead of raising them
    )

    try:
        result = structured_llm.invoke(build_messages(question, docs))
    except Exception as exc:  # noqa: BLE001 — translated to a safe message
        code, message = _describe_with_cause_chain(exc)
        raise GenerationError(code, message) from exc

    parsed = result.get("parsed")
    if result.get("parsing_error") is not None or not isinstance(parsed, GroundedAnswer):
        raise GenerationError(
            "malformed_output",
            "The language model returned an answer in an unexpected format. "
            "Please try again.",
        )
    return parsed