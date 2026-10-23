"""Fake LLM for offline tests: returns a preset answer and records what it was sent."""

from langchain_core.runnables import RunnableLambda

from src.models import GroundedAnswer


class FakeStructuredLLM:
    """Stands in for a chat model's with_structured_output() behaviour.

    Args:
        answer: The GroundedAnswer to return.
        error: If given, raised when invoked (simulates provider failures).
        parsing_error: If given, returned as a parse failure.
    """

    def __init__(self, answer=None, *, error=None, parsing_error=None):
        self.answer = answer
        self.error = error
        self.parsing_error = parsing_error
        self.calls = []                 # the message lists it was sent
        self.structured_kwargs = None   # how with_structured_output was called

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = kwargs

        def respond(messages):
            self.calls.append(messages)
            if self.error is not None:
                raise self.error
            return {
                "raw": None,
                "parsed": None if self.parsing_error else self.answer,
                "parsing_error": self.parsing_error,
            }

        return RunnableLambda(respond)


def answer(level="explicit", text="An answer.", cited=("p1-c1",)) -> GroundedAnswer:
    """Shorthand for building a GroundedAnswer in tests."""
    return GroundedAnswer(evidence_level=level, answer=text, cited_chunk_ids=list(cited))