"""Probe which features each catalog chat model actually supports.

Makes a few tiny real API calls per model (fractions of a cent in total).
Usage: python -m scripts.probe_chat_models
"""

import os

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.llm_factory import CHAT_MODELS
from src.models import GroundedAnswer
from src.prompts import build_messages

PROBE_DOCS = [
    Document(
        page_content="Payment is due within 30 days of the invoice date.",
        metadata={"chunk_id": "p1-c1", "page": 1, "source": "probe.pdf"},
    )
]
SAY_OK = "Reply with the single word OK."


def short_error(exc: Exception) -> str:
    """Brief error description; never prints auth error text (may contain key fragments)."""
    if isinstance(exc, openai.AuthenticationError):
        return "AuthenticationError (key rejected)"
    return f"{type(exc).__name__}: {str(exc)[:160]}"


def attempt(label: str, call):
    """Run one probe call and print OK/FAIL (ASCII only: safe for the Windows console)."""
    try:
        result = call()
        print(f"    OK   {label}")
        return True, result
    except Exception as exc:  # noqa: BLE001 — a probe reports every failure
        print(f"    FAIL {label} -> {short_error(exc)}")
        return False, None


def probe_temperature(base: dict) -> bool:
    """True only if temperature=0 survives construction AND the API accepts it."""
    llm = ChatOpenAI(temperature=0, **base)
    if llm.temperature != 0:
        # The library removed the setting: a successful call would prove nothing.
        print(f"    FAIL temperature=0 -> removed by langchain-openai "
              f"(value is now {llm.temperature!r}); never sent to the API")
        return False
    ok, _ = attempt("temperature=0 (confirmed sent to the API)", lambda: llm.invoke(SAY_OK))
    return ok


def main() -> None:
    load_dotenv()
    key = SecretStr(os.environ.get("OPENAI_API_KEY") or "")

    for spec in CHAT_MODELS.values():
        print(f"\n{spec.label}  [{spec.model}]")

        ok, _ = attempt(
            "basic call, Chat Completions API",
            lambda: ChatOpenAI(model=spec.model, api_key=key, max_retries=0,
                               timeout=60, use_responses_api=False).invoke(SAY_OK),
        )
        use_responses = False
        if not ok:
            ok, _ = attempt(
                "basic call, Responses API",
                lambda: ChatOpenAI(model=spec.model, api_key=key, max_retries=0,
                                   timeout=60, use_responses_api=True).invoke(SAY_OK),
            )
            if not ok:
                print("    -> model unusable with this key; skipping")
                continue
            use_responses = True

        base = dict(model=spec.model, api_key=key, max_retries=0, timeout=60,
                    use_responses_api=use_responses)

        temperature_ok = probe_temperature(base)

        structured = ChatOpenAI(**base).with_structured_output(
            GroundedAnswer, method="json_schema", strict=True
        )
        for question in ["When is payment due?", "What is the late-payment fee?"]:
            ok, parsed = attempt(
                f'structured answer to "{question}"',
                lambda: structured.invoke(build_messages(question, PROBE_DOCS)),
            )
            if ok:
                print(f"         -> {parsed.evidence_level} | {parsed.answer} | cites {parsed.cited_chunk_ids}")

        print(f"    => catalog: supports_temperature={temperature_ok}, "
              f"use_responses_api={use_responses}")


if __name__ == "__main__":
    main()