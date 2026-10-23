"""Shared data schemas used across pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    """The structured answer the LLM must return.

    The field descriptions below are sent to the model as part of the
    schema, so they are effectively part of the prompt.
    """

    evidence_level: Literal["explicit", "inferred", "insufficient"] = Field(
        description=(
            "'explicit' if the excerpts directly state the answer; "
            "'inferred' if the answer is a reasonable conclusion drawn from "
            "the excerpts but not stated outright; 'insufficient' if the "
            "excerpts do not contain enough information to answer."
        )
    )
    answer: str = Field(
        description=(
            "The answer for the user, based only on the excerpts. If "
            "evidence_level is 'inferred', say that it is an inference. If "
            "'insufficient', briefly say the document does not appear to "
            "contain the answer."
        )
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The id attribute of every excerpt the answer relies on, copied "
            "exactly (for example 'p2-c1'). Empty if evidence_level is "
            "'insufficient'."
        ),
    )