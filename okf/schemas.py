"""The data the pipeline hands out, as pydantic models.

Pydantic gives three things the later stages need:
- validation: a field that's the wrong type fails loudly instead of travelling on
- .model_dump() / .model_dump_json(): ready for an API response, a log line or an export
- .model_json_schema(): the MCP server, the A2A agent card and Outlines all read this schema
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SectionHit(BaseModel):
    id: str = Field(description="concept id of the section")
    found_by: str = Field(description="which searches found it: vector, bm25, link, joined with +")
    score: float = Field(description="fused RRF score x 1000")


class Retrieval(BaseModel):
    question: str
    parts: list[str] = Field(default_factory=list, description="sub-questions if the question was split")
    papers: list[str] = Field(default_factory=list, description="papers the question names")
    topics: list[str] = Field(default_factory=list, description="topic pages chosen (they go first in the context)")
    sections: list[SectionHit] = Field(default_factory=list)
    vectors_used: bool = False
    timing: dict[str, float] = Field(default_factory=dict)


class Source(BaseModel):
    id: str = Field(description="concept id, as cited in the answer: [id]")
    kind: Literal["section", "topic"]
    title: str
    paper: str | None = None
    pages: str | None = Field(default=None, description="e.g. 'pp. 31-32'")
    label: str = Field(description="one human-readable line for the UI")


class Passage(BaseModel):
    """One block of retrieved knowledge (what the search tool returns)."""
    source: Source
    text: str


class Answer(BaseModel):
    question: str
    answer: str
    refused: bool = Field(description="the model said the bundle does not contain this")
    sources: list[Source] = Field(default_factory=list, description="the context blocks the answer cites")
    used: list[str] = Field(default_factory=list, description="every concept id that was in the context")
    retrieval_context: list[str] = Field(default_factory=list, description="the exact blocks the model saw")
    retrieval: Retrieval
    model: str
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
