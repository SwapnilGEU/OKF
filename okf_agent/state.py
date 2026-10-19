"""The agent's state: the one dict that flows through the graph.

Every node receives the current state and returns only the keys it changed; LangGraph merges
those updates in. Most keys are simply overwritten. `messages` is special: its reducer
`add_messages` APPENDS new messages instead of replacing the list, so the chat history grows
turn by turn (and the checkpointer saves it to Postgres after every node).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # the chat so far (appended, never replaced)
    question: str              # this turn's question (the last user message)
    followup: bool             # looks like a follow-up to the previous question?
    previous: dict | None      # the previous question and answer, if this is a follow-up
    search_query: str          # what was actually sent to search_okf
    passages: list[dict]       # search_okf's result: [{"source": {...}, "text": "..."}]
    answer: str
    sources: list[dict]        # passages the answer cites
    refused: bool              # "The bundle does not contain this."
    bad_citations: list[str]   # ids cited in the answer but NOT in the passages (made up)
    guard: dict                # what the guardrails found or fixed this turn (okf_agent.guards)
    timings: dict              # milliseconds per step
    llm: dict                  # model name and token counts
    log_id: int | None         # row id in qa_log
    stream_id: str | None      # id of the streamed answer message (the final message reuses it)


class AgentReply(BaseModel):
    """What OKFAgent.ask() returns: one turn, ready to show or to send to another agent."""
    thread_id: str
    question: str
    answer: str
    refused: bool
    sources: list[dict] = Field(default_factory=list)
    bad_citations: list[str] = Field(default_factory=list)
    followup: bool = False
    search_query: str = ""
    timings: dict = Field(default_factory=dict)
    llm: dict = Field(default_factory=dict)
    log_id: int | None = None
    guard: dict = Field(default_factory=dict)
