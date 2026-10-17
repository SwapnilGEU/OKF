"""OKF as an MCP server.

Exposes the okf package to any MCP client (the LangGraph agents, the MCP Inspector, Claude Desktop):
  tools:     search_okf, ask_okf, get_concept
  resource:  okf://status

Run it (from the project folder, in the env where `pip install -e .` was done):
  python -m okf_mcp.server                 # streamable HTTP on http://127.0.0.1:8000/mcp
  python -m okf_mcp.server --stdio         # stdio: the client starts this process itself
  mcp dev okf_mcp/server.py                # opens the MCP Inspector in the browser
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from okf import OKF, Passage, Source

mcp = FastMCP("okf", instructions="Search and answer questions about five AI research papers "
                                  "(GPT-4 report, Sparks of AGI, HELM, Foundation Models, LLM survey).")


@lru_cache(maxsize=1)
def get_okf() -> OKF:
    """Load the pipeline once per server process (bundle, BM25, Qdrant, LLM), then reuse it."""
    return OKF.load()


class OKFAnswer(BaseModel):
    """What ask_okf returns: the answer and its sources, without the 18,000-character context."""
    answer: str
    refused: bool = Field(description="true if the papers don't contain the answer")
    sources: list[Source]
    model: str
    seconds: float


@mcp.tool()
def search_okf(
    question: Annotated[str, Field(description="The question or keywords to search for")],
) -> list[Passage]:
    """Find the passages from the five papers that best match a question (no LLM, about 1-2 s).
    Each passage has a source (concept id, title, paper, pages) and its text."""
    return get_okf().search(question)


@mcp.tool()
def ask_okf(
    question: Annotated[str, Field(description="A question about the five papers")],
) -> OKFAnswer:
    """Answer a question from the five papers with citations like [papers/helm/...].
    Retrieves passages, then the local model writes the answer (about 5-10 s)."""
    a = get_okf().ask(question)
    return OKFAnswer(answer=a.answer, refused=a.refused, sources=a.sources, model=a.model,
                     seconds=round((a.retrieval_ms + a.generation_ms) / 1000, 1))


@mcp.tool()
def get_concept(
    concept_id: Annotated[str, Field(description="A concept id as cited, e.g. papers/helm/4-4-calibration-and-uncertainty")],
) -> Passage:
    """Read one topic page or paper section in full, e.g. to follow a citation."""
    okf = get_okf()
    if concept_id not in okf.bundle:
        raise ValueError(f"No concept {concept_id!r}. Use an id exactly as it appears in a citation.")
    return okf.get_concept(concept_id)


@mcp.resource("okf://status")
def status() -> dict:
    """What's loaded: concepts, vectors on/off, which model answers."""
    return get_okf().status()


def main():
    p = argparse.ArgumentParser(description="OKF MCP server")
    p.add_argument("--stdio", action="store_true", help="talk over stdin/stdout instead of HTTP")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-warmup", action="store_true", help="don't load the model into VRAM at startup")
    args = p.parse_args()
    print("loading the OKF pipeline...", file=sys.stderr)        # stderr: stdout belongs to MCP in --stdio mode
    okf = get_okf()
    print(okf.status(), file=sys.stderr)
    if not args.no_warmup:                                        # load the model into VRAM now, not on the first question
        try:
            okf.llm.complete("Reply with OK.", "OK")
            print("model warmed up", file=sys.stderr)
        except Exception as exc:
            print(f"warm-up failed ({type(exc).__name__}: {exc}); ask_okf will fail until the model is reachable", file=sys.stderr)
    print("ready", file=sys.stderr, flush=True)
    if args.stdio:
        mcp.run(transport="stdio")
    else:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
