"""The OKF agent as a service: what `langgraph dev` loads (see langgraph.json).

    langgraph dev --no-browser          →  http://127.0.0.1:2024
        A2A endpoint   POST /a2a/<assistant id>
        agent card     GET  /.well-known/agent-card.json?assistant_id=<assistant id>

Two graphs, two A2A agents:
    graph        okf_agent    answers from the bundle         (OKF MCP server, :8000)
    scout_graph  paper_scout  finds papers on arXiv            (arXiv MCP server, :8001)

okf_agent is the same graph as OKFAgent (prepare → search → answer → check → log), with two differences:

1. No checkpointer here. The LangGraph server brings its own memory and creates one thread
   per A2A conversation (contextId = thread_id). With `langgraph dev` that memory is kept in
   the .langgraph_api/ folder, so it survives a restart. qa_log still goes to Postgres.
2. Nothing connects at import time. The MCP tools and the Postgres connection open on the first
   question, so the server starts even if the MCP server or Postgres is still starting up.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

import psycopg

from okf.llm import make_llm

from .config import AgentConfig
from .db import ensure_qa_log
from .graph import build_graph
from .mcp_tools import load_tools
from .scout import SCOUT_NAME, build_scout_graph
from .tracing import traced_llm

CONFIG = AgentConfig.load()


class LazyTools(Mapping):
    """{name: tool} from an MCP server, loaded the first time a node asks for a tool."""

    def __init__(self, url: str, name: str = "okf", start_hint: str = "python -m okf_mcp.server"):
        self.url, self.name, self.start_hint = url, name, start_hint
        self._tools, self._lock = None, threading.Lock()

    def _load(self) -> dict:
        with self._lock:
            if self._tools is None:
                try:
                    self._tools = load_tools({self.name: {"url": self.url, "transport": "streamable_http"}})
                except Exception as exc:
                    raise RuntimeError(f"can't reach the {self.name} MCP server at {self.url}. Is it running? "
                                       f"({self.start_hint})") from exc
        return self._tools

    def __getitem__(self, name):
        return self._load()[name]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


class LazyLogConnection:
    """A psycopg connection for qa_log, opened on first use and reopened if Postgres restarted."""

    def __init__(self, url: str):
        self.url, self._conn, self._lock = url, None, threading.Lock()

    def execute(self, *args, **kwargs):
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg.connect(self.url, autocommit=True)
                ensure_qa_log(self._conn)
            return self._conn.execute(*args, **kwargs)


LLM = traced_llm(make_llm(CONFIG.okf))          # one model for both agents (one copy in Ollama)
LOG = LazyLogConnection(CONFIG.postgres_url)

graph = build_graph(LazyTools(CONFIG.mcp_url), LLM, LOG).compile()
graph.name = "okf_agent"

ARXIV_HINT = "set TRANSPORT=http&& set PORT=8001&& uvx arxiv-mcp-server@0.8.2"
scout_graph = build_scout_graph(LazyTools(CONFIG.arxiv_mcp_url, "arxiv", ARXIV_HINT), LLM, LOG).compile()
scout_graph.name = SCOUT_NAME
