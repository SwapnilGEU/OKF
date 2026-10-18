"""OKFAgent: the compiled graph plus everything it needs, behind three simple methods.

    agent = OKFAgent.start()                      # MCP tools, qwen, Postgres memory + qa_log
    reply = agent.ask("What score did GPT-4 get on the bar exam?", thread_id="chat-1")
    reply = agent.ask("And GPT-3.5?", thread_id="chat-1")      # same thread: remembers the first turn
    agent.history("chat-1")
    agent.close()

thread_id names a conversation. The checkpointer saves the state per thread_id in Postgres,
so a thread can be continued later, even after a restart.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from okf.llm import make_llm

from .config import AgentConfig
from .db import ensure_qa_log
from .graph import AGENT_NAME, build_graph
from .mcp_tools import load_tools
from .state import AgentReply


class OKFAgent:
    def __init__(self, config: AgentConfig, graph, tools, log_conn, saver_cm):
        self.config, self.graph, self.tools = config, graph, tools
        self._log_conn, self._saver_cm = log_conn, saver_cm
        self._pool = ThreadPoolExecutor(max_workers=1)     # runs the graph off Jupyter's event-loop thread

    @classmethod
    def start(cls, config: AgentConfig | None = None, memory: bool = True, log: bool = True) -> "OKFAgent":
        config = config or AgentConfig.load()
        tools = load_tools({"okf": {"url": config.mcp_url, "transport": "streamable_http"}})
        missing = {"search_okf"} - set(tools)
        if missing:
            raise RuntimeError(f"the MCP server at {config.mcp_url} doesn't offer {missing}")
        llm = make_llm(config.okf)

        log_conn = None
        if log:
            log_conn = psycopg.connect(config.postgres_url, autocommit=True)
            ensure_qa_log(log_conn)

        saver_cm = saver = None
        if memory:
            saver_cm = PostgresSaver.from_conn_string(config.postgres_url)
            saver = saver_cm.__enter__()
            saver.setup()                                  # creates LangGraph's checkpoint tables (once)

        graph = build_graph(tools, llm, log_conn).compile(checkpointer=saver)
        return cls(config, graph, tools, log_conn, saver_cm)

    # --- use it --------------------------------------------------------------------------------
    def _run(self, fn, *args):
        return self._pool.submit(fn, *args).result()

    def ask(self, question: str, thread_id: str = "default") -> AgentReply:
        cfg = {"configurable": {"thread_id": thread_id}}
        state = self._run(self.graph.invoke, {"messages": [HumanMessage(question)]}, cfg)
        return AgentReply(thread_id=thread_id, question=state["question"], answer=state["answer"],
                          refused=state["refused"], sources=state.get("sources", []),
                          bad_citations=state.get("bad_citations", []), followup=state.get("followup", False),
                          search_query=state.get("search_query", ""),
                          timings={k: v for k, v in state.get("timings", {}).items() if k != "started"},
                          llm=state.get("llm", {}), log_id=state.get("log_id"))

    def steps(self, question: str, thread_id: str = "default") -> list[dict]:
        """Run one turn node by node and report what each node changed (stream_mode='updates')."""
        cfg = {"configurable": {"thread_id": thread_id}}

        def go():
            out, last = [], time.perf_counter()
            for update in self.graph.stream({"messages": [HumanMessage(question)]}, cfg, stream_mode="updates"):
                for node, changes in update.items():
                    now = time.perf_counter()
                    out.append({"node": node, "ms": round((now - last) * 1000), "changed": sorted((changes or {}).keys()),
                                "values": changes})
                    last = now
            return out
        return self._run(go)

    def history(self, thread_id: str = "default") -> list[tuple[str, str]]:
        """The chat saved for a thread: [(role, text), ...]."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return [(m.type, m.content) for m in snapshot.values.get("messages", [])]

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def close(self):
        self._pool.shutdown(wait=True)
        if self._saver_cm is not None:
            self._saver_cm.__exit__(None, None, None)
        if self._log_conn is not None:
            self._log_conn.close()


__all__ = ["OKFAgent", "AGENT_NAME"]
