"""The OKF agent as a LangGraph graph.

    START → prepare → search → answer → check → log → END

prepare  read the question; is it a follow-up? (then search with the previous question added)
search   call search_okf on the MCP server (retrieval only, no LLM)
answer   qwen writes the answer from the passages: the ONE model call per question
check    which citations are real, which are made up, did it refuse? append the reply to the chat
log      one row in Postgres' qa_log

Each node is a plain function: it gets the state, returns the keys it changed.
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from okf import cited_ids, is_refusal
from okf.llm import SYSTEM_PROMPT, user_message
from okf.pipeline import CITATION_RE

from .db import insert_qa_log
from .mcp_tools import call_tool
from .state import AgentState

AGENT_NAME = "okf_agent"

FOLLOWUP_OPENERS = re.compile(r"^\s*(and|but|also|so|then|what about|how about|why|how come|compared)\b", re.I)
FOLLOWUP_POINTERS = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|same|other)\b", re.I)


def looks_like_followup(question: str) -> bool:
    """Cheap rule, no model call: short, and starts like a follow-up or points back at something."""
    short = len(question.split()) <= 15
    return short and bool(FOLLOWUP_OPENERS.search(question) or FOLLOWUP_POINTERS.search(question))


def text_of(message) -> str:
    """A message's text. Messages typed in Python carry a string; messages that arrive over A2A
    can carry a list of parts like [{"type": "text", "text": "..."}]."""
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(p if isinstance(p, str) else p.get("text", "") for p in content
                     if isinstance(p, str) or p.get("type", "text") == "text")


def previous_turn(messages) -> tuple[str | None, str | None]:
    """The last question and answer BEFORE the current question (the current one is messages[-1])."""
    q = a = None
    for m in reversed(messages[:-1]):
        if isinstance(m, AIMessage) and a is None and q is None:
            a = text_of(m)
        elif isinstance(m, HumanMessage):
            q = text_of(m)
            break
    return q, a


def build_graph(tools: dict, llm, log_conn=None, agent_name: str = AGENT_NAME) -> StateGraph:
    """The graph, not yet compiled. `tools` from mcp_tools.load_tools, `llm` from okf.llm.make_llm,
    `log_conn` a psycopg connection for qa_log (None = don't log)."""

    def prepare(state: AgentState) -> dict:
        question = text_of(state["messages"][-1]).strip()
        prev_q, prev_a = previous_turn(state["messages"])
        followup = prev_q is not None and looks_like_followup(question)
        return {"question": question, "followup": followup,
                "search_query": f"{prev_q} {question}" if followup else question,
                "previous": {"question": prev_q, "answer": prev_a} if followup else None,
                "timings": {"started": time.time()}, "bad_citations": [], "log_id": None}

    def search(state: AgentState) -> dict:
        t0 = time.perf_counter()
        passages = call_tool(tools["search_okf"], {"question": state["search_query"]})
        passages = [p for p in passages if isinstance(p, dict) and "source" in p]
        return {"passages": passages,
                "timings": {**state["timings"], "search_ms": round((time.perf_counter() - t0) * 1000, 1)}}

    def answer(state: AgentState) -> dict:
        context = "\n\n".join(p["text"] for p in state["passages"])        # the blocks, in order: same as okf.ask()
        user = user_message(state["question"], context)                    # the evaluated prompt
        prev = state.get("previous")
        if prev:                                                           # follow-up: remind the model of the last turn
            note = f"Earlier in this chat:\nQ: {prev['question']}\nA: {(prev['answer'] or '')[:800]}\n\n"
            user = user.replace("Question: ", note + "Question: ", 1)
        reply = llm.complete(SYSTEM_PROMPT, user)
        return {"answer": reply.text,
                "llm": {"model": llm.name, "prompt_tokens": reply.prompt_tokens, "answer_tokens": reply.answer_tokens},
                "timings": {**state["timings"], "generation_ms": reply.generation_ms}}

    def check(state: AgentState) -> dict:
        used = [p["source"]["id"] for p in state["passages"]]
        cited = cited_ids(state["answer"], used)                           # real citations, in context order
        mentioned = {part.strip() for group in CITATION_RE.findall(state["answer"])
                     for part in re.split(r"[,;]", group)}
        bad = sorted(c for c in mentioned if "/" in c and c not in used)  # looks like an id, but wasn't given
        sources = [p["source"] for p in state["passages"] if p["source"]["id"] in cited]
        reply = AIMessage(content=state["answer"], additional_kwargs={"sources": sources, "agent": agent_name})
        return {"sources": sources, "bad_citations": bad, "refused": is_refusal(state["answer"]),
                "messages": [reply]}                                       # add_messages APPENDS this to the chat

    def log(state: AgentState, config) -> dict:
        timings = {**state["timings"], "total_ms": round((time.time() - state["timings"]["started"]) * 1000, 1)}
        log_id = None
        if log_conn is not None:
            try:
                thread_id = config["configurable"].get("thread_id", "default")
                log_id = insert_qa_log(log_conn, thread_id, agent_name, {**state, "timings": timings})
            except Exception as exc:                                        # logging must never break the answer
                print(f"qa_log insert failed: {type(exc).__name__}: {exc}")
        return {"timings": timings, "log_id": log_id}

    g = StateGraph(AgentState)
    for name, fn in [("prepare", prepare), ("search", search), ("answer", answer), ("check", check), ("log", log)]:
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "search")
    g.add_edge("search", "answer")
    g.add_edge("answer", "check")
    g.add_edge("check", "log")
    g.add_edge("log", END)
    return g
