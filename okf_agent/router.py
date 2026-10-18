"""The router: one entry point that sends each message to the right agent(s) over A2A.

                       ┌─► ask_okf_agent ───┐
    START → route ─────┼─► ask_paper_scout ─┼─► combine → log → END
                       └─► direct ──────────┘
    (conditional edge: one branch, BOTH ask_ branches in parallel, or direct)

route     decides, in this order:
            1. greeting / "what can you do"   → direct (rule, no model call)
            2. qwen picks from the agents' cards: [], one agent, or both (JSON). It also sees the
               previous message and who answered it, so it can tell "And GPT-3.5?" (same agent)
               from a new topic. (A word rule can't: "...newer papers on it?" has "it" but is new.)
            3. qwen's reply unusable          → short follow-up? same agent as last time : keyword rule
ask_<x>   sends the message to agent x over A2A. For a follow-up (same agent(s) as last turn) it reuses that
          agent's A2A conversation (contextId kept in the state), so the agent's own memory works;
          a new request starts a fresh conversation with the agent.
direct    a code-written reply: what the system can do, built from the agents' cards
combine   one answer → as is; two answers → under a heading each
log       qa_log row, agent = "okf_router", search_query = the route taken

The router knows the agents ONLY through their A2A cards (name + description). Add a graph to
langgraph.json and to OKF_ROUTER_AGENTS, and the router can use it.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Annotated

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .a2a_client import A2AClient, langgraph_card_url
from .db import insert_qa_log
from .graph import looks_like_followup, previous_turn, text_of
from .state import AgentState

ROUTER_NAME = "okf_router"
SHORT_FOLLOWUP = 6
HEADINGS = {"okf_agent": "From the bundle", "paper_scout": "On arXiv"}

GREETING_RE = re.compile(r"^\s*(hi+|hello|hey|hiya|yo|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|cool|"
                         r"help|who are you|what can you do|what do you do|how does this work)\b[\s!?.]*$", re.I)
SCOUT_WORDS_RE = re.compile(r"\b(papers?|arxiv|preprints?|publications?|literature|recent|latest|newer|newest|"
                            r"new work|state of the art|sota)\b", re.I)

ROUTE_PROMPT = """You route a user's message to the right assistant(s).

Assistants:
{agents}

Reply with JSON only: {{"agents": [...]}}
- Usually exactly one name.
- Both names only if the message clearly asks for both things.
- [] if no assistant fits (greetings, small talk, unrelated tasks like poems or code).

Examples:
"What score did GPT-4 get on the bar exam?" -> {{"agents": ["okf_agent"]}}
"Find recent papers on mixture of experts" -> {{"agents": ["paper_scout"]}}
"What does the bundle say about RLHF, and are there newer papers on it?" -> {{"agents": ["okf_agent", "paper_scout"]}}
"Write me a poem about cats" -> {{"agents": []}}

If a previous message is shown: when the new message continues it ("And GPT-3.5?", "only ones since 2025",
"what about the second one?"), pick the same assistant(s) as before; when it is a new request, decide from
the new message alone."""


def merge_dict(old: dict | None, new: dict | None) -> dict:
    """Reducer for keys that parallel branches write to at the same time. None = reset (start of a turn)."""
    if new is None:
        return {}
    return {**(old or {}), **new}


class RouterState(AgentState, total=False):
    route: list[str]                                   # agents chosen this turn ([] = direct)
    route_by: str                                      # rule-greeting / followup / llm / rule-keywords
    last_route: list[str]                              # previous turn's route (kept by the server's memory)
    contexts: Annotated[dict, merge_dict]              # {agent: A2A contextId} — one conversation per agent
    replies: Annotated[dict, merge_dict]               # {agent: {"text", "state", "ms"}} — this turn


class AgentDirectory:
    """The agents the router may use, discovered from their A2A cards on first use."""

    def __init__(self, base_url: str, names: list[str]):
        self.base_url, self.names = base_url, names
        self._clients: dict[str, A2AClient] = {}
        self._lock = threading.Lock()

    def clients(self) -> dict[str, A2AClient]:
        with self._lock:
            for name in self.names:
                if name not in self._clients:
                    try:
                        self._clients[name] = A2AClient.from_card(langgraph_card_url(self.base_url, name))
                    except Exception as exc:                 # agent not served (yet): skip it, retry next time
                        print(f"router: can't load the card of {name}: {type(exc).__name__}: {exc}")
            return dict(self._clients)

    def describe(self) -> str:
        return "\n".join(f"- {name}: {c.description}" for name, c in self.clients().items())


def parse_route(text: str, allowed: list[str]) -> list[str] | None:
    """qwen's JSON → a list of allowed agent names, or None if unusable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        agents = json.loads(match.group(0)).get("agents")
    except (ValueError, AttributeError):
        return None
    if not isinstance(agents, list):
        return None
    picked = [a for a in allowed if a in agents]         # keep the card order; drop unknown names
    return picked if (picked or agents == []) else None


def keyword_route(question: str, allowed: list[str]) -> list[str]:
    want = "paper_scout" if SCOUT_WORDS_RE.search(question) else "okf_agent"
    return [want] if want in allowed else allowed[:1]


def build_router_graph(directory: AgentDirectory, llm, log_conn=None, agent_name: str = ROUTER_NAME) -> StateGraph:

    def route(state: RouterState) -> dict:
        t0 = time.perf_counter()
        question = text_of(state["messages"][-1]).strip()
        allowed = list(directory.clients())
        last = state.get("last_route")
        llm_info = {"model": llm.name}
        prev_q, _ = previous_turn(state["messages"])
        if GREETING_RE.match(question):
            chosen, by = [], "rule-greeting"
        else:
            message = question
            if last is not None and prev_q:
                message = (f"Previous message (answered by {', '.join(last) or 'nobody'}): {prev_q}\n"
                           f"New message: {question}")
            reply = llm.complete(ROUTE_PROMPT.format(agents=directory.describe()), message)
            llm_info["route_tokens"] = reply.answer_tokens
            parsed = parse_route(reply.text, allowed)
            if parsed is not None:
                chosen, by = parsed, "llm"
            elif last and (looks_like_followup(question) or len(question.split()) <= SHORT_FOLLOWUP):
                chosen, by = [a for a in last if a in allowed], "rule-followup"
            else:
                chosen, by = keyword_route(question, allowed), "rule-keywords"
        followup = bool(last) and chosen == last and by != "rule-greeting"
        return {"question": question, "route": chosen, "route_by": by, "followup": followup,
                "search_query": f"-> {', '.join(chosen) or 'direct'} ({by})",
                "replies": None, "bad_citations": [], "sources": [], "passages": [], "log_id": None, "llm": llm_info,
                "timings": {"started": time.time(), "route_ms": round((time.perf_counter() - t0) * 1000, 1)}}

    def after_route(state: RouterState) -> list[str]:
        """The conditional edge. Returning SEVERAL node names runs them in parallel."""
        return [f"ask_{a}" for a in state["route"]] or ["direct"]

    def make_ask(agent: str):
        def ask(state: RouterState) -> dict:
            t0 = time.perf_counter()
            client = directory.clients()[agent]
            try:
                # Continue the agent's conversation only for a follow-up of its last turn. A new request
                # starts a fresh one, so the agent's own follow-up rule can't glue it to an old question.
                context_id = state.get("contexts", {}).get(agent) if state["followup"] else None
                r = client.send(state["question"], context_id=context_id)
                reply = {"text": r.text if r.ok else f"{agent} couldn't answer: {r.text}", "state": r.state}
                contexts = {agent: r.context_id}
            except Exception as exc:
                reply = {"text": f"{agent} couldn't be reached ({type(exc).__name__}: {exc})", "state": "ERROR"}
                contexts = {}
            reply["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return {"replies": {agent: reply}, "contexts": contexts}
        ask.__name__ = f"ask_{agent}"
        return ask

    def direct(state: RouterState) -> dict:
        lines = [f"- **{name}**: {c.description}" for name, c in directory.clients().items()]
        if GREETING_RE.match(state["question"]):
            intro = "Hi! I can help with two kinds of questions:"
        else:
            intro = "That's outside what I can do. I can help with:"
        return {"answer": intro + "\n" + "\n".join(lines) + "\n\nAsk a question about the papers, or ask me to find papers on a topic."}

    def combine(state: RouterState) -> dict:
        replies = state.get("replies") or {}
        if not state["route"]:
            text = state["answer"]
        elif len(state["route"]) == 1:
            agent = state["route"][0]
            text = replies.get(agent, {}).get("text", "")
        else:
            text = "\n\n".join(f"### {HEADINGS.get(a, a)} ({a})\n\n{replies.get(a, {}).get('text', '')}"
                               for a in state["route"])
        failed = [a for a in state["route"] if not replies.get(a, {}).get("state", "").endswith("COMPLETED")]
        timings = {**state["timings"], **{f"{a}_ms": r["ms"] for a, r in replies.items()}}
        reply = AIMessage(content=text, additional_kwargs={"agent": agent_name, "route": state["route"]})
        return {"answer": text, "refused": bool(failed), "last_route": state["route"], "timings": timings,
                "messages": [reply]}

    def log(state: RouterState, config) -> dict:
        timings = {**state["timings"], "total_ms": round((time.time() - state["timings"]["started"]) * 1000, 1)}
        log_id = None
        if log_conn is not None:
            try:
                thread_id = config["configurable"].get("thread_id", "default")
                log_id = insert_qa_log(log_conn, thread_id, agent_name, {**state, "timings": timings})
            except Exception as exc:
                print(f"qa_log insert failed: {type(exc).__name__}: {exc}")
        return {"timings": timings, "log_id": log_id}

    g = StateGraph(RouterState)
    g.add_node("route", route)
    ask_nodes = [f"ask_{a}" for a in directory.names]
    for agent in directory.names:
        g.add_node(f"ask_{agent}", make_ask(agent))
    g.add_node("direct", direct)
    g.add_node("combine", combine)
    g.add_node("log", log)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ask_nodes + ["direct"])
    for node in ask_nodes + ["direct"]:
        g.add_edge(node, "combine")                     # combine waits until every branch that ran is done
    g.add_edge("combine", "log")
    g.add_edge("log", END)
    return g


def default_directory() -> AgentDirectory:
    base = os.getenv("OKF_AGENT_URL", "http://127.0.0.1:2024")
    names = [n.strip() for n in os.getenv("OKF_ROUTER_AGENTS", "okf_agent,paper_scout").split(",") if n.strip()]
    return AgentDirectory(base, names)


__all__ = ["build_router_graph", "AgentDirectory", "default_directory", "ROUTER_NAME", "RouterState",
           "parse_route", "keyword_route"]
