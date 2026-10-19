"""The OKF chat UI, built with Chainlit.

    browser ──► Chainlit (this file, :8501) ──A2A──► okf_router (:2024) ──A2A──► okf_agent / paper_scout / report_agent

The UI is just another A2A client: it sends your message to the router with SendStreamingMessage, so
the answer appears as qwen writes it (okf_agent.streaming explains how tokens get through two A2A hops).
It knows nothing about LangGraph graphs, MCP or Ollama. It also doesn't import okf_agent, so it runs in
its OWN small environment (started with uvx, like the arXiv MCP server), and Chainlit's dependencies
can't clash with the backend's.

For the details under each answer (route, steps, sources, guardrails) it reads the LangGraph server's
thread state: an A2A contextId IS a LangGraph thread id, so GET /threads/<contextId>/state returns the
router's state, and the router's state holds each agent's contextId → that agent's state.

Run (start_backend.bat does this in a 4th window):
    cd ui
    uvx --python 3.12 --from chainlit==2.12.0 chainlit run app.py --port 8501
(--python 3.12: uvx would otherwise pick the newest Python, and on 3.14 Chainlit's nest_asyncio breaks
 anyio's event-loop detection: every file request fails with NoEventLoopError.)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import chainlit as cl
import httpx
from chainlit.utils import utc_now

AGENT_URL = os.getenv("OKF_AGENT_URL", "http://127.0.0.1:2024").rstrip("/")
ROUTER = os.getenv("OKF_UI_ROUTER", "okf_router")
TIMEOUT = 300                                  # qwen can take a while on a long answer

AGENTS = {"okf_agent": "OKF bundle", "paper_scout": "arXiv scout", "report_agent": "Report writer"}
BLOCKED = {"injection": "prompt injection", "too_long": "message too long", "empty": "empty message"}


# --- talking to the backend ------------------------------------------------------------------------
class Backend:
    """A2A to send messages, the LangGraph REST API to read state. One shared async HTTP client."""

    def __init__(self, base_url: str, router: str):
        self.base, self.router = base_url, router
        self._http: httpx.AsyncClient | None = None
        self._router_url: str | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=TIMEOUT)
        return self._http

    async def router_url(self) -> str:
        """The router's A2A endpoint, from its agent card (found once, then cached)."""
        if self._router_url is None:
            found = (await self.http.post(f"{self.base}/assistants/search",
                                          json={"graph_id": self.router, "limit": 1})).raise_for_status().json()
            if not found:
                raise RuntimeError(f"no {self.router!r} graph on {self.base}")
            card = (await self.http.get(f"{self.base}/.well-known/agent-card.json",
                                        params={"assistant_id": found[0]["assistant_id"]})).raise_for_status().json()
            self._router_url = card["url"]
        return self._router_url

    async def stream(self, text: str, context_id: str | None) -> AsyncIterator[tuple[str, object]]:
        """One A2A SendStreamingMessage. Yields ("partial", answer so far) while the agent writes, then
        ("done", {"text", "state", "context_id"}) with the FINAL answer (after the guardrails' fixes)."""
        message = {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": str(uuid.uuid4())}
        if context_id:
            message["contextId"] = context_id
        payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "SendStreamingMessage",
                   "params": {"message": message}}
        final, partial, state, ctx = None, "", "", context_id
        async with self.http.stream("POST", await self.router_url(), json=payload,
                                    headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue                                   # "event: message", heartbeats, blank lines
                body = json.loads(line[5:])
                if "error" in body:
                    raise RuntimeError(body["error"].get("message", body["error"]))
                event = (body.get("result") or {})
                event = event.get("task", event)
                ctx = event.get("contextId") or ctx
                if event.get("kind") == "artifact-update":    # the final answer
                    final = "".join(p.get("text", "") for p in event["artifact"].get("parts", []))
                elif event.get("kind") in ("status-update", "task"):
                    status = event.get("status") or {}
                    state = status.get("state", state)
                    text_so_far = "".join(p.get("text", "") for p in (status.get("message") or {}).get("parts", []))
                    if text_so_far:
                        partial = text_so_far
                        if state.endswith("WORKING"):
                            yield "partial", partial          # the whole answer so far, not just the new piece
        yield "done", {"text": final if final is not None else partial, "state": state, "context_id": ctx}

    async def state(self, thread_id: str | None) -> dict:
        """A thread's current state values ({} if it can't be read: the answer still shows)."""
        if not thread_id:
            return {}
        try:
            r = await self.http.get(f"{self.base}/threads/{thread_id}/state")
            return r.json().get("values") or {} if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}


BACKEND = Backend(AGENT_URL, ROUTER)


def last_ai(values: dict) -> dict:
    """The last assistant message in a state (as JSON)."""
    for m in reversed(values.get("messages") or []):
        if m.get("type") == "ai":
            return m
    return {}


def ms(timings: dict, key: str) -> str:
    v = (timings or {}).get(key)
    return f"{v / 1000:.1f} s" if isinstance(v, (int, float)) else "?"


# --- what one agent did, as a step + side elements -------------------------------------------------
def describe_agent(agent: str, values: dict, failed: bool) -> tuple[str, list]:
    """(the step's text, the elements to attach to the answer) for one agent's turn."""
    lines, elements = [], []
    timings, guard = values.get("timings") or {}, values.get("guard") or {}
    if failed:
        lines.append("⚠️ this agent failed or couldn't answer")
    if values.get("search_query"):
        lines.append(f"**search:** `{values['search_query']}`")
    lines.append(f"**time:** {ms(timings, 'total_ms')}"
                 + (f" (search {ms(timings, 'search_ms')}, qwen {ms(timings, 'generation_ms')})"
                    if "generation_ms" in timings else ""))

    sources = values.get("sources") or []
    texts = {p["source"]["id"]: p.get("text", "") for p in values.get("passages") or [] if "source" in p}
    if sources:                                   # under the answer: one line per cited source
        if agent == "paper_scout":
            listing = [f"- **[{s['id']}]** [{s.get('title', '')}]({s.get('url', '')}) ({s.get('published', '')[:4]})"
                       for s in sources]
        else:
            listing = [f"- **[{s['id']}]** {s.get('label') or s.get('title', '')}" for s in sources]
        elements.append(cl.Text(name=f"Sources · {AGENTS.get(agent, agent)}", content="\n".join(listing),
                                display="inline"))
        evidence = [f"> **[{s['id']}]**\n> {texts[s['id']][:600]}" for s in sources if texts.get(s["id"])]
        lines.append(f"**sources:** {len(sources)}" + ("\n\n**what the model read:**\n\n" + "\n\n".join(evidence)
                                                       if evidence else ""))

    if guard.get("repaired_citations"):
        lines.append("🛡️ **citations repaired:** " + ", ".join(f"`{a}` → `{b}`" for a, b in guard["repaired_citations"].items()))
    if guard.get("removed_citations"):
        lines.append("🛡️ **made-up citations removed:** " + ", ".join(f"`{c}`" for c in guard["removed_citations"]))
    if guard.get("unsupported_numbers"):
        lines.append("🛡️ **numbers not found in the sources:** " + ", ".join(guard["unsupported_numbers"]))
    if guard.get("abstracts_cleaned"):
        lines.append("🛡️ **instructions removed from abstracts:** " + ", ".join(guard["abstracts_cleaned"]))

    path = values.get("path") or last_ai(values).get("additional_kwargs", {}).get("path")
    if agent == "report_agent" and path and Path(path).exists():
        elements.append(cl.File(name=Path(path).name, path=path, display="inline"))
        lines.append(f"**file:** `{path}`")
    return "\n\n".join(lines), elements


# --- Chainlit hooks --------------------------------------------------------------------------------
@cl.set_starters
async def starters():
    return [
        cl.Starter(label="Bar exam score", message="What score did GPT-4 get on the bar exam?"),
        cl.Starter(label="Find new papers", message="Find recent papers on mixture of experts"),
        cl.Starter(label="Bundle + arXiv", message="What does the bundle say about RLHF, and are there newer papers on it?"),
        cl.Starter(label="Write a report", message="Write a report of our conversation"),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("context_id", None)        # a new chat = a new A2A conversation with the router


@cl.on_message
async def on_message(message: cl.Message):
    t0 = time.perf_counter()
    elements: list = []
    # The router step is opened by hand (not `async with`): the answer must stream BELOW it, as its own
    # message, not inside it. It shows a spinner until we close it with an end time.
    # parent_id: the step Chainlit opened for this on_message call, as `async with` would set it. Without
    # it the step lands outside this turn's group and is drawn below the answer.
    router_step = cl.Step(name="okf_router", type="run", show_input=False,
                          parent_id=cl.context.current_step.id if cl.context.current_step else None)
    router_step.start = utc_now()
    await router_step.send()
    answer = cl.Message(content="")
    streamed, reply = False, None

    async def close_router(output: str):
        router_step.output, router_step.end = output, utc_now()
        await router_step.update()

    try:
        async for kind, data in BACKEND.stream(message.content, cl.user_session.get("context_id")):
            if kind == "partial":
                await answer.stream_token(data, is_sequence=True)      # replace the text with the longer version
                streamed = True
            else:
                reply = data
    except httpx.ConnectError:
        await close_router("can't reach the agent server")
        await cl.Message(content=f"I can't reach the agent server at `{AGENT_URL}`. Is the backend running? "
                                 "(`start_backend.bat`, then wait for *API: http://127.0.0.1:2024*)").send()
        return
    except Exception as exc:
        await close_router(f"{type(exc).__name__}: {exc}")
        await cl.Message(content=f"Something went wrong: `{type(exc).__name__}: {exc}`").send()
        return
    cl.user_session.set("context_id", reply["context_id"])

    # The details, as child steps of okf_router (read from the LangGraph server's thread state).
    values = await BACKEND.state(reply["context_id"])
    kwargs = last_ai(values).get("additional_kwargs", {})
    route, failed, blocked = kwargs.get("route") or [], kwargs.get("failed") or [], kwargs.get("blocked")
    guard_in = values.get("guard") or {}
    summary = (f"blocked by guardrails ({BLOCKED.get(blocked, blocked)})" if blocked else
               f"→ {', '.join(AGENTS.get(a, a) for a in route) or 'answered directly'}"
               f"  ·  decided by {values.get('route_by', '?')}  ·  {time.perf_counter() - t0:.1f} s")

    if values.get("question") and values["question"] != message.content.strip():
        message.content = values["question"]                  # an API key was masked: show the masked text
        await message.update()

    if blocked or guard_in:                                   # what the input guard did
        async with cl.Step(name="guardrails", type="tool", show_input=False, parent_id=router_step.id) as g:
            g.output = "\n".join(f"- **{name}**: {reason}" for name, reason in guard_in.items()) or "-"

    for agent in route:                                       # one step per agent that answered
        agent_values = await BACKEND.state((values.get("contexts") or {}).get(agent))
        text, els = describe_agent(agent, agent_values, agent in failed)
        async with cl.Step(name=AGENTS.get(agent, agent), type="tool", show_input=False,
                           parent_id=router_step.id) as s:
            asked = (values.get("parts") or {}).get(agent) or values.get("question") or message.content
            s.output = f"**asked:** {asked.split(chr(10) + '=== CONVERSATION ===')[0]}\n\n{text}"
        elements += els
    await close_router(summary)

    # The final text replaces the streamed one: the output guard may have repaired or removed citations,
    # and the scout adds its links, after the model finished writing.
    answer.content = reply["text"]
    if blocked:
        answer.content = f"🛡️ **Blocked by guardrails** ({BLOCKED.get(blocked, blocked)})\n\n{answer.content}"
    answer.elements = elements
    await (answer.update() if streamed else answer.send())
