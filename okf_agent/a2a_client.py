"""A small A2A client: how one agent talks to another over HTTP.

    from okf_agent.a2a_client import A2AClient, langgraph_card_url
    okf = A2AClient.from_card(langgraph_card_url("http://127.0.0.1:2024", "okf_agent"))
    r1 = okf.send("What score did GPT-4 get on the bar exam?")
    r2 = okf.send("And GPT-3.5?", context_id=r1.context_id)       # same conversation

A2A in three ideas:
  agent card   a JSON "business card" at a well-known URL: name, description, skills, and the URL to
               send messages to. A router reads the cards to decide which agent gets a question.
  message      JSON-RPC: POST {"method": "SendMessage", "params": {"message": {"parts": [{"text": ...}]}}}
  task         what comes back: a status (COMPLETED, FAILED, ...) and artifacts holding the answer.
               Its contextId names the conversation; send it again for a follow-up.

The client only uses the A2A protocol, so it works with any A2A agent, not just LangGraph ones.
Only langgraph_card_url() knows LangGraph's URL layout.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx


@dataclass
class A2AReply:
    text: str                  # the agent's answer (all text parts of the response artifact)
    state: str                 # TASK_STATE_COMPLETED, TASK_STATE_FAILED, ...
    context_id: str            # the conversation; pass it to send() for a follow-up
    task_id: str
    raw: dict = field(repr=False, default_factory=dict)      # the full JSON-RPC result, for inspection

    @property
    def ok(self) -> bool:
        return self.state.endswith("COMPLETED")


class A2AError(RuntimeError):
    pass


class A2AClient:
    def __init__(self, url: str, card: dict | None = None, timeout: float = 300):
        self.url, self.card = url, card or {}
        self._http = httpx.Client(timeout=timeout)       # 300 s: qwen can take a while on a long answer

    @classmethod
    def from_card(cls, card_url: str, timeout: float = 300) -> "A2AClient":
        """Read the agent card, then talk to the URL it names."""
        card = httpx.get(card_url, timeout=30).raise_for_status().json()
        if "error" in card:
            raise A2AError(f"{card_url}: {card['error'].get('message', card['error'])}")
        return cls(card["url"], card, timeout)

    @property
    def name(self) -> str:
        return self.card.get("name", self.url)

    @property
    def description(self) -> str:
        return self.card.get("description", "")

    @staticmethod
    def _payload(method: str, text: str, context_id: str | None) -> dict:
        message = {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": str(uuid.uuid4())}
        if context_id:
            message["contextId"] = context_id                 # continue that conversation
        return {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": {"message": message}}

    def _reply(self, result: dict) -> A2AReply:
        task = result.get("task", result)                     # v1.0 wraps the task in result.task
        texts = [p["text"] for a in task.get("artifacts", []) for p in a.get("parts", []) if "text" in p]
        if not texts:              # a failed task has no artifact; the reason is in its status message
            status_msg = task["status"].get("message") or {}
            texts = [p["text"] for p in status_msg.get("parts", []) if "text" in p]
        return A2AReply(text="\n".join(texts), state=task["status"]["state"],
                        context_id=task["contextId"], task_id=task["id"], raw=result)

    def send(self, text: str, context_id: str | None = None) -> A2AReply:
        """Send one user message and wait for the finished task (SendMessage blocks by default)."""
        body = self._http.post(self.url, json=self._payload("SendMessage", text, context_id)).raise_for_status().json()
        if "error" in body:
            raise A2AError(f"{self.name}: {body['error'].get('message', body['error'])}")
        return self._reply(body["result"])

    def send_stream(self, text: str, context_id: str | None = None) -> "A2AStream":
        """Like send(), but the answer arrives in pieces: iterate the result for text pieces as the agent
        writes them; afterwards `.reply` is the finished task, same as send() returns."""
        return A2AStream(self, self._payload("SendStreamingMessage", text, context_id))

    def close(self):
        self._http.close()


class A2AStream:
    """SendStreamingMessage: the server answers with server-sent events, one JSON-RPC result per
    `data:` line (LangGraph's A2A server, checked by hand):
        task             state SUBMITTED: the task id and contextId
        status-update    state WORKING, message = the answer SO FAR (the whole text, not just the new piece)
        artifact-update  the final answer (after the agent's check node: citations fixed etc.)
        status-update    state COMPLETED (or FAILED), final: true
    Iterating yields only the NEW text of each update; afterwards `.reply` is like send()'s result."""

    def __init__(self, client: A2AClient, payload: dict):
        self.client, self.payload = client, payload
        self.reply: A2AReply | None = None

    def __iter__(self) -> Iterator[str]:
        so_far, final_text, state, task_id, context_id, failed_msg = "", None, "", "", "", ""
        headers = {"Accept": "text/event-stream"}
        with self.client._http.stream("POST", self.client.url, json=self.payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue                                   # "event: message", ": heartbeat", blank lines
                body = json.loads(line[5:])
                if "error" in body:
                    raise A2AError(f"{self.client.name}: {body['error'].get('message', body['error'])}")
                result = body.get("result") or {}
                event = result.get("task", result)
                task_id = event.get("taskId") or event.get("id") or task_id
                context_id = event.get("contextId") or context_id
                kind = event.get("kind")
                if kind == "artifact-update":
                    final_text = "".join(p.get("text", "") for p in event["artifact"].get("parts", []))
                elif kind in ("status-update", "task"):
                    status = event.get("status") or {}
                    state = status.get("state", state)
                    text = "".join(p.get("text", "") for p in (status.get("message") or {}).get("parts", []))
                    if state.endswith("WORKING") and text:
                        if text.startswith(so_far):
                            yield text[len(so_far):]            # just the new piece
                        else:
                            yield "\n\n" + text                # a different message started
                        so_far = text
                    elif text:
                        failed_msg = text                      # FAILED: the reason
        self.reply = A2AReply(text=final_text if final_text is not None else (failed_msg or so_far), state=state,
                              context_id=context_id, task_id=task_id)


def langgraph_assistant_id(base_url: str, graph: str) -> str:
    """The id of the assistant `langgraph dev` created for a graph in langgraph.json."""
    resp = httpx.post(f"{base_url}/assistants/search", json={"graph_id": graph, "limit": 1}, timeout=30)
    if resp.status_code == 404:          # e.g. "Graph 'okf' not found. Expected one of: ['okf_agent']"
        raise A2AError(f"{base_url}: {resp.json().get('detail', resp.text)}")
    found = resp.raise_for_status().json()
    if not found:
        raise A2AError(f"no assistant for graph {graph!r} at {base_url}. Is it in langgraph.json?")
    return found[0]["assistant_id"]


def langgraph_card_url(base_url: str, graph: str) -> str:
    """Agent card URL of a graph served by `langgraph dev` (A2A needs the assistant's UUID, not the name)."""
    base_url = base_url.rstrip("/")
    return f"{base_url}/.well-known/agent-card.json?assistant_id={langgraph_assistant_id(base_url, graph)}"


__all__ = ["A2AClient", "A2AReply", "A2AStream", "A2AError", "langgraph_card_url", "langgraph_assistant_id"]
