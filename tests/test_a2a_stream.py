"""okf_agent.a2a_client.A2AStream: parsing LangGraph's A2A server-sent events (recorded shapes)."""

import json

import httpx

from okf_agent.a2a_client import A2AClient


def sse(*results):
    return "".join(f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': '1', 'result': r})}\n\n"
                   for r in results)


def status(state, text=None, final=False):
    msg = {"kind": "message", "role": "ROLE_AGENT", "parts": [{"kind": "text", "text": text}]} if text else None
    return {"taskId": "t", "contextId": "c", "kind": "status-update",
            "status": {"state": state, **({"message": msg} if msg else {})}, "final": final}


BODY = sse(
    {"task": {"kind": "task", "id": "t", "contextId": "c", "status": {"state": "TASK_STATE_SUBMITTED"}}},
    status("TASK_STATE_WORKING", "GPT-4 "),
    status("TASK_STATE_WORKING", "GPT-4 scored "),
    status("TASK_STATE_WORKING", "GPT-4 scored 298/400 [papers/made/up]."),
    {"taskId": "t", "contextId": "c", "kind": "artifact-update",
     "artifact": {"artifactId": "a", "parts": [{"kind": "text", "text": "GPT-4 scored 298/400."}]}},
    status("TASK_STATE_COMPLETED", final=True),
)


def test_stream_yields_new_pieces_then_final_answer():
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, text=BODY, headers={"content-type": "text/event-stream"}))
    client = A2AClient("http://agent/a2a", {"name": "okf_agent"})
    client._http = httpx.Client(transport=transport)
    stream = client.send_stream("bar exam?")
    assert list(stream) == ["GPT-4 ", "scored ", "298/400 [papers/made/up]."]
    assert stream.reply.text == "GPT-4 scored 298/400."            # the guarded final text
    assert stream.reply.ok and stream.reply.context_id == "c"
