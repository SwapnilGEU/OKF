"""The report agent: turns a conversation into a Markdown report file, through the filesystem MCP server.

    START → prepare ─┬─(conversation given)──► summarize → save ─┐
                     ├─("list my reports")───► list_reports ────┼─► respond → log → END
                     └─(nothing to report on)───────────────────┘

prepare       split the incoming message into the request and the conversation (the router sends both)
summarize     qwen writes a title and 3-5 summary bullets — the ONLY model call. The answers themselves go
              into the report word for word, so the report can't misquote them.
save          builds the Markdown in code and writes it with the filesystem server's `write_file`
list_reports  the filesystem server's `list_directory`
respond       the reply: where the file is + a preview

The filesystem server is the official MCP reference server (Node.js, run with npx). It only allows the one
folder it's started with (reports/). A path outside it is refused by the SERVER, not just by our code.

Message format (what the router sends):
    Write a report on what we discussed
    === CONVERSATION ===
    Q: What score did GPT-4 get on the bar exam?
    A: GPT-4 scored 298/400 ... [topics/simulated-exams]
    ---
    Q: ...
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .db import insert_qa_log
from .graph import text_of
from .state import AgentState

REPORT_NAME = "report_agent"
SEPARATOR = "=== CONVERSATION ==="
LIST_RE = re.compile(r"\b(list|show|which|what)\b[^.?!]{0,25}\breports\b", re.I)   # plural: "list my reports"
ARXIV_LINK_RE = re.compile(r"^- (.+?): (https://arxiv\.org/abs/\S+)\s*$", re.M)
CITE_RE = re.compile(r"\[([^\[\]]+)\]")

SUMMARY_PROMPT = """You write the top of a short research report about a conversation.
Reply in exactly this format, nothing else:
TITLE: <a specific title, at most 10 words>
- <summary bullet 1>
- <summary bullet 2>
- <summary bullet 3>
Write 3 to 5 bullets. Use only facts from the conversation, keep each bullet to one sentence, and keep the
numbers exactly as written. Do not add citations."""


class ReportState(AgentState, total=False):
    request: str
    transcript: list[dict]          # [{"q": ..., "a": ...}]
    action: str                     # write / list / empty
    title: str
    summary: list[str]
    path: str | None
    markdown: str
    file_result: str


# --- plain helpers -------------------------------------------------------------------------------
def split_message(text: str) -> tuple[str, list[dict]]:
    """'request \\n=== CONVERSATION ===\\n Q: ..\\n A: ..\\n --- ...' → (request, [{"q", "a"}, ...])."""
    request, _, conv = text.partition(SEPARATOR)
    turns = []
    for block in re.split(r"\n-{3,}\n", conv.strip()):
        m = re.match(r"\s*Q:\s*(.*?)\nA:\s*(.*)", block, re.S)
        if m:
            turns.append({"q": m.group(1).strip(), "a": m.group(2).strip()})
    return request.strip(), turns


def format_conversation(turns: list[dict]) -> str:
    """The reverse of split_message (the router uses it)."""
    return "\n---\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in turns)


def parse_summary(text: str, fallback_title: str) -> tuple[str, list[str]]:
    m = re.search(r"TITLE:\s*(.+)", text or "")
    title = (m.group(1).strip().strip('"*# ') if m else "") or fallback_title
    bullets = [b.strip() for b in re.findall(r"^\s*[-*•]\s+(.+)$", text or "", re.M)][:5]
    return title[:120], bullets


NUMBER_RE = re.compile(r"\d+(?:[.,/:\-]\d+)*%?")


def check_numbers(bullets: list[str], source_text: str) -> tuple[list[str], list[str]]:
    """Keep summary bullets whose numbers all appear in the conversation. qwen writes the summary, and a
    4B model can garble a number ("213/400" → "213/40-0"); a wrong number in a report is worse than none."""
    kept, dropped = [], []
    for b in bullets:
        (kept if all(n in source_text for n in NUMBER_RE.findall(b)) else dropped).append(b)
    return kept, dropped


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "report"


def build_markdown(title: str, summary: list[str], turns: list[dict], when: datetime) -> str:
    """The report, assembled by code: summary from qwen, everything else copied from the conversation."""
    links, cited = {}, []
    for t in turns:
        for name, url in ARXIV_LINK_RE.findall(t["a"]):
            links[url] = name
        for group in CITE_RE.findall(t["a"]):
            for c in re.split(r"[,;]", group):
                c = c.strip()
                if "/" in c and not c.startswith("arXiv:") and c not in cited:
                    cited.append(c)
    lines = [f"# {title}", "", f"*Generated {when:%Y-%m-%d %H:%M} by report_agent from {len(turns)} question(s).*", ""]
    if summary:
        lines += ["## Summary", "", *[f"- {b}" for b in summary], ""]
    lines += ["## Questions and answers", ""]
    for i, t in enumerate(turns, 1):
        lines += [f"### {i}. {t['q']}", "", t["a"], ""]
    if cited or links:
        lines += ["## Sources", ""]
        if cited:
            lines += ["From the OKF bundle (concept ids):", "", *[f"- `{c}`" for c in cited], ""]
        if links:
            lines += ["From arXiv:", "", *[f"- [{name}]({url})" for url, name in links.items()], ""]
    return "\n".join(lines).rstrip() + "\n"


# --- the graph ------------------------------------------------------------------------------------
def build_report_graph(files, llm, reports_dir: str, log_conn=None, agent_name: str = REPORT_NAME) -> StateGraph:
    """`files` is a PersistentMCP for the filesystem server, allowed to write only in `reports_dir`."""
    reports_dir = str(Path(reports_dir).resolve())

    def prepare(state: ReportState) -> dict:
        text = text_of(state["messages"][-1])
        request, turns = split_message(text)
        if LIST_RE.search(request):
            action = "list"
        else:
            action = "write" if turns else "empty"
        return {"question": request or text[:200], "request": request, "transcript": turns, "action": action,
                "path": None, "file_result": "", "summary": [], "bad_citations": [], "sources": [], "passages": [],
                "followup": False, "log_id": None, "llm": {"model": llm.name},
                "timings": {"started": time.time()}}

    def after_prepare(state: ReportState) -> str:
        return {"write": "summarize", "list": "list_reports"}.get(state["action"], "respond")

    def summarize(state: ReportState) -> dict:
        t0 = time.perf_counter()
        convo = "\n\n".join(f"Q: {t['q']}\nA: {t['a'][:700]}" for t in state["transcript"])  # trimmed: ~1.5k tokens
        reply = llm.complete(SUMMARY_PROMPT, f"Request: {state['request']}\n\nConversation:\n\n{convo}")
        title, bullets = parse_summary(reply.text, fallback_title=state["transcript"][0]["q"][:80])
        source = "\n".join(t["q"] + "\n" + t["a"] for t in state["transcript"])
        bullets, dropped = check_numbers(bullets, source)
        title = title if all(n in source for n in NUMBER_RE.findall(title)) else state["transcript"][0]["q"][:80]
        return {"title": title, "summary": bullets, "bad_citations": [f"summary: {b}" for b in dropped],
                "llm": {"model": llm.name, "prompt_tokens": reply.prompt_tokens, "answer_tokens": reply.answer_tokens},
                "timings": {**state["timings"], "generation_ms": round((time.perf_counter() - t0) * 1000, 1)}}

    def save(state: ReportState) -> dict:
        now = datetime.now()
        md = build_markdown(state["title"], state["summary"], state["transcript"], now)
        path = str(Path(reports_dir) / f"{now:%Y-%m-%d_%H%M%S}_{slugify(state['title'])}.md")
        t0 = time.perf_counter()
        try:
            result = " ".join(str(r) for r in files.call("write_file", {"path": path, "content": md}))
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
        ok = result.startswith("Successfully wrote")
        return {"markdown": md, "path": path if ok else None, "file_result": result, "search_query": path,
                "timings": {**state["timings"], "write_ms": round((time.perf_counter() - t0) * 1000, 1)}}

    def list_reports(state: ReportState) -> dict:
        try:
            result = " ".join(str(r) for r in files.call("list_directory", {"path": reports_dir}))
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
        return {"file_result": result, "search_query": f"list {reports_dir}"}

    def respond(state: ReportState) -> dict:
        action = state["action"]
        if action == "empty":
            text = ("There's nothing to put in a report yet. Ask some questions first (about the papers, or "
                    "to find papers), then ask me for a report of the conversation.")
        elif action == "list":
            names = sorted(re.findall(r"\[FILE\]\s*(\S+\.md)", state["file_result"]), reverse=True)
            text = (f"Reports in `{reports_dir}`:\n\n" + "\n".join(f"- {n}" for n in names)) if names else \
                   f"No reports yet in `{reports_dir}`." if "[FILE]" in state["file_result"] or not state["file_result"] \
                   else f"Couldn't list the reports: {state['file_result']}"
        elif state.get("path"):
            preview = "\n".join(state["markdown"].splitlines()[:12])
            text = (f"Saved the report ({len(state['transcript'])} question(s)) to:\n`{state['path']}`\n\n"
                    f"Preview:\n\n{preview}\n…")
        else:
            text = f"I wrote the report but couldn't save it: {state['file_result']}"
        refused = action == "empty" or (action == "write" and not state.get("path"))   # a real failure only
        return {"answer": text, "refused": refused,
                "messages": [AIMessage(content=text, additional_kwargs={"agent": agent_name, "path": state.get("path")})]}

    def log(state: ReportState, config) -> dict:
        timings = {**state["timings"], "total_ms": round((time.time() - state["timings"]["started"]) * 1000, 1)}
        log_id = None
        if log_conn is not None:
            try:
                thread_id = config["configurable"].get("thread_id", "default")
                log_id = insert_qa_log(log_conn, thread_id, agent_name, {**state, "timings": timings})
            except Exception as exc:
                print(f"qa_log insert failed: {type(exc).__name__}: {exc}")
        return {"timings": timings, "log_id": log_id}

    g = StateGraph(ReportState)
    for name, fn in [("prepare", prepare), ("summarize", summarize), ("save", save), ("list_reports", list_reports),
                     ("respond", respond), ("log", log)]:
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", after_prepare, ["summarize", "list_reports", "respond"])
    g.add_edge("summarize", "save")
    g.add_edge("save", "respond")
    g.add_edge("list_reports", "respond")
    g.add_edge("respond", "log")
    g.add_edge("log", END)
    return g


__all__ = ["check_numbers", "build_report_graph", "REPORT_NAME", "SEPARATOR", "split_message", "format_conversation",
           "build_markdown", "parse_summary", "slugify"]
