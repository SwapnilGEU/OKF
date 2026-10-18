"""Load MCP tools and call them from ordinary (synchronous) code.

langchain-mcp-adapters gives every MCP tool as a LangChain tool, but MCP tools are async-only.
Our graph nodes are plain functions, so `call_tool` runs the async call on a fresh event loop.
(Why plain functions: Postgres' async driver can't use Windows' default event loop, while MCP
servers started with npx/uvx need it. Sync nodes sidestep that and work the same everywhere.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from langchain_mcp_adapters.client import MultiServerMCPClient

logging.getLogger("httpx").setLevel(logging.WARNING)      # don't print a line for every MCP HTTP request


def run_sync(coro):
    """Run a coroutine to completion from sync code, even inside Jupyter (which already has a loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)                      # no loop in this thread: the simple case
    box = {}

    def worker():                                     # a loop is running here (Jupyter): use a thread
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:                  # pass the error back to the caller
            box["error"] = exc
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def load_tools(servers: dict, server_name: str | None = None) -> dict:
    """{tool name: LangChain tool} from one or more MCP servers.
    `servers` is MultiServerMCPClient's config, e.g. {"okf": {"url": ..., "transport": "streamable_http"}}."""
    client = MultiServerMCPClient(servers)
    tools = run_sync(client.get_tools(server_name=server_name))
    return {t.name: t for t in tools}


def call_tool(tool, args: dict):
    """Call an MCP tool and turn its result back into Python data (parsed JSON where possible)."""
    raw = run_sync(tool.ainvoke(args))
    return parse_tool_result(raw)


def parse_tool_result(raw):
    """MCP results arrive as text blocks [{'type': 'text', 'text': '<json>'}, ...]. Parse each one."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    if isinstance(raw, list):
        out = []
        for block in raw:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", block)
            try:
                out.append(json.loads(text))
            except (TypeError, ValueError):
                out.append(text)
        return out
    return raw
