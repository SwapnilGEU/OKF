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
import sys
import tempfile
import threading
from pathlib import Path

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


class PersistentMCP:
    """One MCP session kept open for the life of the program, for servers that run over stdio.

    Over HTTP (our OKF server, the arXiv server) the server is already running, so connecting per
    call is cheap. Over stdio the server is a CHILD PROCESS that the client starts itself, and
    `load_tools` + `call_tool` would start a new one for every call (npx: seconds each, or a minute
    if npx checks the registry). So here one child process is started on first use, kept alive by
    an event loop in a background thread, and every call goes to that same session.

        files = PersistentMCP("files", {"command": "npx", "args": [...], "transport": "stdio"})
        files.tools()                       # {name: tool}
        files.call("write_file", {"path": ..., "content": ...})
    """

    def __init__(self, name: str, connection: dict):
        self.name, self.connection = name, connection
        self._loop = self._cm = None
        self._tools: dict | None = None
        self.log_path: Path | None = None              # where a stdio server's own log lines go
        self._lock = threading.Lock()

    def _new_loop(self):
        if sys.platform == "win32":                     # subprocesses need the Proactor loop on Windows
            return asyncio.ProactorEventLoop()
        return asyncio.new_event_loop()

    async def _open(self) -> dict:
        from contextlib import AsyncExitStack

        from langchain_mcp_adapters.tools import load_mcp_tools
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._cm = AsyncExitStack()
        if self.connection.get("transport") == "stdio":
            errlog = self._cm.enter_context(self._errlog)         # opened in tools(), outside the event loop
            params = StdioServerParameters(command=self.connection["command"], args=self.connection.get("args", []),
                                           env=self.connection.get("env"))
            read, write = await self._cm.enter_async_context(stdio_client(params, errlog=errlog))  # starts the child
            session = await self._cm.enter_async_context(ClientSession(read, write))
            await session.initialize()                                                             # MCP handshake
        else:
            client = MultiServerMCPClient({self.name: self.connection})
            session = await self._cm.enter_async_context(client.session(self.name))
        return {t.name: t for t in await load_mcp_tools(session)}

    def tools(self, timeout: float = 300) -> dict:
        with self._lock:
            if self._tools is None:
                if self.connection.get("transport") == "stdio":
                    # The child process writes its log lines to stderr. By default that's Python's sys.stderr,
                    # which inside Jupyter is not a real file, and Windows refuses it ("UnsupportedOperation:
                    # fileno"). So the child gets a real log file. Opened HERE, before the event loop, because
                    # file calls inside a loop are "blocking calls" that `langgraph dev` refuses.
                    self.log_path = Path(tempfile.gettempdir()) / f"okf-mcp-{self.name}.log"
                    self._errlog = open(self.log_path, "a", encoding="utf-8")
                self._loop = self._new_loop()
                threading.Thread(target=self._loop.run_forever, name=f"mcp-{self.name}", daemon=True).start()
                try:
                    self._tools = asyncio.run_coroutine_threadsafe(self._open(), self._loop).result(timeout)
                except Exception:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    self._loop = None
                    raise
            return self._tools

    def call(self, tool_name: str, args: dict, timeout: float = 120):
        tool = self.tools()[tool_name]
        try:
            raw = asyncio.run_coroutine_threadsafe(tool.ainvoke(args), self._loop).result(timeout)
        except Exception as exc:
            if "ToolException" in type(exc).__name__ or "Access denied" in str(exc):
                raise                                   # the server answered with an error: report it
            self.reset()                                # the child process died: start fresh next time
            raise
        return parse_tool_result(raw)

    def reset(self):
        """Close the session (this ends a stdio child process) and stop the background loop."""
        with self._lock:
            if self._loop is not None:
                if self._cm is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(self._cm.aclose(), self._loop).result(10)
                    except Exception:
                        pass                                # already dead: nothing to close
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = self._cm = self._tools = None
