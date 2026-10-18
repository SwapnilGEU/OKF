@echo off
REM Starts the OKF backend in three windows. Run it from your activated `unsloth` terminal,
REM in the OKF folder (the new windows inherit that environment).
REM   1. OKF MCP server      http://127.0.0.1:8000/mcp   (search_okf, ask_okf, get_concept)
REM   2. arXiv MCP server    http://127.0.0.1:8001/mcp   (third-party: arxiv-mcp-server, run by uvx)
REM   3. agent server        http://127.0.0.1:2024       (okf_router, okf_agent, paper_scout over A2A, langgraph dev)
REM      --n-jobs-per-worker 4: the router waits for the agents while they run on the same server
REM   (the filesystem MCP server for report_agent is NOT a window: the agent server starts it as a
REM    child process over stdio, with npx, the first time a report is written)
REM Postgres and Ollama run as Windows services/apps: start them first if they aren't running.
REM Close a window (or Ctrl+C in it) to stop that server.

cd /d "%~dp0"

where langgraph >nul 2>nul
if errorlevel 1 (
    echo langgraph command not found. Activate the unsloth env, then: pip install -e ".[agent]"
    exit /b 1
)
where uvx >nul 2>nul
if errorlevel 1 (
    echo uvx command not found. Activate the unsloth env, then: pip install -e ".[agent]"   ^(it installs uv^)
    exit /b 1
)

where npx >nul 2>nul
if errorlevel 1 (
    echo WARNING: npx not found - the report agent can't start the filesystem MCP server.
    echo          Install Node.js LTS ^(https://nodejs.org or: winget install OpenJS.NodeJS.LTS^), then reopen the terminal.
)

start "OKF MCP server :8000" cmd /k python -m okf_mcp.server
start "arXiv MCP server :8001" cmd /k "set TRANSPORT=http&& set PORT=8001&& uvx arxiv-mcp-server@0.8.2"
start "OKF agent server :2024" cmd /k langgraph dev --no-browser --port 2024 --n-jobs-per-worker 4

echo Started. Wait for "ready" (MCP), "Uvicorn running on http://127.0.0.1:8001" (arXiv; the first start
echo downloads it, ~1 min) and "API: http://127.0.0.1:2024" (agents).
