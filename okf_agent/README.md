# `okf_agent/`: agents, router, guardrails, streaming

Four LangGraph graphs, served together by `langgraph dev` (see `langgraph.json`) and talking to each other over A2A.

| Agent | File | Graph |
|---|---|---|
| **okf_router** | `router.py` | `guard_in → route → ask_<agent>(s) in parallel / direct → combine → log` |
| **okf_agent** | `graph.py` | `prepare → search (MCP) → answer (qwen) → check (output guard) → log` |
| **paper_scout** | `scout.py` | `plan → search arXiv (MCP) → [relax and retry] → summarize → check → log` |
| **report_agent** | `report.py` | `prepare → summarize → save (filesystem MCP) / list → respond → log` |

| Supporting file | Role |
|---|---|
| `server.py` | Builds the four graphs for `langgraph dev`; MCP tools and Postgres open lazily |
| `a2a_client.py` | A minimal A2A client: `send()` and `send_stream()` (server-sent events) |
| `mcp_tools.py` | Loads MCP tools; `PersistentMCP` keeps a stdio server (filesystem) alive |
| `guards.py` | Guardrails AI guards: input (injection, length, API keys), output (citations, numbers), untrusted text (arXiv abstracts) |
| `streaming.py` | Passes tokens through LangGraph and A2A: Ollama → agent → router → UI |
| `state.py` | The shared graph state and the `AgentReply` model |
| `db.py` | The `qa_log` table: one row per answered message |
| `agent.py` | `OKFAgent`: run okf_agent in-process with Postgres memory (used by the notebooks) |
| `config.py` | URLs, Postgres connection, reports folder, filesystem server command |
| `tracing.py` | Optional LangSmith tracing (on when `LANGSMITH_*` is in `.env`) |

**How a message flows:** the router's input guard checks the message. qwen then picks the agent(s) from their agent cards, and the router sends the message over A2A. The agent searches through MCP, qwen writes the answer (streamed), the output guard fixes citations, and both the agent and the router log to Postgres.

Guardrail settings (`.env`): `OKF_GUARD_MAX_CHARS`, `OKF_GUARD_JAILBREAK` (0/1), `OKF_GUARD_JAILBREAK_THRESHOLD`, `OKF_GUARD_DROP_UNSOURCED`.
