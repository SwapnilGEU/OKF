# `okf_mcp/`: the OKF MCP server

Exposes the `okf` library as an [MCP](https://modelcontextprotocol.io) server, so any MCP client (our agents, Claude Desktop, an IDE) can search the papers.

```bash
python -m okf_mcp.server            # streamable HTTP on http://127.0.0.1:8000/mcp
python -m okf_mcp.server --stdio    # for clients that start it as a child process
```

| Tool / resource | What it does |
|---|---|
| `search_okf(question)` | The best-matching sections, with sources. No model call, about 1–2 s |
| `ask_okf(question)` | A full cited answer (for outside clients; our agents write the answer themselves) |
| `get_concept(id)` | One section or topic page by its id |
| `okf://status` | What's loaded: concepts, vectors on/off, which model answers |

Uses MCP Python SDK 1.x (`mcp<2`), which `langchain-mcp-adapters` requires.
