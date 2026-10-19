# `notebooks/`: the build, one step per notebook

Each notebook explains one step and runs it. Read them in order to see how the project was built and why each decision was made.

**Phase 1: knowledge and retrieval** (which method answers best?)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_Chunking` | PDFs → Markdown with MinerU, then chunks for vector search |
| 02 | `02_OKF_query` | Question answering over the OKF bundle (keywords + links) |
| 03 | `03_knowlegde_graph` | Entity and relation extraction with qwen3 4B: the knowledge graph |
| 04 | `04_graph_query` | Question answering over the graph |
| 05 | `05_graph_communities` | Graph communities + a report per community (GraphRAG) |
| 06 | `06_graph_query_hybrid` | Graph + vector search |
| 07 | `07_okf_hybrid_query` | OKF + vector search: **the chosen method** |
| 08 | `08_evaluate_methods` | All four methods on 30 questions, judged with DeepEval |

**Phase 2: the application**

| # | Notebook | What it does |
|---|---|---|
| 09 | `09_okf_package` | The `okf` package (notebook 07 as a library) |
| 10 | `10_mcp_server` | The OKF MCP server |
| 11 | `11_langgraph_agent` | okf_agent as a LangGraph graph with Postgres memory |
| 12 | `12_a2a_service` | Serving it with `langgraph dev` and talking A2A |
| 13 | `13_paper_scout` | The arXiv agent (third-party MCP server) |
| 14 | `14_router` | The router: routing, parallel agents, follow-ups |
| 15 | `15_report_agent` | Reports through the filesystem MCP server (stdio) |
| 16 | `16_guardrails` | Guardrails AI: input, output and untrusted-text guards, measured |
| 17 | `17_chainlit_ui` | The chat UI and token streaming |

**Running them:** open the project folder in VS Code and select the `okf` kernel. Each notebook switches to the project root itself, so relative paths work wherever Jupyter starts it. Notebooks 12–17 need the backend running (`start_backend.bat`). Run one notebook at a time: one local model serves one request at a time.
