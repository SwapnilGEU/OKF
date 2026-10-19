# `ui/`: the Chainlit chat UI

A chat page at **http://localhost:8501** that talks to `okf_router` over A2A, like any other client.

- The answer **streams** as the model writes it. The guarded final text replaces the draft at the end.
- **Used okf_router** expands to show the route and why it was chosen, each agent's search, timings, what the model read, and what the guardrails did.
- A **Sources** box under each answer; a download card for reports; a 🛡️ badge when a message was blocked.

```bash
cd ui
uvx --python 3.12 --from chainlit==2.12.0 chainlit run app.py --port 8501
```

(`start_backend.bat` does this for you.)

- It runs in **its own environment** through `uvx`: Chainlit's FastAPI needs a newer OpenTelemetry than `langgraph-api` allows, so the two can't share one env. The UI doesn't import any project code; it only speaks HTTP.
- **`--python 3.12`** is needed: on Python 3.14, Chainlit's `nest_asyncio` breaks anyio and the page stays blank.
- **Port 8501**, because 8000 is the OKF MCP server.

| File | Role |
|---|---|
| `app.py` | The app: streaming A2A client, steps, sources, guardrail badges |
| `chainlit.md` | The Readme page inside the UI |
| `.chainlit/config.toml` | UI name and layout |
