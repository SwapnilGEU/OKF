# `okf/`: the retrieval and answering library

The core of the project: load the OKF bundle, find the right sections for a question, and let the model answer from them with citations. Everything else (MCP server, agents, UI) builds on this package.

```python
from okf import OKF

okf = OKF.load()
answer = okf.ask("What score did GPT-4 get on the bar exam?")
print(answer.text)          # "... 298/400 ... [topics/simulated-exams]"
print(answer.sources)       # the sections it cited
```

```bash
python -m okf "How does HELM measure calibration?"
python -m okf --search "bar exam"        # retrieval only, no model call
python -m okf --status                   # what's loaded and connected
```

| File | Role |
|---|---|
| `config.py` | Settings: the evaluated defaults, overridable with `OKF_<NAME>` in `.env` |
| `bundle.py` | Reads `OKF_Bundle/`: one concept per Markdown file, plus its links |
| `search.py` | Keyword (BM25) search over the concepts |
| `vectors.py` | Qdrant hybrid search: dense + sparse + ColBERT re-ranking |
| `chunkmap.py` | Maps vector-search chunks back to bundle sections (cached in `artifacts/cache/`) |
| `retrieve.py` | Combines keywords, vectors and links into the final context (18,000-character budget) |
| `llm.py` | The answering model: Ollama or any OpenAI-compatible API; `complete()` and `stream()` |
| `pipeline.py` | `OKF.load()`, `ask()`, `search()`, `get_concept()`; citation and refusal checks |
| `schemas.py` | Pydantic models: `Answer`, `Passage`, `Source` |
| `__main__.py` | The command line above |

Switch the model without code changes: `OKF_LLM_PROVIDER=openai`, `OKF_LLM_BASE_URL`, `OKF_LLM_MODEL`, `OKF_LLM_API_KEY`.
