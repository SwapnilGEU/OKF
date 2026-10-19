# `artifacts/`: generated files

Everything here is produced by the notebooks and can be rebuilt. Only this README, the ingestion manifest and the evaluation summaries are in git.

| Path | Made by | Contents |
|---|---|---|
| `ingestion_manifest.json` | `01_Chunking` | Pages and characters parsed per PDF |
| `extractions_*.jsonl`, `graph_entities*.csv`, `graph_facts*.jsonl` | `03_knowlegde_graph` | The knowledge graph: 2,394 entities, 2,801 facts |
| `graph_communities.jsonl` | `05_graph_communities` | 130 community reports |
| `results/*.jsonl` | `02`, `04`, `06`, `07` | Each method's answers and retrieved context for the 30 questions |
| `results/summary.csv`, `results/per_question_scores.csv` | `08_evaluate_methods` | The evaluation scores in the main README |
| `cache/chunk_to_section.json` | `okf` package | Vector-search chunk → bundle section map |
| `logs/` | `10_mcp_server` | Server logs |
