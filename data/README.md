# `data/`: source papers and evaluation questions

| Folder | Contents | In git |
|---|---|---|
| `raw/` | The five source PDFs (775 pages) | yes |
| `processed/markdown/` | One Markdown file per paper, parsed with MinerU (`01_Chunking`) | yes |
| `processed/clean/` | Cleaned Markdown, the input for the bundle's section split | no (rebuildable) |
| `processed/chunks.jsonl` | 2,927 chunks for vector search and the knowledge graph | no (rebuildable) |
| `eval/eval_questions.jsonl` | The 30 evaluation questions: question, reference answer, source sections, key evidence | yes |

The papers: GPT-4 Technical Report (100 pages), Sparks of AGI (155), HELM (162), A Survey of Large Language Models (144), On the Opportunities and Risks of Foundation Models (214).
