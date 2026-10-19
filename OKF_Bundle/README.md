# `OKF_Bundle/`: the knowledge bundle

The knowledge the assistant answers from: plain Markdown files with YAML front matter, one **concept** per file. A concept's id is its path without `.md`, and that's what the answers cite.

```
OKF_Bundle/
├── index.md                 entry point: lists the papers and topics
├── log.md                   how and when the bundle was built
├── papers/<paper>/          798 section pages (one per paper section, with page numbers
│                            and parent / child / previous / next links) + an overview per paper
└── topics/                  54 curated pages (Model, Method, Benchmark, Metric, Concept)
                             that summarise what the papers say and link to the exact sections
```

**How it was built (about 30 minutes):**

1. **Section layer (no LLM):** a script split the cleaned Markdown of each paper into sections and added the links.
2. **Curated layer:** a coding agent with an OKF skill read the sections and wrote the topic pages and paper overviews. Every page lists the sections it relies on in `sources`, and is marked `status: draft` until reviewed.

Compare that with the knowledge graph, which took more than 10 hours of local extraction (see the main README).

To edit knowledge, edit a Markdown file. The vector index and the chunk → section map rebuild themselves when the bundle changes. `README.md`, `index.md` and `log.md` are not loaded as concepts.
