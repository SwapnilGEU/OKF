"""Settings for the OKF pipeline.

The defaults are exactly what 07_okf_hybrid_query.ipynb was evaluated with, so the package
reproduces the evaluated system out of the box.

Any setting can be overridden without touching code:
  - an environment variable OKF_<NAME> (also read from .env), e.g. OKF_N_SECTIONS=4
  - or keyword arguments: Settings.load(n_sections=4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    """The folder that holds OKF_Bundle/ (walks up from `start`, the working directory, then this file)."""
    if os.getenv("OKF_PROJECT_ROOT"):
        return Path(os.environ["OKF_PROJECT_ROOT"]).resolve()
    for base in [start, Path.cwd(), Path(__file__).resolve().parent]:
        if base is None:
            continue
        for folder in [base, *base.resolve().parents]:
            if (folder / "OKF_Bundle" / "index.md").exists():
                return folder
    raise FileNotFoundError("Can't find the project folder (the one with OKF_Bundle/index.md). "
                            "Run from inside it or set OKF_PROJECT_ROOT.")


@dataclass(frozen=True)
class Settings:
    project_root: Path

    # answering model
    llm_provider: str = "ollama"            # "ollama", or "openai" for any OpenAI-compatible API (Mercury, OpenRouter, ...)
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    llm_base_url: str = ""                  # provider "openai": e.g. https://api.inceptionlabs.ai/v1
    llm_api_key: str = ""                   # provider "openai": the key (put OKF_LLM_API_KEY in .env, never in code)
    num_ctx: int = 8192
    max_answer_tokens: int = 700
    llm_timeout: int = 600

    # context
    context_chars: int = 18000
    max_chars_per_concept: int = 4500

    # Qdrant (okf_chunks from 01_Chunking, okf_topics from 07)
    qdrant_url: str = ""                    # read from QDRANT_URL
    qdrant_api_key: str = ""                # read from QDRANT_API_KEY
    chunk_collection: str = "okf_chunks"
    topic_collection: str = "okf_topics"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "qdrant/bm25"
    late_interaction_model: str = "answerdotai/answerai-colbert-small-v1"
    prefetch_limit: int = 30

    # switches
    use_vectors: bool = True
    use_links: bool = True
    split_questions: bool = True

    # knobs
    n_vector: int = 10
    n_topics: int = 2
    n_sections: int = 5
    per_paper: int = 2
    paper_boost: float = 1.5
    rrf_k: int = 60

    @property
    def bundle_dir(self) -> Path:
        return self.project_root / "OKF_Bundle"

    @property
    def chunks_path(self) -> Path:
        return self.project_root / "data" / "processed" / "chunks.jsonl"

    @classmethod
    def load(cls, project_root: Path | str | None = None, **overrides) -> "Settings":
        root = Path(project_root).resolve() if project_root else find_project_root()
        load_dotenv(root / ".env")
        values = {"qdrant_url": os.getenv("QDRANT_URL", ""), "qdrant_api_key": os.getenv("QDRANT_API_KEY", "")}
        for f in fields(cls):
            raw = os.getenv(f"OKF_{f.name.upper()}")
            if raw is None or f.name == "project_root":
                continue
            default = f.default
            if isinstance(default, bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(project_root=root, **values)

    def with_(self, **changes) -> "Settings":
        return replace(self, **changes)

    def public(self) -> dict:
        """The settings without secrets (for logs and status endpoints)."""
        hidden = {"llm_api_key", "qdrant_api_key"}
        return {f.name: (str(getattr(self, f.name)) if f.name == "project_root" else getattr(self, f.name))
                for f in fields(self) if f.name not in hidden}
