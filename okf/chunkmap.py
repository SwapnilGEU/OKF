"""Map every Qdrant chunk (~1,000 characters, from 01_Chunking) to the OKF section that contains it.

Both were cut from the same cleaned markdown, so a chunk's text sits inside exactly one section.
We look up three short probes of the chunk inside the sections of the same paper.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .bundle import Bundle

DOC_TO_PAPER = {"AGI": "sparks-of-agi", "GPT-4": "gpt-4-technical-report", "Holistic_Eval": "helm",
                "Foundation_Models": "foundation-models", "LLMs": "llm-survey"}


def normalize(text: str) -> str:
    text = re.sub(r"\[p\. \d+\]", " ", text)                  # page markers exist only in the bundle
    text = re.sub(r"[*_`#|>\-–—]+", " ", text)                  # markdown that differs between the two
    return re.sub(r"\s+", " ", text).strip().lower()


def load_chunks(chunks_path: Path) -> list[dict]:
    with Path(chunks_path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def map_chunks(bundle: Bundle, chunks: list[dict]) -> dict[str, str]:
    """chunk_id -> section concept id (chunks outside the bundle, like front matter, are left out)."""
    section_text = {cid: normalize(c.body.split("\n## Links")[0])
                    for cid, c in bundle.concepts.items() if not bundle.is_wiki(cid)}
    sections_of_paper = defaultdict(list)
    for cid in section_text:
        sections_of_paper[cid.split("/")[1]].append(cid)

    section_of_chunk = {}
    for c in chunks:
        text = normalize(c["text"])
        for start in (0, len(text) // 3, 2 * len(text) // 3):     # 3 probes from different places
            probe = text[start:start + 60]
            if len(probe) < 20:
                continue
            hits = [cid for cid in sections_of_paper[DOC_TO_PAPER[c["document"]]] if probe in section_text[cid]]
            if hits:
                section_of_chunk[c["chunk_id"]] = min(hits, key=lambda h: len(section_text[h]))
                break
    return section_of_chunk


def load_chunk_map(bundle: Bundle, chunks_path: Path, cache_path: Path | None = None) -> dict[str, str]:
    """map_chunks with a cache file, so a server starts in a second instead of re-matching every chunk.
    The cache is rebuilt when the bundle or the chunks file is newer than it."""
    chunks_path = Path(chunks_path)
    if cache_path is not None:
        cache_path = Path(cache_path)
        newest_input = max([chunks_path.stat().st_mtime] + [p.stat().st_mtime for p in bundle.dir.rglob("*.md")])
        if cache_path.exists() and cache_path.stat().st_mtime >= newest_input:
            return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = map_chunks(bundle, load_chunks(chunks_path))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(mapping), encoding="utf-8")
    return mapping
