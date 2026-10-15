"""OKF: the whole question-answering pipeline in one object.

    from okf import OKF
    okf = OKF.load()                      # bundle, BM25, chunk map, Qdrant, LLM
    answer = okf.ask("How does HELM measure calibration?")
    print(answer.answer, answer.sources)

This is what every later stage wraps: the MCP server exposes search/ask/get_concept as tools,
the OKF agent calls them, LitServe serves them.
"""

from __future__ import annotations

import re
import time

from .bundle import Bundle
from .chunkmap import load_chunk_map
from .config import Settings
from .llm import SYSTEM_PROMPT, make_llm, user_message
from .retrieve import Retriever
from .schemas import Answer, Passage, Retrieval, Source
from .vectors import VectorStore

REFUSAL_RE = re.compile(r"\b(bundle|graph|knowledge|context)\s+(does not|doesn't|do not)\s+contain(\s+this)?\b", re.I)
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


class OKF:
    def __init__(self, settings: Settings, bundle: Bundle, retriever: Retriever, llm):
        self.settings, self.bundle, self.retriever, self.llm = settings, bundle, retriever, llm

    @classmethod
    def load(cls, settings: Settings | None = None, connect_vectors: bool = True, **overrides) -> "OKF":
        s = settings or Settings.load(**overrides)
        bundle = Bundle(s.bundle_dir)
        vectors, section_of_chunk = None, {}
        if connect_vectors and s.use_vectors:
            vectors = VectorStore(s)
            if vectors.ok:
                cache = s.project_root / "artifacts" / "cache" / "chunk_to_section.json"
                section_of_chunk = load_chunk_map(bundle, s.chunks_path, cache)
        return cls(s, bundle, Retriever(bundle, s, vectors, section_of_chunk), make_llm(s))

    # --- tools --------------------------------------------------------------------------------
    def source(self, cid: str) -> Source:
        c = self.bundle[cid]
        if c.is_section:
            return Source(id=cid, kind="section", title=c.title, paper=c.paper, pages=c.page_label,
                          label=self.bundle.cite_label(cid))
        return Source(id=cid, kind="topic", title=c.title, label=self.bundle.cite_label(cid))

    def retrieve(self, question: str, **kwargs) -> Retrieval:
        return self.retriever.retrieve(question, **kwargs)

    def search(self, question: str, **kwargs) -> list[Passage]:
        """Retrieval without the LLM: the context blocks, in order, with their sources."""
        r = self.retriever.retrieve(question, **kwargs)
        _, used, blocks = self.retriever.build_context(r)
        return [Passage(source=self.source(cid), text=block) for cid, block in zip(used, blocks)]

    def get_concept(self, cid: str) -> Passage:
        """One concept (topic page or section) in full, e.g. to follow a citation."""
        if cid not in self.bundle:
            raise KeyError(f"no concept {cid!r}")
        return Passage(source=self.source(cid), text=self.bundle.render(cid, max_chars=10**9))

    def ask(self, question: str, **kwargs) -> Answer:
        t0 = time.perf_counter()
        r = self.retriever.retrieve(question, **kwargs)
        context, used, blocks = self.retriever.build_context(r)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        reply = self.llm.complete(SYSTEM_PROMPT, user_message(question, context))
        cited = cited_ids(reply.text, used)
        return Answer(question=question, answer=reply.text, refused=is_refusal(reply.text),
                      sources=[self.source(cid) for cid in cited], used=used, retrieval_context=blocks,
                      retrieval=r, model=self.llm.name, retrieval_ms=round(retrieval_ms, 1),
                      generation_ms=reply.generation_ms, prompt_tokens=reply.prompt_tokens,
                      answer_tokens=reply.answer_tokens)

    def status(self) -> dict:
        v = self.retriever.vectors
        llm_ok, llm_msg = self.llm.health()
        return {"concepts": len(self.bundle), "topic_pages": len(self.bundle.wiki_ids),
                "sections": len(self.bundle) - len(self.bundle.wiki_ids),
                "vectors": "on" if self.retriever.vectors_available else f"off ({v.error if v else 'not connected'})",
                "topic_vectors": bool(v and v.topics_ok), "chunks_mapped": len(self.retriever.section_of_chunk),
                "llm": self.llm.name, "llm_ok": llm_ok, "llm_status": llm_msg}


def is_refusal(answer: str) -> bool:
    return REFUSAL_RE.search(answer) is not None


def cited_ids(answer: str, used: list[str]) -> list[str]:
    """Concept ids the answer cites, in context order, counting only ones that were in the context.
    Handles [a][b] and [a, b] / [a; b]."""
    cited = set()
    for group in CITATION_RE.findall(answer):
        cited.update(part.strip() for part in re.split(r"[,;]", group))
    return [cid for cid in used if cid in cited]
