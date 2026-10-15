"""Hybrid retrieval: the `retrieve()` and `build_context()` of 07_okf_hybrid_query.ipynb.

1. vector:  Qdrant chunks for the question (and each part) -> mapped to sections
2. BM25:    sections and topic pages by keywords (best score over the question and its parts)
3. topics:  topic pages from Qdrant okf_topics + BM25 + backlinks from the vector sections, fused
4. links:   sections cited by the chosen topic pages
5. fuse:    RRF over the section lists, paper boost, at most PER_PAPER sections per paper
"""

from __future__ import annotations

import time
from collections import Counter

from .bundle import Bundle
from .config import Settings
from .schemas import Retrieval, SectionHit
from .search import BM25, QuestionSplitter, papers_named, rrf
from .vectors import VectorStore


class Retriever:
    def __init__(self, bundle: Bundle, settings: Settings, vectors: VectorStore | None = None,
                 section_of_chunk: dict[str, str] | None = None):
        self.bundle, self.s = bundle, settings
        self.vectors = vectors
        self.section_of_chunk = section_of_chunk or {}
        self.index = BM25(bundle)
        self.splitter = QuestionSplitter(bundle)

    @property
    def vectors_available(self) -> bool:
        return bool(self.vectors and self.vectors.ok and self.section_of_chunk)

    def retrieve(self, question: str, use_vectors: bool | None = None, use_links: bool | None = None,
                 split: bool | None = None, use_topics: bool = True) -> Retrieval:
        s, b = self.s, self.bundle
        use_vectors = (s.use_vectors if use_vectors is None else use_vectors) and self.vectors_available
        use_links = s.use_links if use_links is None else use_links
        split = s.split_questions if split is None else split
        timing = {}
        papers = papers_named(question)
        parts = self.splitter.split(question) if split else []
        queries = [question] + parts

        # 1) vector: chunks -> sections
        t = time.perf_counter()
        vector_lists = []
        if use_vectors:
            for qx in queries:
                mapped = [self.section_of_chunk[c] for c in self.vectors.search_chunks(qx, s.n_vector)
                          if c in self.section_of_chunk]
                vector_lists.append(list(dict.fromkeys(mapped)))
        timing["vector_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 2) BM25 over the bundle
        t = time.perf_counter()
        bm25 = Counter()
        for qx in queries:
            for cid, sc in self.index.scores(qx).items():
                bm25[cid] = max(bm25[cid], sc)
        bm25_sections = [cid for cid, _ in bm25.most_common() if not b.is_wiki(cid)][:30]
        bm25_topics = [cid for cid, _ in bm25.most_common() if b.is_wiki(cid)][:10]

        # 3) topic pages: Qdrant + BM25 + backlinks from the vector sections
        topics = []
        if use_topics:
            searched = ([self.vectors.search_topics(qx, 5) for qx in queries]
                        if (use_vectors and self.vectors.topics_ok) else [])
            vector_sections = [c for lst in vector_lists for c in lst[:5]]
            backlinks = (Counter(tp for c in vector_sections for tp in b.links_in(c) if b.is_wiki(tp))
                         if use_links else Counter())
            backlink_list = [tp for tp, _ in backlinks.most_common()]
            fused_topics = rrf(*searched, bm25_topics, backlink_list, k=s.rrf_k)
            topics = [cid for cid, _ in fused_topics.most_common(s.n_topics) if cid in b]

        # 4) links: sections cited by the chosen topic pages
        linked = []
        if use_links:
            cited = sorted({sec for tp in topics for sec in b.links_out(tp) if not b.is_wiki(sec)})
            linked = sorted(cited, key=lambda sec: bm25.get(sec, 0), reverse=True)[:15]
        timing["okf_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 5) fuse the section lists
        fused = rrf(*vector_lists, bm25_sections, linked, k=s.rrf_k)
        if papers:
            for cid in fused:
                if b.paper_of(cid) in papers:
                    fused[cid] *= s.paper_boost
        found = {"vector": {c for lst in vector_lists for c in lst}, "bm25": set(bm25_sections), "link": set(linked)}
        sections, per_paper = [], Counter()
        for cid, sc in fused.most_common():
            if not papers and per_paper[b.paper_of(cid)] >= s.per_paper:
                continue
            found_by = "+".join(name for name, ids in found.items() if cid in ids)
            sections.append(SectionHit(id=cid, found_by=found_by, score=round(sc * 1000, 1)))
            per_paper[b.paper_of(cid)] += 1
            if len(sections) == s.n_sections:
                break
        return Retrieval(question=question, parts=parts, papers=papers, topics=topics, sections=sections,
                         vectors_used=use_vectors, timing=timing)

    def build_context(self, r: Retrieval, budget: int | None = None) -> tuple[str, list[str], list[str]]:
        """Topic pages first, then sections, skipping any block that would overflow the budget.
        Returns (context text, concept ids used, the blocks)."""
        budget = self.s.context_chars if budget is None else budget
        blocks, used, size = [], [], 0
        for cid in r.topics + [h.id for h in r.sections]:
            block = self.bundle.render(cid, self.s.max_chars_per_concept)
            if size + len(block) > budget:
                continue
            blocks.append(block)
            used.append(cid)
            size += len(block)
        return "\n\n".join(blocks), used, blocks
