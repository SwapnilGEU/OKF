"""Keyword search and query helpers: BM25, question splitting, paper hints, RRF."""

from __future__ import annotations

import math
import re
from collections import Counter

from .bundle import LINK_RE, Bundle

STOPWORDS = set('''a an the of and or to in on for with by from at as is are was were be been being it its this that these
those what which who whom how why when where do does did can could should would will may might about into than then
there their they them we our you your i me my he she his her not no yes vs'''.split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class BM25:
    """Field-weighted BM25 over the whole bundle: title x3, aliases/tags/description x2, then the body."""

    def __init__(self, bundle: Bundle, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf, self.length = {}, {}
        for cid, c in bundle.concepts.items():
            body = LINK_RE.sub(r"\1", c.body.split("\n## Links")[0])
            fields = ((c.title + " ") * 3 + (" ".join(c.aliases + c.tags) + " " + c.description + " ") * 2 + body)
            tokens = tokenize(fields)
            self.tf[cid], self.length[cid] = Counter(tokens), len(tokens)
        self.avg_len = sum(self.length.values()) / len(self.length)
        df = Counter(w for tf in self.tf.values() for w in tf)
        n = len(self.tf)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str, keys=None) -> dict[str, float]:
        q = tokenize(query)
        result = {}
        for cid in (self.tf if keys is None else keys):
            tf = self.tf[cid]
            norm = 1 - self.b + self.b * self.length[cid] / self.avg_len
            s = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * norm) for w in q if w in tf)
            if s > 0:
                result[cid] = s
        return result


def rrf(*ranked_lists, k: int = 60) -> Counter:
    """Reciprocal rank fusion: each list gives an item 1/(k + rank)."""
    score = Counter()
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            score[item] += 1 / (k + rank)
    return score


PAPER_HINTS = {
    "sparks-of-agi": ["sparks of agi", "sparks of artificial general intelligence", "sparks paper"],
    "gpt-4-technical-report": ["gpt-4 technical report", "gpt-4 report", "technical report", "system card"],
    "helm": ["helm", "holistic evaluation of language models"],
    "foundation-models": ["foundation models report", "opportunities and risks of foundation models"],
    "llm-survey": ["llm survey", "survey of large language models"],
}


def _mentions(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def papers_named(question: str) -> list[str]:
    q = question.lower()
    return [p for p, hints in PAPER_HINTS.items() if any(_mentions(q, h) for h in hints)]


SPLIT_RE = re.compile(r",?\s+and\s+(?=(?:how|what|why|which|who|when|where|does|did|is|are|was|were)\b)", re.IGNORECASE)


class QuestionSplitter:
    """Split 'X..., and how/what/... Y?' into two queries, but only if each half names a topic in the wiki."""

    def __init__(self, bundle: Bundle):
        self.names = sorted({n.lower() for cid in bundle.wiki_ids
                             for n in [bundle[cid].title] + bundle[cid].aliases
                             if len(n) >= 3 and not cid.endswith("/overview")}, key=len, reverse=True)

    def names_a_topic(self, text: str) -> bool:
        text = text.lower()
        return any(_mentions(text, n) for n in self.names)

    def split(self, question: str) -> list[str]:
        parts = [p.strip(" ?,") for p in SPLIT_RE.split(question)]
        if len(parts) > 1 and all(self.names_a_topic(p) for p in parts):
            return parts
        return []
