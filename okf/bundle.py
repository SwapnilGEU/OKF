"""The OKF bundle: one markdown file = one concept, markdown links = edges.

Same rules as 02/07: the concept ID is the path without `.md`; `index.md` and `log.md` are skipped.
- out_links[A]: what A points to (a topic page points to the sections it cites)
- in_links[A]:  what points to A (for a section: the topic pages that cite it)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RESERVED = {"index.md", "log.md"}
LINK_RE = re.compile(r"\[([^\]]*)\]\((/[^)#\s]+)\.md\)")


@dataclass
class Concept:
    id: str
    type: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    paper: str | None = None
    pages: list[int] | None = None
    body: str = ""

    @property
    def is_section(self) -> bool:
        return self.type == "Paper Section"

    @property
    def page_label(self) -> str:
        first, last = self.pages or [None, None]
        return f"p. {first}" if first == last else f"pp. {first}-{last}"


class Bundle:
    def __init__(self, bundle_dir: Path):
        self.dir = Path(bundle_dir)
        if not (self.dir / "index.md").exists():
            raise FileNotFoundError(f"OKF bundle not found at {self.dir}")
        self.concepts: dict[str, Concept] = {}
        for path in sorted(self.dir.rglob("*.md")):
            if path.name in RESERVED:
                continue
            _, frontmatter, body = path.read_text(encoding="utf-8").split("---\n", 2)
            fm = yaml.safe_load(frontmatter)
            cid = path.relative_to(self.dir).with_suffix("").as_posix()
            self.concepts[cid] = Concept(id=cid, type=fm["type"], title=fm.get("title", cid),
                                         description=fm.get("description", ""), tags=fm.get("tags") or [],
                                         aliases=fm.get("aliases") or [], paper=fm.get("paper"),
                                         pages=fm.get("pages"), body=body)
        self.out_links: dict[str, set[str]] = defaultdict(set)
        self.in_links: dict[str, set[str]] = defaultdict(set)
        for cid, c in self.concepts.items():
            for _, target in LINK_RE.findall(c.body):
                t = target.lstrip("/")
                if t in self.concepts and t != cid:
                    self.out_links[cid].add(t)
                    self.in_links[t].add(cid)
        self.wiki_ids = [cid for cid in self.concepts if self.is_wiki(cid)]

    # --- lookups -------------------------------------------------------------------------------
    def __contains__(self, cid: str) -> bool:
        return cid in self.concepts

    def __getitem__(self, cid: str) -> Concept:
        return self.concepts[cid]

    def __len__(self) -> int:
        return len(self.concepts)

    def is_wiki(self, cid: str) -> bool:
        """Topic and overview pages (curated) vs paper sections (source text)."""
        return not self.concepts[cid].is_section

    def links_out(self, cid: str) -> list[str]:
        return sorted(self.out_links.get(cid, ()))     # sorted: the same order on every run

    def links_in(self, cid: str) -> list[str]:
        return sorted(self.in_links.get(cid, ()))

    @staticmethod
    def paper_of(cid: str) -> str | None:
        return cid.split("/")[1] if cid.startswith("papers/") else None

    # --- text ----------------------------------------------------------------------------------
    def clean_body(self, cid: str) -> str:
        """The body without the trailing link lists, links turned into plain text."""
        body = self.concepts[cid].body.split("\n## Links")[0].split("\n## Related topics")[0]
        return re.sub(r"\n{3,}", "\n\n", LINK_RE.sub(r"\1", body)).strip()

    def render(self, cid: str, max_chars: int = 4500) -> str:
        """One context block, as the answering model sees it: '### [concept-id] (where)' + text."""
        c = self.concepts[cid]
        body = self.clean_body(cid)[:max_chars]
        if c.is_section:
            return f"### [{cid}] ({c.paper}, {c.page_label})\n{body}"
        return f"### [{cid}] (curated {c.type} page)\n{body}"

    def cite_label(self, cid: str) -> str:
        """Human-readable source line, e.g. 'helm, 4.4 Calibration, pp. 31-32'."""
        c = self.concepts[cid]
        if c.is_section:
            return f"{c.paper}, {c.title}, {c.page_label}"
        return f"curated {c.type} page: {c.title}"

    def topic_text(self, cid: str) -> str:
        """What gets embedded for a topic page in okf_topics: title, description, aliases, summary."""
        c = self.concepts[cid]
        body = LINK_RE.sub(r"\1", c.body)
        summary = body.split("## Summary", 1)[-1].split("\n## ", 1)[0].strip() if "## Summary" in body else body[:1500]
        return f"{c.title}\n{c.description}\nAlso called: {', '.join(c.aliases)}\n\n{summary}"[:2000]
