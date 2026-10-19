"""okf.bundle and okf.search on a tiny bundle written to a temp folder."""

from okf.bundle import Bundle
from okf.search import BM25

PAGES = {
    "index.md": "---\nokf_version: '0.2'\n---\n# Index\n",
    "README.md": "# Not a concept\n",
    "topics/simulated-exams.md": ("---\ntype: Benchmark\ntitle: Simulated exams\naliases: [bar exam]\n---\n"
                                  "GPT-4 scored 298/400 on the bar exam. See [the section](/papers/gpt-4/4-capabilities.md).\n"),
    "papers/gpt-4/4-capabilities.md": ("---\ntype: Paper Section\ntitle: 4 Capabilities\npages: [4, 5]\n---\n"
                                       "Table 1 lists exam results.\n"),
}


def make_bundle(tmp_path):
    for rel, text in PAGES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Bundle(tmp_path)


def test_bundle_loads_concepts_and_links(tmp_path):
    bundle = make_bundle(tmp_path)
    assert set(bundle.concepts) == {"topics/simulated-exams", "papers/gpt-4/4-capabilities"}   # README skipped
    assert bundle.out_links["topics/simulated-exams"] == {"papers/gpt-4/4-capabilities"}
    assert bundle["papers/gpt-4/4-capabilities"].page_label == "pp. 4-5"


def test_bm25_finds_the_topic(tmp_path):
    scores = BM25(make_bundle(tmp_path)).scores("bar exam score")
    assert max(scores, key=scores.get) == "topics/simulated-exams"
