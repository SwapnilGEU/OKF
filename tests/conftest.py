"""Shared test setup. Everything here runs offline: no Ollama, Qdrant, Postgres, arXiv or MCP servers.
The model is a fake that returns fixed text, so the tests check OUR code (routing, guards, streaming,
parsing), not qwen."""

import json
import os

os.environ.setdefault("OKF_GUARD_JAILBREAK", "0")     # no ML model download in CI
os.environ.setdefault("LANGSMITH_TRACING", "false")

import pytest

from okf.llm import LLMReply, LLMStream


class FakeLLM:
    """Stands in for qwen. Answers by looking at the system prompt, like the real agents would ask."""

    name = "fake/qwen"

    def __init__(self, answer: str | None = None):
        self.answer = answer or ("<think>checking</think>GPT-4 scored 298/400 on the bar exam [topics/simulated-exams]. "
                                 "GPT-3.5 scored 213/400 [papers/made/up].")

    def _text(self, system: str, user: str) -> str:
        if "route a user's message" in system:
            u = user.lower().split("new message:")[-1]
            if "and are there" in u:
                return json.dumps({"agents": ["okf_agent", "paper_scout"],
                                   "parts": {"okf_agent": "What does the bundle say about RLHF?",
                                             "paper_scout": "Find newer papers on RLHF"}})
            return json.dumps({"agents": ["paper_scout"] if "papers" in u else ["okf_agent"]})
        if "paper scout" in system:
            return "- Mixtral of Experts [arXiv:2401.04088] (2024): a sparse mixture-of-experts model."
        if "arXiv search" in system:
            return '{"keywords": ["mixture of experts"], "sort_by": "relevance", "date_from": null}'
        return self.answer

    def complete(self, system, user):
        return LLMReply(self._text(system, user), 10, 10, 1.0)

    def stream(self, system, user):
        text = self._text(system, user)

        def pieces():
            for i in range(0, len(text), 8):
                yield text[i:i + 8], ({} if i + 8 < len(text) else {"prompt_tokens": 10, "answer_tokens": 20})
        return LLMStream(pieces())


@pytest.fixture
def fake_llm():
    return FakeLLM()


PASSAGES = [{"source": {"id": "topics/simulated-exams", "kind": "topic", "title": "Simulated exams",
                        "paper": None, "pages": None, "label": "Simulated exams (Benchmark page)"},
             "text": "### [topics/simulated-exams]\nGPT-4 scored 298/400 on the Uniform Bar Exam (top 10%). "
                     "GPT-3.5 scored 213/400."}]


@pytest.fixture
def passages():
    return [dict(p) for p in PASSAGES]
