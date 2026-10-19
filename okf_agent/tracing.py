"""Optional LangSmith tracing.

Off unless .env has:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=okf-agent              (optional; default project otherwise)
    LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   (only if your account is in the EU region)

LangGraph traces every graph run and node by itself once these are set. This module adds the one
thing it can't see on its own: our qwen call goes through okf.llm (plain HTTP to Ollama), not a
LangChain chat model, so `traced_llm` wraps it as an "llm" run with the prompt, the answer and the
token counts. All tracing code lives here, so switching to another tracer (e.g. Langfuse) means
changing this file only.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from okf.config import find_project_root

try:                                                   # load .env before anything reads LANGSMITH_* variables
    load_dotenv(find_project_root() / ".env")
except FileNotFoundError:
    pass

try:
    import langsmith
    from langsmith import traceable
    from langsmith.utils import get_env_var
    get_env_var.cache_clear()                          # langsmith caches env lookups: forget any read before .env loaded
except ImportError:                                    # langsmith missing: tracing simply stays off
    langsmith = traceable = None


def tracing_enabled() -> bool:
    on = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() in ("1", "true", "yes")
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return bool(traceable) and on and has_key


def tracing_status() -> dict:
    return {"enabled": tracing_enabled(),
            "project": os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "default")),
            "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            "key_set": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))}


class TracedLLM:
    """Same interface as okf.llm's OllamaLLM / OpenAICompatLLM (`.name`, `.complete()`, `.health()`),
    but every complete() call shows up in LangSmith as an LLM run."""

    def __init__(self, llm):
        self._llm = llm
        self.name = llm.name
        provider, _, model = llm.name.partition("/")
        self._traced = traceable(run_type="llm", name=llm.name,
                                 metadata={"ls_provider": provider, "ls_model_name": model})(self._call)

    def _call(self, messages: list[dict]) -> dict:
        reply = self._llm.complete(messages[0]["content"], messages[1]["content"])
        self._last = reply
        return {"choices": [{"message": {"role": "assistant", "content": reply.text}}],
                "usage_metadata": {"input_tokens": reply.prompt_tokens or 0, "output_tokens": reply.answer_tokens or 0,
                                   "total_tokens": (reply.prompt_tokens or 0) + (reply.answer_tokens or 0)},
                "generation_ms": reply.generation_ms}

    def complete(self, system: str, user: str):
        self._traced([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return self._last

    def stream(self, system: str, user: str):
        # Streamed answers go through okf_agent.streaming.relay, a LangChain chat model, which LangSmith
        # already traces as an LLM run (with the text). So no extra wrapping here.
        return self._llm.stream(system, user)

    def health(self):
        return self._llm.health()


def traced_llm(llm):
    """Wrap the LLM for tracing when LangSmith is on; otherwise return it unchanged."""
    return TracedLLM(llm) if tracing_enabled() else llm


def flush():
    """Send any traces still waiting in the background (call before the program exits)."""
    if tracing_enabled():
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers
            wait_for_all_tracers()
        except Exception:
            pass
