"""The answering model, behind one small interface so it can be swapped.

- OllamaLLM: local (the evaluated setup: qwen3 4B, temperature 0, num_ctx 8192)
- OpenAICompatLLM: any OpenAI-compatible API (Mercury, OpenRouter, ...) for when the app goes public
  and there's no GPU on the server.

Both return an LLMReply. Everything above this module only calls `complete(system, user)`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from .config import Settings

SYSTEM_PROMPT = '''You answer questions about five AI research papers using ONLY the knowledge below.
Each block starts with ### [concept-id]. Rules:
- Use only facts stated in the knowledge blocks. Do not add outside knowledge.
- After each claim, cite the concept id(s) you used in square brackets, e.g. [topics/calibration] or [papers/helm/4-4-calibration-and-uncertainty].
- Keep numbers, model names and percentages exactly as written.
- If the knowledge does not contain the answer, say: "The bundle does not contain this." and stop.
- Be concise: a short paragraph or a few bullets.'''


def user_message(question: str, context: str) -> str:
    return f"Knowledge:\n\n{context}\n\n---\n\nQuestion: {question}"


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int | None
    answer_tokens: int | None
    generation_ms: float


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.url = settings.ollama_url.rstrip("/")
        self.name = f"ollama/{settings.llm_model}"

    def complete(self, system: str, user: str) -> LLMReply:
        start = time.perf_counter()
        response = requests.post(
            f"{self.url}/api/chat",
            json={"model": self.s.llm_model, "stream": False,
                  "options": {"temperature": 0, "num_ctx": self.s.num_ctx, "num_predict": self.s.max_answer_tokens},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=self.s.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMReply(_strip_thinking(data["message"]["content"]), data.get("prompt_eval_count"),
                        data.get("eval_count"), round((time.perf_counter() - start) * 1000, 1))

    def health(self) -> tuple[bool, str]:
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=5).json()
        except Exception as exc:
            return False, f"Ollama not reachable at {self.url} ({type(exc).__name__})"
        if self.s.llm_model not in [m["name"] for m in tags.get("models", [])]:
            return False, f"model missing: ollama pull {self.s.llm_model}"
        return True, "ok"


class OpenAICompatLLM:
    def __init__(self, settings: Settings):
        if not settings.llm_base_url or not settings.llm_api_key:
            raise ValueError("provider 'openai' needs OKF_LLM_BASE_URL and OKF_LLM_API_KEY (in .env)")
        self.s = settings
        self.url = settings.llm_base_url.rstrip("/")
        self.name = f"{self.url.split('//')[-1].split('/')[0]}/{settings.llm_model}"

    def complete(self, system: str, user: str) -> LLMReply:
        start = time.perf_counter()
        response = requests.post(
            f"{self.url}/chat/completions",
            headers={"Authorization": f"Bearer {self.s.llm_api_key}"},
            json={"model": self.s.llm_model, "temperature": 0, "max_tokens": self.s.max_answer_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=self.s.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        text = (data["choices"][0].get("message") or {}).get("content") or ""
        return LLMReply(_strip_thinking(text), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        round((time.perf_counter() - start) * 1000, 1))

    def health(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.url}/models", headers={"Authorization": f"Bearer {self.s.llm_api_key}"}, timeout=10)
            return r.ok, "ok" if r.ok else f"HTTP {r.status_code}"
        except Exception as exc:
            return False, f"{type(exc).__name__}"


def make_llm(settings: Settings):
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings)
    if settings.llm_provider == "openai":
        return OpenAICompatLLM(settings)
    raise ValueError(f"unknown llm_provider {settings.llm_provider!r} (use 'ollama' or 'openai')")
