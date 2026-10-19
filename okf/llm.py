"""The answering model, behind one small interface so it can be swapped.

- OllamaLLM: local (the evaluated setup: qwen3 4B, temperature 0, num_ctx 8192)
- OpenAICompatLLM: any OpenAI-compatible API (Mercury, OpenRouter, ...) for when the app goes public
  and there's no GPU on the server.

Both return an LLMReply. Everything above this module calls `complete(system, user)`, or
`stream(system, user)` to get the answer token by token (the chat UI shows it as it's written).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
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


class LLMStream:
    """The answer as it's generated: iterate it for text pieces; afterwards `.reply` holds the LLMReply
    (full text, token counts, time), exactly what complete() would have returned.

    qwen3 may open with <think>...</think>: those pieces are held back, never yielded, so a streamed
    answer and complete()'s answer are the same text."""

    def __init__(self, pieces: Iterator[tuple[str, dict]]):
        self._pieces = pieces            # (text delta, final stats or {}) from the provider
        self._start = time.perf_counter()
        self.reply: LLMReply | None = None

    def __iter__(self) -> Iterator[str]:
        raw, sent, stats = "", 0, {}
        for delta, final in self._pieces:
            raw += delta
            stats = final or stats
            head = raw.lstrip()
            if not head or "<think>".startswith(head):
                continue                                  # nothing yet, or maybe the start of "<think>"
            if head.startswith("<think>"):
                if "</think>" not in head:
                    continue                              # still thinking: show nothing yet
                visible = head.split("</think>", 1)[1].lstrip()
            else:
                visible = head
            if len(visible) > sent:
                yield visible[sent:]
                sent = len(visible)
        self.reply = LLMReply(_strip_thinking(raw), stats.get("prompt_tokens"), stats.get("answer_tokens"),
                              round((time.perf_counter() - self._start) * 1000, 1))


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

    def stream(self, system: str, user: str) -> LLMStream:
        """Same request with "stream": true. Ollama then sends one JSON object per line (NDJSON):
        {"message": {"content": "<a few characters>"}, "done": false} ... and a last one with "done": true
        and the token counts."""
        def pieces():
            with requests.post(
                f"{self.url}/api/chat",
                json={"model": self.s.llm_model, "stream": True,
                      "options": {"temperature": 0, "num_ctx": self.s.num_ctx, "num_predict": self.s.max_answer_tokens},
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=self.s.llm_timeout, stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    final = {"prompt_tokens": data.get("prompt_eval_count"),
                             "answer_tokens": data.get("eval_count")} if data.get("done") else {}
                    yield (data.get("message") or {}).get("content") or "", final
        return LLMStream(pieces())

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

    def stream(self, system: str, user: str) -> LLMStream:
        """OpenAI-style streaming: server-sent events, one `data: {...}` line per piece, then `data: [DONE]`."""
        def pieces():
            with requests.post(
                f"{self.url}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.llm_api_key}"},
                json={"model": self.s.llm_model, "temperature": 0, "max_tokens": self.s.max_answer_tokens,
                      "stream": True, "stream_options": {"include_usage": True},
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=self.s.llm_timeout, stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    data = json.loads(line[5:])
                    usage = data.get("usage") or {}
                    final = {"prompt_tokens": usage.get("prompt_tokens"),
                             "answer_tokens": usage.get("completion_tokens")} if usage else {}
                    delta = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    yield delta, final
        return LLMStream(pieces())

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
