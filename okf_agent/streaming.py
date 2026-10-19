"""Token streaming through LangGraph and A2A.

How a token gets from qwen to the chat window:

    Ollama ──NDJSON──► okf_agent ──A2A SSE──► okf_router ──A2A SSE──► Chainlit
             okf.llm.stream()   relay()       A2AClient.send_stream()  relay()

LangGraph's "messages" stream mode (which the A2A endpoint uses for SendStreamingMessage) only sees
tokens from a LangChain chat model. Our qwen client is plain HTTP, and the router doesn't call a model
for the answer at all: it forwards another agent's tokens. So both use `relay()`: a tiny LangChain
chat model whose "generation" is just passing on text pieces it's given. Invoked inside a node, every
piece becomes a token event in that graph's stream, and the A2A server sends it on as an SSE event.

The id trick: LangGraph also streams every NEW message a node writes to the state (whole). The final
answer message reuses the relayed message's id, so LangGraph sees it as already streamed and doesn't
send the full text a second time. The final, guarded text still arrives in the A2A task result.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict


class TokenRelay(BaseChatModel):
    """A "chat model" that generates nothing itself: it yields the pieces of `tokens`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokens: Any                                   # an iterable of str, consumed once
    model_name: str = "relay"

    @property
    def _llm_type(self) -> str:
        return "okf-token-relay"

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs) -> Iterator[ChatGenerationChunk]:
        for piece in self.tokens:
            if not piece:
                continue
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)    # ← this is what LangGraph streams
            yield chunk

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Nobody is streaming (e.g. a plain SendMessage call): just collect the pieces.
        text = "".join(c.message.content for c in self._stream(messages, stop, run_manager))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def relay(tokens: Iterable[str], config: dict | None, name: str = "relay") -> AIMessage:
    """Pass `tokens` through as this node's streamed answer. Returns the whole message (with its id).
    `config` is the node's config: it carries LangGraph's stream callbacks."""
    return TokenRelay(tokens=tokens, model_name=name).invoke([HumanMessage(content=name)], config)


def generate(llm, system: str, user: str, config: dict | None):
    """One model call whose answer is streamed (if the LLM can stream) and returned whole.
    Returns (LLMReply, id of the streamed message or None)."""
    if not hasattr(llm, "stream"):                   # e.g. a test double with only complete()
        return llm.complete(system, user), None
    stream = llm.stream(system, user)
    message = relay(stream, config, name=llm.name)
    return stream.reply, message.id


__all__ = ["TokenRelay", "relay", "generate"]
