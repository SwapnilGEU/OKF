"""okf.llm.LLMStream: streamed pieces add up to complete()'s text, with qwen's <think> block hidden."""

import pytest

from okf.llm import LLMStream


def run(parts):
    stream = LLMStream((p, {"prompt_tokens": 5, "answer_tokens": 7} if i == len(parts) - 1 else {})
                       for i, p in enumerate(parts))
    pieces = list(stream)
    return pieces, stream.reply


@pytest.mark.parametrize("parts, expected", [
    (["Hello", " world"], "Hello world"),
    (["<thi", "nk>hmm", "m</th", "ink>\n\nGPT", "-4 got", " 298."], "GPT-4 got 298."),
    (["<think></think>", "Hi"], "Hi"),
    (["<", "b>bold</b> x"], "<b>bold</b> x"),
])
def test_stream_matches_final_text(parts, expected):
    pieces, reply = run(parts)
    assert "".join(pieces).strip() == expected == reply.text
    assert "<think>" not in "".join(pieces)
    assert reply.answer_tokens == 7
