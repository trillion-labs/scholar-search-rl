"""Regression tests for the GLM-4.5 tool parser (patch 0008).

A GLM-style chat template emits tool calls as
`<tool_call>name\\n<arg_key>k</arg_key>\\n<arg_value>v</arg_value></tool_call>`
XML, not the hermes JSON body. `Glm45ToolParser` must extract these (it backs the
launcher's `multi_turn.format=glm45`). A stub tokenizer feeds the parser fixed
decoded text, so the test is GPU-free and needs no model.
"""

import asyncio
import json

import pytest

# Glm45ToolParser.__init__ imports sglang's Glm4MoeDetector lazily.
pytest.importorskip("sglang")

from verl.experimental.agent_loop.tool_parser import ToolParser  # noqa: E402


class _StubTokenizer:
    """Returns a fixed decoded string regardless of ids (the parser only decodes)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def decode(self, ids, *args, **kwargs) -> str:
        return self._text


def _parse(text: str):
    parser = ToolParser.get_tool_parser("glm45", _StubTokenizer(text))
    return asyncio.run(parser.extract_tool_calls([0, 1, 2]))


def test_glm45_is_registered():
    assert "glm45" in ToolParser._registry


def test_single_call_with_typed_args():
    # Leading think block + the exact XML the GLM-style chat template renders.
    text = (
        "\n<think>I should search.</think>\n"
        "<tool_call>search_papers\n"
        "<arg_key>query</arg_key>\n<arg_value>graph neural networks</arg_value>\n"
        "<arg_key>limit</arg_key>\n<arg_value>10</arg_value>\n"
        "</tool_call>"
    )
    _, calls = _parse(text)
    assert len(calls) == 1
    assert calls[0].name == "search_papers"
    args = json.loads(calls[0].arguments)
    assert args == {"query": "graph neural networks", "limit": 10}
    # Non-string args are JSON-encoded by the template, so json-first coercion
    # yields the right type without the schema.
    assert isinstance(args["limit"], int)


def test_array_arg():
    text = (
        "<tool_call>search_snippets\n"
        "<arg_key>query</arg_key>\n<arg_value>retrieval</arg_value>\n"
        "<arg_key>paper_ids</arg_key>\n<arg_value>[1, 2, 3]</arg_value>\n"
        "</tool_call>"
    )
    _, calls = _parse(text)
    args = json.loads(calls[0].arguments)
    assert args["paper_ids"] == [1, 2, 3]


def test_multiple_calls_in_one_turn():
    text = (
        "<tool_call>paper_info\n<arg_key>paper_id</arg_key>\n<arg_value>42</arg_value>\n</tool_call>\n"
        "<tool_call>submit_answer\n<arg_key>answer</arg_key>\n<arg_value>done</arg_value>\n</tool_call>"
    )
    _, calls = _parse(text)
    assert [c.name for c in calls] == ["paper_info", "submit_answer"]
    assert json.loads(calls[0].arguments) == {"paper_id": 42}
    assert json.loads(calls[1].arguments) == {"answer": "done"}


def test_no_tool_call_returns_empty():
    _, calls = _parse("\n<think>no tools needed</think>\nHere is a plain answer.")
    assert calls == []
