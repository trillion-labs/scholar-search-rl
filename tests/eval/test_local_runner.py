import asyncio
from unittest.mock import AsyncMock, MagicMock

from s2cs.eval.local_runner import _strip_glm_special, make_policy, parse_glm_tool_call, parse_tool_args
from tests.agent.conftest import make_fake_resp


def _fake_client(resp=None, exc=None):
    c = MagicMock()
    c.chat = MagicMock()
    c.chat.completions = MagicMock()
    c.chat.completions.create = AsyncMock(side_effect=exc) if exc is not None else AsyncMock(return_value=resp)
    return c


def test_parse_tool_args_clean_json():
    assert parse_tool_args('{"query": "graph nets", "limit": 10}') == {"query": "graph nets", "limit": 10}


def test_parse_tool_args_empty_or_none():
    assert parse_tool_args("") == {}
    assert parse_tool_args(None) == {}
    assert parse_tool_args("{}") == {}


def test_parse_tool_args_extra_data_salvages_leading_object():
    # gpt-oss via OpenRouter has appended text after the JSON object; the leading
    # object must be recovered instead of raising JSONDecodeError("Extra data").
    raw = '{"query": "llm quantization"}\n{"stray": true}'
    assert parse_tool_args(raw) == {"query": "llm quantization"}


def test_parse_tool_args_garbage_falls_back_to_empty():
    assert parse_tool_args("not json at all") == {}
    assert parse_tool_args('{"unterminated": ') == {}


def test_parse_tool_args_non_dict_coerced_to_empty():
    assert parse_tool_args("[1, 2, 3]") == {}
    assert parse_tool_args("42") == {}


# parse_glm_tool_call — client-side recovery of GLM-style tool calls when the
# server `glm` parser doesn't return them (the </tool_call> stop trims the tag).


def test_glm_single_call_typed_args():
    # The exact XML the GLM-style chat template renders, leading <think> block.
    text = (
        "I should search.</think>\n"
        "<tool_call>search_papers\n"
        "<arg_key>query</arg_key>\n<arg_value>graph neural networks</arg_value>\n"
        "<arg_key>limit</arg_key>\n<arg_value>10</arg_value>\n"
        "</tool_call>"
    )
    call = parse_glm_tool_call(text)
    assert call == {"name": "search_papers", "arguments": {"query": "graph neural networks", "limit": 10}}
    # Non-string args are JSON-encoded by the template -> json-first coercion types them.
    assert isinstance(call["arguments"]["limit"], int)


def test_glm52_no_newline_after_name():
    # GLM-5.2 runs straight from the name into <arg_key> with no separator (unlike
    # GLM-4.5-style format, which puts a newline there) and emits no closing tag under our
    # </tool_call> stop. Verbatim capture from a glm-5.2-nvfp4 rollout.
    text = (
        "<tool_call>search_papers<arg_key>query</arg_key>"
        "<arg_value>Evolutionary Monte Carlo within-host malaria dynamics</arg_value>"
        "<arg_key>limit</arg_key><arg_value>10</arg_value>"
    )
    assert parse_glm_tool_call(text) == {
        "name": "search_papers",
        "arguments": {"query": "Evolutionary Monte Carlo within-host malaria dynamics", "limit": 10},
    }


def test_glm_array_arg():
    text = "<tool_call>search_snippets\n<arg_key>paper_ids</arg_key>\n<arg_value>[1, 2, 3]</arg_value>\n</tool_call>"
    assert parse_glm_tool_call(text)["arguments"]["paper_ids"] == [1, 2, 3]


def test_glm_submit_answer_prose_stays_string():
    text = (
        "<tool_call>submit_answer\n"
        "<arg_key>answer</arg_key>\n<arg_value>The detection rate was 95%.</arg_value>\n"
        "</tool_call>"
    )
    call = parse_glm_tool_call(text)
    assert call["name"] == "submit_answer"
    assert call["arguments"]["answer"] == "The detection rate was 95%."


def test_glm_missing_closing_tag_still_parses():
    # The </tool_call> stop is trimmed by default, so recovery must read to EOT.
    text = "<tool_call>search_papers\n<arg_key>query</arg_key>\n<arg_value>quantization</arg_value>"
    assert parse_glm_tool_call(text) == {"name": "search_papers", "arguments": {"query": "quantization"}}


def test_glm_returns_first_call_only():
    # The </tool_call> stop yields one call per turn, but be robust if two appear.
    text = (
        "<tool_call>paper_info\n<arg_key>paper_id</arg_key>\n<arg_value>42</arg_value>\n</tool_call>\n"
        "<tool_call>submit_answer\n<arg_key>answer</arg_key>\n<arg_value>done</arg_value>\n</tool_call>"
    )
    assert parse_glm_tool_call(text) == {"name": "paper_info", "arguments": {"paper_id": 42}}


def test_glm_no_tool_call_returns_none():
    assert parse_glm_tool_call("Here is a plain prose answer, no tool call.") is None
    assert parse_glm_tool_call("") is None
    assert parse_glm_tool_call(None) is None


# GLM control-token stripping + make_policy glm45 integration.


def test_strip_glm_special_tokens():
    assert _strip_glm_special('{"answer": "E"}<|user|>') == '{"answer": "E"}'
    assert _strip_glm_special("E<|observation|>") == "E"
    assert _strip_glm_special("clean text") == "clean text"
    assert _strip_glm_special(10) == 10  # non-strings pass through


def test_glm_policy_recovers_call_and_strips_leaked_token():
    # Server returned no tool_calls; the GLM call (with a leaked <|user|> eos inside
    # the answer arg) is in content -> client recovery + strip.
    resp = make_fake_resp(
        content="<tool_call>submit_answer\n<arg_key>answer</arg_key>\n<arg_value>E<|user|></arg_value>\n</tool_call>",
        tool_name=None,
    )
    policy = make_policy(_fake_client(resp=resp), "m", [], temperature=0.7, chat_format="glm45")
    step = asyncio.run(policy("q", []))
    assert step.action == {"name": "submit_answer", "arguments": {"answer": "E"}}


def test_glm_policy_degrades_on_400_instead_of_raising():
    import httpx
    import openai

    err = openai.BadRequestError(
        "input is longer than the model's context length",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions")),
        body=None,
    )
    policy = make_policy(_fake_client(exc=err), "m", [], temperature=0.7, chat_format="glm45")
    step = asyncio.run(policy("q", []))
    assert step.action is None  # bench-killing 400 swallowed; sample degrades gracefully
