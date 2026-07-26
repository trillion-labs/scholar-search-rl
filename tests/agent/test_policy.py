import asyncio
import json

from s2cs.agent.policy import make_openai_policy
from s2cs.agent.trajectory import Turn
from tests.agent.conftest import make_fake_resp


def some_tool(query: str, limit: int = 10) -> list[dict]:
    """A search tool."""
    return [{"q": query}]


def submit_answer(answer: str):
    """Submit the final answer."""
    return {"answer": answer}


def test_policy_extracts_tool_call(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        tool_name="some_tool", tool_args={"query": "x", "limit": 3},
    )
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    step = asyncio.run(policy("hello", []))
    assert step.action == {"name": "some_tool", "arguments": {"query": "x", "limit": 3}}


def test_policy_extracts_reasoning_into_thought(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content=None, tool_name="some_tool", tool_args={"query": "x"},
        reasoning="The user asked X, so I will search for Y.",
    )
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    step = asyncio.run(policy("hello", []))
    assert "search for Y" in step.thought


def test_policy_falls_back_to_content_when_no_reasoning(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content="just thinking out loud", tool_name="some_tool", tool_args={"query": "x"},
    )
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    step = asyncio.run(policy("hello", []))
    assert step.thought == "just thinking out loud"


def test_policy_action_none_when_no_tool_call(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content="just text", tool_name=None)
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    step = asyncio.run(policy("hello", []))
    assert step.action is None


def test_policy_replays_prior_turns_in_messages(fake_openai_client):
    captured = {}

    async def capture(**kwargs):
        captured["messages"] = list(kwargs["messages"])
        return make_fake_resp(tool_name="some_tool", tool_args={"query": "y"})

    fake_openai_client.chat.completions.create.side_effect = capture
    prior = [
        Turn(thought="thought0", action={"name": "some_tool", "arguments": {"query": "x"}}, observation=[{"q": "x"}]),
    ]
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    asyncio.run(policy("hello", prior))

    msgs = captured["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = msgs[3]
    assert tool_msg["role"] == "tool"
    assert "x" in tool_msg["content"]
    assistant_msg = msgs[2]
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "some_tool"
    assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}


def test_policy_preserves_raw_assistant_message(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content="raw assistant text",
        tool_name="some_tool",
        tool_args={"query": "x"},
    )
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    step = asyncio.run(policy("hello", []))

    assert step.tool_call_id == "call_1"
    assert step.assistant_message == {
        "role": "assistant",
        "content": "raw assistant text",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "some_tool", "arguments": '{"query": "x"}'},
        }],
    }


def test_policy_replays_raw_assistant_message_for_concat_parenting(fake_openai_client):
    captured = {}

    async def capture(**kwargs):
        captured["messages"] = list(kwargs["messages"])
        return make_fake_resp(tool_name="some_tool", tool_args={"query": "y"})

    raw_assistant = {
        "role": "assistant",
        "content": "exact proxy output",
        "tool_calls": [{
            "id": "real_call_id",
            "type": "function",
            "function": {"name": "some_tool", "arguments": '{"query": "x"}'},
        }],
    }
    fake_openai_client.chat.completions.create.side_effect = capture
    prior = [
        Turn(
            thought="normalized thought",
            action={"name": "some_tool", "arguments": {"query": "x"}},
            observation=[{"q": "x"}],
            assistant_message=raw_assistant,
        ),
    ]
    policy = make_openai_policy(fake_openai_client, "m", {"some_tool": some_tool})
    asyncio.run(policy("hello", prior))

    msgs = captured["messages"]
    assert msgs[2] == raw_assistant
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "real_call_id"
