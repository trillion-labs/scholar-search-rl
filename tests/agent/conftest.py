import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclasses.dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    cost: float | None = None


@dataclasses.dataclass
class FakeToolCallFn:
    name: str
    arguments: str


@dataclasses.dataclass
class FakeToolCall:
    id: str
    type: str
    function: FakeToolCallFn


@dataclasses.dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall]
    reasoning: str | None = None
    reasoning_content: str | None = None


@dataclasses.dataclass
class FakeChoice:
    message: FakeMessage


@dataclasses.dataclass
class FakeResp:
    choices: list[FakeChoice]
    usage: FakeUsage


def make_fake_resp(content: str | None = None, tool_name: str | None = None,
                   tool_args: dict | None = None, reasoning: str | None = None,
                   pt: int = 10, ct: int = 5, cost: float | None = None) -> FakeResp:
    import json
    tcs: list[FakeToolCall] = []
    if tool_name is not None:
        tcs.append(FakeToolCall(
            id="call_1", type="function",
            function=FakeToolCallFn(name=tool_name, arguments=json.dumps(tool_args or {})),
        ))
    return FakeResp(
        choices=[FakeChoice(message=FakeMessage(content=content, tool_calls=tcs, reasoning=reasoning))],
        usage=FakeUsage(prompt_tokens=pt, completion_tokens=ct, cost=cost),
    )


@pytest.fixture
def fake_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client
