import asyncio

from tests.agent.conftest import make_fake_resp

from s2cs.agent.llm import chat, chat_json


def test_chat_returns_text_and_tokens(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content="hello", pt=3, ct=4)
    result = asyncio.run(chat(fake_openai_client, "m", [{"role": "user", "content": "hi"}]))
    assert result.text == "hello"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4


def test_chat_json_success_first_try(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content='{"k": 1}')
    out = asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "go"}]))
    assert out == {"k": 1}
    assert fake_openai_client.chat.completions.create.call_count == 1


def test_chat_json_strips_code_fence(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='```json\n[{"k": 1}]\n```'
    )
    out = asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "go"}]))
    assert out == [{"k": 1}]
    assert fake_openai_client.chat.completions.create.call_count == 1


def test_chat_json_retries_then_succeeds(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="not json"),
        make_fake_resp(content='{"k": 2}'),
    ]
    out = asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "go"}], max_attempts=3))
    assert out == {"k": 2}
    assert fake_openai_client.chat.completions.create.call_count == 2


def test_chat_json_exhausts_attempts_returns_none(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="x"),
        make_fake_resp(content="y"),
        make_fake_resp(content="z"),
    ]
    out = asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "go"}], max_attempts=3))
    assert out is None
    assert fake_openai_client.chat.completions.create.call_count == 3


def test_chat_injects_extra_body_from_env(fake_openai_client, monkeypatch):
    # GLM/thinking models need chat_template_kwargs={"enable_thinking": false} or they
    # emit reasoning into a separate field and leave content empty -> JSON parse fails.
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return make_fake_resp(content="ok")

    fake_openai_client.chat.completions.create.side_effect = capture
    monkeypatch.setenv("S2CS_LLM_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    asyncio.run(chat(fake_openai_client, "m", [{"role": "user", "content": "hi"}]))
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_chat_no_extra_body_when_env_unset(fake_openai_client, monkeypatch):
    monkeypatch.delenv("S2CS_LLM_EXTRA_BODY", raising=False)
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return make_fake_resp(content="ok")

    fake_openai_client.chat.completions.create.side_effect = capture
    asyncio.run(chat(fake_openai_client, "m", [{"role": "user", "content": "hi"}]))
    assert "extra_body" not in seen


def test_chat_json_appends_fix_message_on_retry(fake_openai_client):
    seen_messages = []

    async def capture(**kwargs):
        seen_messages.append(list(kwargs["messages"]))
        if len(seen_messages) == 1:
            return make_fake_resp(content="not json")
        return make_fake_resp(content='{"ok": true}')

    fake_openai_client.chat.completions.create.side_effect = capture
    asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "go"}], max_attempts=3))
    second_call_messages = seen_messages[1]
    assert any("not valid JSON" in (m.get("content") or "") for m in second_call_messages)
