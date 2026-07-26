import dataclasses
import json
import logging
import os
from typing import Any

import openai

log = logging.getLogger(__name__)


def _extra_body() -> dict | None:
    """Per-request extra body from S2CS_LLM_EXTRA_BODY (JSON), or None. Lets a launcher
    pass provider-specific knobs the OpenAI schema lacks — e.g. GLM/thinking models need
    `{"chat_template_kwargs": {"enable_thinking": false}}` or they emit reasoning into a
    separate field and leave `content` empty (JSON parse then fails)."""
    raw = os.environ.get("S2CS_LLM_EXTRA_BODY")
    return json.loads(raw) if raw else None


@dataclasses.dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


async def chat(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> ChatResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None: kwargs["max_tokens"] = max_tokens
    if tools is not None:      kwargs["tools"] = tools
    if tool_choice is not None: kwargs["tool_choice"] = tool_choice
    extra = _extra_body()
    if extra is not None:      kwargs["extra_body"] = extra
    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    usage = resp.usage  # some providers omit usage on the response
    return ChatResult(
        text=choice.message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def _strip_code_fence(text: str) -> str:
    """Models often wrap JSON in a ```json … ``` fence; strip it before parsing."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


async def chat_json(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_attempts: int = 3,
    temperature: float = 0.7,
) -> dict | list | None:
    """Call the model and parse its output as JSON. Retry up to `max_attempts` on parse failure."""
    convo = list(messages)
    for attempt in range(1, max_attempts + 1):
        result = await chat(client, model, convo, temperature=temperature)
        try:
            return json.loads(_strip_code_fence(result.text))
        except json.JSONDecodeError as exc:
            log.warning("chat_json attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt == max_attempts:
                return None
            convo = convo + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": "Your previous output was not valid JSON. Return only valid JSON, no prose."},
            ]
    return None
