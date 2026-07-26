import json
import logging
import re
from copy import deepcopy
from typing import Any, Callable

import openai

from s2cs.agent.react import PolicyStep
from s2cs.agent.tools import specs
from s2cs.agent.trajectory import Turn

log = logging.getLogger(__name__)

_FN_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_TC_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _coerce(value: str, ptype: str | None) -> Any:
    v = value.strip()
    if ptype in ("integer", "number"):
        try:
            return int(v) if ptype == "integer" else float(v)
        except ValueError:
            return v
    if ptype == "boolean":
        return v.lower() in ("true", "1", "yes")
    return v


def _parse_content_tool_call(text: str, param_types: dict[str, dict[str, str]]) -> tuple[dict | None, str]:
    """Fallback for models that emit tool calls in the message content when the
    server-side parser misses them — Hermes JSON (`<tool_call>{"name":...}</tool_call>`,
    e.g. Qwen3) and Hermes-XML (`<function=...><parameter=...>`, e.g. Qwen3.5).
    Returns (action, thought_without_call); XML parameter values are coerced per the
    tool's declared JSON-schema type (JSON arguments are already typed)."""
    m = _TC_JSON_RE.search(text)
    if m:
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name")
            args = payload.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(name, str) and isinstance(args, dict):
                thought = (text[: m.start()] + text[m.end():]).strip()
                return {"name": name, "arguments": args}, thought
        except json.JSONDecodeError:
            pass
    m = _FN_RE.search(text)
    if not m:
        return None, text
    name = m.group(1).strip()
    types = param_types.get(name, {})
    args = {k.strip(): _coerce(val, types.get(k.strip())) for k, val in _PARAM_RE.findall(m.group(2))}
    thought = (text[: m.start()] + text[m.end():]).strip()
    return {"name": name, "arguments": args}, thought

SYSTEM_PROMPT = """You are a research agent answering questions over a scientific paper corpus.

You operate in a tool-calling loop: on every turn you MUST respond by calling
exactly one of the available tools — never answer in plain text. The only way to
give your final answer is to call submit_answer.

Workflow:
- Use the search and read tools to gather evidence.
- Think briefly about each step before acting.
- When confident, call submit_answer with your final answer.
- Submitting ends the session. Do not submit until you have enough evidence.
"""

PAPER_SET_SYSTEM_PROMPT = """You are a research agent finding ALL papers in a scientific corpus that match a query's criteria.

You operate in a tool-calling loop: on every turn you MUST respond by calling
exactly one of the available tools — never answer in plain text. The only way to
deliver your result is to call submit_answer.

Workflow:
- Use the search and read tools to find every paper matching the criteria. Be
  comprehensive yet efficient — cast a wide net, then keep the papers that fit.
- Each search returns at most ~128 results. Don't try to pull everything in one
  query; instead issue several focused searches (vary the wording, split the
  criteria) and accumulate the matches you find across them.
- When done, call submit_answer with a JSON object listing the papers you judged
  relevant, ORDERED most-relevant-first (ranking is scored):
    {"results": [{"paper_id": <corpus_id>, "markdown_evidence": "<title + why it matches>"}, ...]}
  `paper_id` is the Semantic Scholar corpus_id (digits only). Put the strongest
  matches first; do not pad the list with weak or irrelevant papers.
- Submitting ends the session. Do not submit until you have searched thoroughly.
"""

NUDGE_PROMPT = (
    "You did not call any tool. You must respond by calling exactly one of the "
    "available tools — do not answer in plain text. If you already have your "
    "final answer, call submit_answer with it."
)


def _serialize_observation(obs: Any) -> str:
    import dataclasses as _dc
    def default(o: Any) -> Any:
        if _dc.is_dataclass(o):
            return _dc.asdict(o)
        return str(o)
    return json.dumps(obs, ensure_ascii=False, default=default)[:8000]


def _message_to_dict(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    raw = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        raw["tool_calls"] = [
            tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                },
            }
            for tc in tool_calls
        ]
    return {k: v for k, v in raw.items() if v is not None}


def _first_tool_call_id(message: dict[str, Any] | None) -> str | None:
    if not message:
        return None
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    return tool_calls[0].get("id")


def make_openai_policy(
    client: openai.AsyncOpenAI,
    model: str,
    tools: dict[str, Callable],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Callable:
    tool_specs = specs(tools)
    param_types = {
        s["function"]["name"]: {
            p: (spec or {}).get("type")
            for p, spec in s["function"].get("parameters", {}).get("properties", {}).items()
        }
        for s in tool_specs
    }

    async def policy(query: str, turns: list[Turn]) -> PolicyStep:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        for i, t in enumerate(turns):
            if t.action is None:
                # The model answered in prose without a tool call. Keep its text in
                # context and nudge it to emit a tool call on this retry.
                if t.assistant_message is not None:
                    messages.append(deepcopy(t.assistant_message))
                elif t.thought:
                    messages.append({"role": "assistant", "content": t.thought})
                messages.append({"role": "user", "content": NUDGE_PROMPT})
                continue
            if t.assistant_message is not None:
                assistant_message = deepcopy(t.assistant_message)
                call_id = _first_tool_call_id(assistant_message) or t.tool_call_id or f"call_{i}"
            else:
                call_id = t.tool_call_id or f"call_{i}"
                assistant_message = {
                    "role": "assistant",
                    "content": t.thought,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": t.action["name"],
                            "arguments": json.dumps(t.action.get("arguments", {})),
                        },
                    }],
                }
            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _serialize_observation(t.observation),
            })

        # On a nudge retry (the previous turn produced no tool call) force a tool
        # call so the model can't answer in prose again.
        tool_choice = "required" if turns and turns[-1].action is None else "auto"
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_specs,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        choice = resp.choices[0]
        msg = choice.message
        assistant_message = _message_to_dict(msg)
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        action = None
        tool_call_id = _first_tool_call_id(assistant_message)
        thought = reasoning or msg.content or ""
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            action = {"name": tc.function.name, "arguments": args}
            tool_call_id = tc.id
        else:
            # Server parser missed a Hermes-XML tool call; recover it from whichever
            # channel it landed in. Thinking models often emit the call inside
            # reasoning_content (no clean <think> split), leaving content empty.
            blob = "\n".join(x for x in (reasoning, msg.content) if x)
            if "<function=" in blob or "<tool_call>" in blob:
                action, thought = _parse_content_tool_call(blob, param_types)
        usage = resp.usage
        return PolicyStep(
            thought=thought,
            action=action,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            assistant_message=assistant_message,
            tool_call_id=tool_call_id,
        )

    return policy
