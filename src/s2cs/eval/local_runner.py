import asyncio
import dataclasses
import json
import logging
import os
import re
from typing import Any, Callable

import openai

from s2cs.agent.policy import SYSTEM_PROMPT
from s2cs.agent.react import PolicyStep, rollout
from s2cs.agent.tools import specs as build_specs
from s2cs.agent.trajectory import Trajectory, Turn

log = logging.getLogger(__name__)


def serialize_observation(obs: Any) -> str:
    def default(o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return str(o)

    return json.dumps(obs, ensure_ascii=False, default=default)[:8000]


def parse_tool_args(raw: str | None) -> dict[str, Any]:
    """Tolerantly parse a tool call's `arguments` string into a dict.

    Some models (seen with gpt-oss via OpenRouter) append text after the JSON
    object, so strict `json.loads` raises `Extra data`. Unguarded, that single
    exception propagates out of the rollout and aborts the whole inspect task
    (the bench scores 0 samples). Salvage the leading object via `raw_decode`,
    falling back to `{}` when nothing parses."""
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw.lstrip())
        except (json.JSONDecodeError, ValueError):
            return {}
    return obj if isinstance(obj, dict) else {}


# GLM-style tool calls are NOT hermes JSON. The chat template emits, per call:
#   <tool_call>{name}
#   <arg_key>k</arg_key><arg_value>v</arg_value> ...
#   </tool_call>
# The name is followed by the arg block: GLM-4.5-style templates put a newline after the
# name, GLM-5.2 runs straight into <arg_key> with no separator -- so the gap is
# \s* (zero-or-more), not a required \n. The closing tag is optional too: the
# </tool_call> stop we pass for GLM (see make_policy) is trimmed by default, so a
# client-side recovery must tolerate its absence and read to end-of-text.
_GLM_TOOL_CALL = re.compile(r"<tool_call>\s*([A-Za-z_]\w*)\s*(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_GLM_ARG = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)


def _coerce_glm_arg(value: str) -> Any:
    """The template JSON-encodes non-string args (10, [1,2,3], true) and leaves
    strings bare. Mirror sglang's Glm4MoeDetector: json-first, fall back to str."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_glm_tool_call(text: str | None) -> dict[str, Any] | None:
    """Recover the FIRST GLM-XML tool call from raw model text, or None if absent.

    The server's `glm` tool parser normally returns these as `tool_calls`; this is
    the client-side fallback for when it doesn't (the </tool_call> stop trims the
    closing tag the server parser's regex needs). Returns {"name", "arguments"}."""
    if not text or "<tool_call>" not in text:
        return None
    m = _GLM_TOOL_CALL.search(text)
    if m is None:
        return None
    args = {k.strip(): _coerce_glm_arg(v.strip()) for k, v in _GLM_ARG.findall(m.group(2))}
    return {"name": m.group(1).strip(), "arguments": args}


# GLM control tokens (<|user|>, <|observation|>, ...) leak into a tool-call arg
# when the model ends its turn on an eos that no_stop_trim keeps in the output
# (seen on submit_answer: answer="...E\"}<|user|>"). They never belong in an arg.
_GLM_SPECIAL = re.compile(r"<\|\w+\|>")


def _strip_glm_special(value: Any) -> Any:
    return _GLM_SPECIAL.sub("", value).strip() if isinstance(value, str) else value


def make_policy(
    client: openai.AsyncOpenAI,
    model: str,
    tool_specs: list[dict[str, Any]],
    *,
    temperature: float,
    system_prompt: str = SYSTEM_PROMPT,
    chat_format: str = "qwen",
) -> Callable:
    """Build the OpenAI-compatible chat policy that `react.rollout` drives.

    Inspect-free: the AstaBench solver and the external-benchmark adapters both
    use this. The tool specs are built by the caller (inspect `ToolDef` for the
    Asta path, `s2cs.agent.tools.specs` for our local callables).

    `chat_format` adapts the request/response to the served model's dialect:
    "qwen" (default, hermes tool calls + server-parsed `tool_calls`) or "glm45"
    (GLM-style template — stop at </tool_call> to avoid the runaway, and
    recover the GLM-XML call client-side if the server parser misses it).
    """

    async def policy(query: str, turns: list[Turn]) -> PolicyStep:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        for i, t in enumerate(turns):
            if t.action is None:
                continue
            call_id = f"call_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": t.thought,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": t.action["name"],
                                "arguments": json.dumps(t.action.get("arguments", {})),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": serialize_observation(t.observation),
                }
            )

        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            tools=tool_specs,
            tool_choice="auto",
            temperature=temperature,
            stream=False,
        )
        if chat_format == "glm45":
            # This GLM-style format ends the assistant span at </tool_call> and
            # never emits <|observation|> itself, so with no stop it runs away
            # spewing tool calls until the token budget is gone (never submits ->
            # reward 0; mirrors trainer patch 0012). Stop at the tag; no_stop_trim
            # keeps it so the server's `glm` tool parser still sees a complete call.
            create_kwargs["stop"] = ["</tool_call>"]
            create_kwargs["extra_body"] = {"no_stop_trim": True}
        try:
            resp = await client.chat.completions.create(**create_kwargs)
        except openai.BadRequestError as exc:
            # Non-retryable 400 — almost always prompt+max_tokens over the model's
            # context window on high-coverage rollouts like paper_finder. Degrade
            # THIS sample (no tool call -> the rollout
            # nudges then ends) instead of letting it propagate out of the solver and
            # interrupt the WHOLE bench (we saw 1/66 logged then abort).
            log.warning("policy create 400, ending sample: %s", str(exc)[:200])
            return PolicyStep(thought="", action=None, prompt_tokens=0, completion_tokens=0)
        msg = resp.choices[0].message
        thought = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or msg.content or ""
        action = None
        glm_call = parse_glm_tool_call(msg.content or thought) if chat_format == "glm45" else None
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = parse_tool_args(tc.function.arguments)
            action = {"name": tc.function.name, "arguments": args}
        elif glm_call is not None:
            # Server `glm` parser missed it (the </tool_call> stop trimmed the
            # closing tag): recover the GLM-XML call before mistaking the raw
            # <tool_call> blob for a prose final answer.
            action = glm_call
        elif msg.content:
            action = {"name": "submit_answer", "arguments": {"answer": msg.content}}
        else:
            log.warning(
                "no tool_calls and empty content: finish=%s",
                resp.choices[0].finish_reason,
            )
        if action is not None and chat_format == "glm45":
            raw_args = action.get("arguments")
            if isinstance(raw_args, dict):
                action["arguments"] = {k: _strip_glm_special(v) for k, v in raw_args.items()}
        # Some OpenAI-compatible providers (seen via OpenRouter) omit usage; treat
        # missing token counts as 0 rather than crashing the rollout.
        usage = resp.usage
        return PolicyStep(
            thought=thought,
            action=action,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    return policy


def dump_trajectory(traj: Trajectory, path: str, sample_id: str) -> None:
    """Persist a rollout. Tool observations may be arbitrary objects, so use a
    str-fallback encoder and never let a dump failure kill the sample."""
    try:
        payload = dataclasses.asdict(traj)
        payload["sample_id"] = sample_id
        with open(path, "w") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        log.warning("trajectory dump failed for %s: %s", sample_id, exc)


async def run_rollouts(
    queries: list[str],
    *,
    base_url: str,
    model: str,
    tools: dict[str, Callable],
    api_key: str = "EMPTY",
    max_turns: int = 40,
    temperature: float = 0.7,
    system_prompt: str = SYSTEM_PROMPT,
    concurrency: int = 16,
    trajectory_dir: str | None = None,
    sample_ids: list[str] | None = None,
    max_retries: int = 2,
    chat_format: str = "qwen",
) -> list[Trajectory]:
    """Drive `react.rollout` over a list of queries against local callables.

    Returns one Trajectory per query, in input order. Trajectories are always
    persisted when `trajectory_dir` is given (eval keeps the turn-by-turn record
    permanently; inspect never sees these rollouts).
    """
    ids = sample_ids or [str(i) for i in range(len(queries))]
    if len(ids) != len(queries):
        raise ValueError(f"sample_ids length {len(ids)} != queries length {len(queries)}")

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
    tool_specs = build_specs(tools)
    policy = make_policy(
        client,
        model,
        tool_specs,
        temperature=temperature,
        system_prompt=system_prompt,
        chat_format=chat_format,
    )

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    async def one(query: str, sid: str) -> Trajectory:
        async with sem:
            traj = await rollout(query, policy, tools, max_turns=max_turns)
        if trajectory_dir:
            safe_id = sid.replace("/", "_")
            dump_trajectory(traj, os.path.join(trajectory_dir, f"{safe_id}.jsonl"), safe_id)
        return traj

    return await asyncio.gather(*(one(q, sid) for q, sid in zip(queries, ids)))
