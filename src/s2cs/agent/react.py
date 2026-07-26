import dataclasses
import logging
from typing import Any, Awaitable, Callable

from s2cs.agent.tools import dispatch
from s2cs.agent.trajectory import Trajectory, Turn
from s2cs.env.tools.submit_answer import AnswerSubmission

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    thought: str
    action: dict[str, Any] | None
    prompt_tokens: int
    completion_tokens: int
    assistant_message: dict[str, Any] | None = None
    tool_call_id: str | None = None


Policy = Callable[[str, list[Turn]], Awaitable[PolicyStep]]


async def rollout(
    query: str,
    policy: Policy,
    tools: dict[str, Callable],
    *,
    max_turns: int = 40,
    max_nudges: int = 2,
) -> Trajectory:
    turns: list[Turn] = []
    pt = ct = 0
    nudge_count = 0
    for _ in range(max_turns):
        step = await policy(query, turns)
        pt += step.prompt_tokens
        ct += step.completion_tokens

        if step.action is None:
            # No valid tool call: the model answered in prose. Keep the failed turn
            # in context (the policy renders it + a nudge and forces a tool call on
            # the retry) instead of aborting. Give up after max_nudges.
            turns.append(
                Turn(
                    thought=step.thought,
                    action=None,
                    observation=None,
                    assistant_message=step.assistant_message,
                    tool_call_id=step.tool_call_id,
                )
            )
            nudge_count += 1
            if nudge_count > max_nudges:
                return Trajectory(query, list(tools), turns, None, "error", pt, ct, nudge_count)
            continue

        name = step.action["name"]
        args = step.action.get("arguments", {}) or {}
        observation = await dispatch(name, args, tools)
        turns.append(
            Turn(
                thought=step.thought,
                action=step.action,
                observation=observation,
                assistant_message=step.assistant_message,
                tool_call_id=step.tool_call_id,
            )
        )

        if isinstance(observation, AnswerSubmission):
            return Trajectory(query, list(tools), turns, observation.answer, "submit_answer", pt, ct, nudge_count)

    log.info("rollout hit max_turns=%d", max_turns)
    return Trajectory(query, list(tools), turns, None, "max_turns", pt, ct, nudge_count)
