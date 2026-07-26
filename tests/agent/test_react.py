import asyncio

from s2cs.agent.react import PolicyStep, rollout
from s2cs.env.tools.submit_answer import AnswerSubmission, make_submit_answer


def _scripted_policy(*steps: PolicyStep):
    iterator = iter(steps)

    async def policy(query, turns):
        try:
            return next(iterator)
        except StopIteration:
            return PolicyStep(thought="(no more steps)", action=None, prompt_tokens=0, completion_tokens=0)
    return policy


def test_submit_answer_terminates_rollout():
    policy = _scripted_policy(
        PolicyStep(thought="thinking", action={"name": "submit_answer", "arguments": {"answer": "42"}}, prompt_tokens=10, completion_tokens=2),
    )
    tools = {"submit_answer": make_submit_answer()}
    traj = asyncio.run(rollout("q", policy, tools, max_turns=5))
    assert traj.terminated_reason == "submit_answer"
    assert traj.answer == "42"
    assert len(traj.turns) == 1
    assert traj.prompt_tokens == 10
    assert traj.completion_tokens == 2


def test_max_turns_terminates_rollout():
    def fake_search(query: str, limit: int = 10) -> list[str]:
        """Search."""
        return ["x"]
    repeat_step = PolicyStep(thought="t", action={"name": "search", "arguments": {"query": "q"}}, prompt_tokens=1, completion_tokens=1)
    policy = _scripted_policy(*([repeat_step] * 10))
    tools = {"search": fake_search, "submit_answer": make_submit_answer()}
    traj = asyncio.run(rollout("q", policy, tools, max_turns=3))
    assert traj.terminated_reason == "max_turns"
    assert traj.answer is None
    assert len(traj.turns) == 3


def test_none_action_nudges_then_terminates_with_error():
    # A policy that never emits a tool call is nudged up to max_nudges times, then
    # the rollout gives up with "error" (one initial turn + max_nudges retries).
    policy = _scripted_policy(
        PolicyStep(thought="(no plan)", action=None, prompt_tokens=5, completion_tokens=0),
    )
    traj = asyncio.run(
        rollout("q", policy, {"submit_answer": make_submit_answer()}, max_turns=5, max_nudges=2)
    )
    assert traj.terminated_reason == "error"
    assert traj.answer is None
    assert traj.nudge_count == 3
    assert len(traj.turns) == 3


def test_unknown_tool_observation_then_continues():
    policy = _scripted_policy(
        PolicyStep(thought="bad", action={"name": "missing", "arguments": {}}, prompt_tokens=1, completion_tokens=1),
        PolicyStep(thought="recovered", action={"name": "submit_answer", "arguments": {"answer": "ok"}}, prompt_tokens=1, completion_tokens=1),
    )
    traj = asyncio.run(rollout("q", policy, {"submit_answer": make_submit_answer()}, max_turns=5))
    assert traj.turns[0].observation == {"error": "unknown tool: missing"}
    assert traj.terminated_reason == "submit_answer"
    assert traj.answer == "ok"


def test_submit_answer_returns_answer_submission_observation():
    policy = _scripted_policy(
        PolicyStep(thought="done", action={"name": "submit_answer", "arguments": {"answer": "hi"}}, prompt_tokens=0, completion_tokens=0),
    )
    traj = asyncio.run(rollout("q", policy, {"submit_answer": make_submit_answer()}, max_turns=2))
    assert isinstance(traj.turns[0].observation, AnswerSubmission)
    assert traj.turns[0].observation.answer == "hi"
