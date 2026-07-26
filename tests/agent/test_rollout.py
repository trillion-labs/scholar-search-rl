import asyncio

from s2cs.agent.judge import Verdict
from s2cs.agent.react import PolicyStep
from s2cs.agent.rollout import rollout_many
from s2cs.env.tools.submit_answer import make_submit_answer


def _scripted_policy(*steps_per_call):
    """Returns a policy that yields each step on successive calls, cycling per query."""
    state = {"i": 0}

    async def policy(query, turns):
        i = state["i"]
        state["i"] = i + 1
        return steps_per_call[i % len(steps_per_call)]
    return policy


def test_rollout_many_groups_by_k():
    submit = PolicyStep(thought="done", action={"name": "submit_answer", "arguments": {"answer": "42"}}, prompt_tokens=1, completion_tokens=1)
    policy = _scripted_policy(submit)
    tools = {"submit_answer": make_submit_answer()}

    async def judge_fn(q, gold, pred):
        return Verdict(verdict="Correct" if pred == gold else "Incorrect", reasoning="x")

    grouped = asyncio.run(rollout_many([("q1", "42"), ("q2", "x")], policy, tools, k=3, judge_fn=judge_fn))
    assert len(grouped) == 2
    assert all(len(g) == 3 for g in grouped)
    assert all(s.verdict.verdict == "Correct" for s in grouped[0])
    assert all(s.verdict.verdict == "Incorrect" for s in grouped[1])


def test_rollout_many_skips_judge_when_no_answer():
    no_action = PolicyStep(thought="stuck", action=None, prompt_tokens=0, completion_tokens=0)
    policy = _scripted_policy(no_action)
    tools = {"submit_answer": make_submit_answer()}

    judge_calls = {"n": 0}

    async def judge_fn(q, gold, pred):
        judge_calls["n"] += 1
        return Verdict(verdict="Correct", reasoning="")

    grouped = asyncio.run(rollout_many([("q", "g")], policy, tools, k=2, judge_fn=judge_fn))
    assert judge_calls["n"] == 0
    assert all(s.verdict is None for s in grouped[0])
    assert all(s.trajectory.terminated_reason == "error" for s in grouped[0])
