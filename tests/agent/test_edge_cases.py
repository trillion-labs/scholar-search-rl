import asyncio

from tests.agent.conftest import make_fake_resp

from s2cs.agent.llm import chat_json
from s2cs.agent.react import PolicyStep, rollout
from s2cs.agent.trajectory import Trajectory
from s2cs.env.tools.submit_answer import make_submit_answer


def test_chat_json_empty_string_treated_as_invalid(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content=""),
        make_fake_resp(content='{"ok": true}'),
    ]
    out = asyncio.run(chat_json(fake_openai_client, "m", [{"role": "user", "content": "x"}], max_attempts=3))
    assert out == {"ok": True}


def test_rollout_max_turns_zero_returns_empty_trajectory():
    async def never_called(query, turns):
        raise AssertionError("policy should not be called when max_turns=0")
    traj = asyncio.run(rollout("q", never_called, {"submit_answer": make_submit_answer()}, max_turns=0))
    assert isinstance(traj, Trajectory)
    assert traj.terminated_reason == "max_turns"
    assert traj.turns == []


def test_trajectory_jsonl_with_empty_turns():
    import json
    traj = Trajectory(query="q", tool_set=[], turns=[], answer=None, terminated_reason="max_turns",
                      prompt_tokens=0, completion_tokens=0)
    data = json.loads(traj.to_jsonl())
    assert data["turns"] == []
    assert data["answer"] is None


def test_rollout_records_token_totals_across_turns():
    step1 = PolicyStep(thought="s1", action={"name": "submit_answer", "arguments": {"answer": "ok"}},
                      prompt_tokens=100, completion_tokens=20)

    async def policy(q, turns):
        return step1
    traj = asyncio.run(rollout("q", policy, {"submit_answer": make_submit_answer()}, max_turns=1))
    assert traj.prompt_tokens == 100
    assert traj.completion_tokens == 20
