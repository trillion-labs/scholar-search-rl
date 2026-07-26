import pytest

from s2cs.agent.judge import Verdict
from s2cs.agent.trajectory import Trajectory
from s2cs.synthesis.difficulty import GradedItem, GradedRollout, grade, stage_split


def _rec(sample_id, pass_at_k):
    return {"sample_id": sample_id, "pass_at_k": pass_at_k}


def test_stage_split_drops_extremes():
    records = [_rec("a", 0), _rec("b", 1), _rec("c", 4), _rec("d", 7), _rec("e", 8)]
    buckets = stage_split(records, lo=1, hi=7)
    assert set(buckets) == {1, 4, 7}
    assert "a" not in [r["sample_id"] for v in buckets.values() for r in v]  # pass@k==0 dropped
    assert "e" not in [r["sample_id"] for v in buckets.values() for r in v]  # pass@k==8 dropped


def test_stage_split_groups_by_pass_count():
    records = [_rec("a", 3), _rec("b", 3), _rec("c", 5)]
    buckets = stage_split(records, lo=1, hi=7)
    assert len(buckets[3]) == 2
    assert len(buckets[5]) == 1


def _traj(answer):
    return Trajectory(
        query="q", tool_set=[], turns=[], answer=answer,
        terminated_reason="submit_answer" if answer else "max_turns",
        prompt_tokens=0, completion_tokens=0,
    )


@pytest.mark.asyncio
async def test_grade_counts_correct_verdicts(monkeypatch):
    # policy alternates: answers on even calls, gives up on odd calls.
    calls = {"n": 0}

    async def fake_rollout(query, policy, tools, *, max_turns):
        i = calls["n"]
        calls["n"] += 1
        return _traj(f"ans{i}" if i % 2 == 0 else None)

    async def fake_judge(question, gold, prediction):
        # "ans0" is correct, "ans2" is wrong.
        return Verdict(verdict="Correct" if prediction == "ans0" else "Incorrect", reasoning="")

    monkeypatch.setattr("s2cs.synthesis.difficulty.rollout", fake_rollout)

    item = await grade("q", "gold", policy=None, tools={}, k=4, max_turns=10, judge_fn=fake_judge)
    assert item.k == 4
    assert len(item.rollouts) == 4
    # 2 answered (ans0, ans2), only ans0 judged Correct -> pass@4 == 1
    assert item.pass_at_k == 1
    assert sum(r.solved for r in item.rollouts) == 1


@pytest.mark.asyncio
async def test_grade_unanswered_never_calls_judge(monkeypatch):
    async def fake_rollout(query, policy, tools, *, max_turns):
        return _traj(None)

    judged = {"n": 0}

    async def tracking_judge(question, gold, prediction):
        judged["n"] += 1  # grade now catches exceptions, so a called-flag is the real guard
        return Verdict(verdict="Correct", reasoning="")

    monkeypatch.setattr("s2cs.synthesis.difficulty.rollout", fake_rollout)

    item = await grade("q", "gold", policy=None, tools={}, k=3, max_turns=10, judge_fn=tracking_judge)
    assert judged["n"] == 0
    assert item.pass_at_k == 0
    assert all(r.verdict is None and r.error is None for r in item.rollouts)


@pytest.mark.asyncio
async def test_grade_catches_rollout_error(monkeypatch):
    # A rollout that raises (timeout / 429 after retries) must be recorded as an
    # error, not crash the item or the batch.
    async def boom_rollout(query, policy, tools, *, max_turns):
        raise TimeoutError("upstream timed out")

    async def fake_judge(question, gold, prediction):
        return Verdict(verdict="Correct", reasoning="")

    events = []
    monkeypatch.setattr("s2cs.synthesis.difficulty.rollout", boom_rollout)

    item = await grade("q", "gold", policy=None, tools={}, k=3, max_turns=10,
                       judge_fn=fake_judge, on_event=events.append)
    assert item.pass_at_k == 0
    assert all(r.error == "TimeoutError" and not r.solved for r in item.rollouts)
    assert all(r.trajectory.terminated_reason == "error" for r in item.rollouts)
    assert events.count("start") == 3 and events.count("error") == 3 and events.count("done") == 0


@pytest.mark.asyncio
async def test_grade_times_out_hung_rollout(monkeypatch):
    # A rollout that hangs must not stall the item forever — rollout_timeout
    # bounds each rollout and a timed-out one is recorded as an error.
    import asyncio

    async def hang_rollout(query, policy, tools, *, max_turns):
        await asyncio.sleep(60)
        return _traj("late")

    async def fake_judge(question, gold, prediction):
        return Verdict(verdict="Correct", reasoning="")

    monkeypatch.setattr("s2cs.synthesis.difficulty.rollout", hang_rollout)

    item = await grade("q", "gold", policy=None, tools={}, k=2, max_turns=10,
                       judge_fn=fake_judge, rollout_timeout=0.05)
    assert item.pass_at_k == 0
    assert all(r.error == "TimeoutError" and not r.solved for r in item.rollouts)
