"""pass@k difficulty grading (M2.3).

Runs `k` independent rollouts of one policy on a QA item and judges each, then
counts how many reached the gold answer. An item nobody solves (`pass@k == 0`)
is unsolvable; one everybody solves (`pass@k == k`) is trivial. Only the band in
between carries a learning signal, so `stage_split` keeps `[lo, hi]` and drops
the extremes.

`grade` is the per-item primitive; batch orchestration (concurrency, resume,
persistence) lives in `run_difficulty.py`.
"""

import asyncio
import dataclasses
import time
from collections import defaultdict
from typing import Awaitable, Callable

from s2cs.agent.judge import Verdict, judge
from s2cs.agent.react import Policy, rollout
from s2cs.agent.trajectory import Trajectory

JudgeFn = Callable[[str, str, str], Awaitable[Verdict]]


@dataclasses.dataclass
class GradedRollout:
    trajectory: Trajectory
    verdict: Verdict | None
    solved: bool
    elapsed_s: float
    error: str | None  # exception type name if rollout/judge raised; None on success


@dataclasses.dataclass
class GradedItem:
    pass_at_k: int
    k: int
    rollouts: list[GradedRollout]


async def grade(
    query: str,
    gold: str,
    policy: Policy,
    tools: dict[str, Callable],
    *,
    k: int,
    max_turns: int,
    judge_fn: JudgeFn,
    on_event: Callable[[str], None] | None = None,
    rollout_timeout: float | None = 300.0,
) -> GradedItem:
    """Run `k` rollouts of `policy` on one item and judge each; count the solves.

    The `k` rollouts run concurrently. A rollout that never submits an answer
    counts as unsolved (`verdict=None`); one that *raises* (timeout, 429 after
    retries, dispatch error) is caught and recorded as `error` rather than
    bringing down the whole item or the batch — essential at high concurrency.

    `on_event(ev)` (optional) fires `"start"` when a rollout begins and exactly
    one of `"done"` / `"error"` when it finishes — the live-concurrency hook the
    batch runner uses to track in-flight count, throughput, and error rate.
    """

    async def one() -> GradedRollout:
        if on_event:
            on_event("start")
        t0 = time.monotonic()
        try:
            traj = await asyncio.wait_for(
                rollout(query, policy, tools, max_turns=max_turns), timeout=rollout_timeout
            )
            verdict = None
            if traj.answer is not None:
                verdict = await judge_fn(query, gold, traj.answer)
            solved = verdict is not None and verdict.verdict == "Correct"
            if on_event:
                on_event("done")
            return GradedRollout(trajectory=traj, verdict=verdict, solved=solved,
                                 elapsed_s=time.monotonic() - t0, error=None)
        except Exception as exc:
            if on_event:
                on_event("error")
            empty = Trajectory(query=query, tool_set=list(tools), turns=[], answer=None,
                               terminated_reason="error", prompt_tokens=0, completion_tokens=0)
            return GradedRollout(trajectory=empty, verdict=None, solved=False,
                                 elapsed_s=time.monotonic() - t0, error=type(exc).__name__)

    rollouts = list(await asyncio.gather(*(one() for _ in range(k))))
    return GradedItem(pass_at_k=sum(r.solved for r in rollouts), k=k, rollouts=rollouts)


def make_judge_fn(client, model: str) -> JudgeFn:
    async def judge_fn(question: str, gold: str, prediction: str) -> Verdict:
        return await judge(question, gold, prediction, client=client, model=model)

    return judge_fn


def stage_split(records: list[dict], *, lo: int = 1, hi: int = 7) -> dict[int, list[dict]]:
    """Bucket graded records by `pass_at_k`, keeping only the trainable band.

    Drops `pass_at_k < lo` (unsolvable) and `> hi` (trivial). Each record must
    carry an integer `pass_at_k`.
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        p = rec["pass_at_k"]
        if lo <= p <= hi:
            buckets[p].append(rec)
    return dict(buckets)
