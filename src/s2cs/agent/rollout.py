import asyncio
import dataclasses
import logging
from typing import Awaitable, Callable

from s2cs.agent.judge import Verdict
from s2cs.agent.react import Policy, rollout
from s2cs.agent.trajectory import Trajectory

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ScoredTrajectory:
    trajectory: Trajectory
    gold: str
    verdict: Verdict | None


JudgeFn = Callable[[str, str, str], Awaitable[Verdict]]


async def rollout_many(
    queries: list[tuple[str, str]],
    policy: Policy,
    tools: dict[str, Callable],
    *,
    k: int = 8,
    max_turns: int = 40,
    judge_fn: JudgeFn,
) -> list[list[ScoredTrajectory]]:
    async def one(query: str, gold: str) -> ScoredTrajectory:
        traj = await rollout(query, policy, tools, max_turns=max_turns)
        verdict = None
        if traj.answer is not None:
            verdict = await judge_fn(query, gold, traj.answer)
        return ScoredTrajectory(trajectory=traj, gold=gold, verdict=verdict)

    grouped: list[list[ScoredTrajectory]] = []
    for query, gold in queries:
        tasks = [one(query, gold) for _ in range(k)]
        grouped.append(await asyncio.gather(*tasks))
    return grouped
