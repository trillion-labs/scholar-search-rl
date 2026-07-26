"""Grade a QA pool by pass@k under a single policy (M2.3 difficulty filter).

Loads a QA jsonl (`{sample_id, query, gold_answer, ...}`), builds the live tool
registry (Milvus + BGE-M3 + DuckDB + graph), then for each item runs `k`
rollouts of the policy and judges each. Writes every rollout's full trajectory
(trajectories are always persisted) plus a per-item pass@k summary, and logs the
pass@k histogram and the trainable-band counts.

Resumable: items already present in the summary file are skipped, so a requeued
or walltime-killed job continues where it left off.

    uv run python -m s2cs.synthesis.run_difficulty \\
        --qa-path data/qa/trainer_13k.jsonl \\
        --base-url http://localhost:30000/v1 --policy-model qwen3-4b-thinking-2507 \\
        --judge-base-url https://openrouter.ai/api/v1 --judge-model openai/gpt-5.4-mini

Needs the live env (Milvus up, a GPU for the encoder), a policy endpoint, and
OPENROUTER_API_KEY for the judge.
"""

import asyncio
import dataclasses
import json
import logging
import os
import time
from collections import Counter, deque
from pathlib import Path

import openai
import tyro
from dotenv import load_dotenv

from s2cs.agent.policy import make_openai_policy
from s2cs.env.builder import build_tools
from s2cs.synthesis.difficulty import grade, make_judge_fn, stage_split

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GradeArgs:
    qa_path: Path = Path("data/qa/trainer_13k.jsonl")
    out_dir: Path = Path("data/qa/difficulty/qwen3-4b-thinking-2507")
    base_url: str = "http://localhost:30000/v1"
    policy_model: str = "qwen3-4b-thinking-2507"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_model: str = "openai/gpt-5.4-mini"
    k: int = 8
    max_turns: int = 20
    temperature: float = 0.6
    item_concurrency: int = 16
    monitor_interval_s: float = 30.0
    limit: int | None = None


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, int(q * len(s)))
    return round(s[i], 2)


def _done_sample_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    return {
        json.loads(line)["sample_id"]
        for line in summary_path.read_text().splitlines()
        if line.strip()
    }


async def _amain(args: GradeArgs) -> None:
    load_dotenv()

    rows = _load_rows(args.qa_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "pass_at_k.jsonl"
    rollouts_path = args.out_dir / "rollouts.jsonl"

    done = _done_sample_ids(summary_path)
    pending = [r for r in rows if r["sample_id"] not in done]
    log.info("loaded %d items from %s; %d already graded, %d to do",
             len(rows), args.qa_path, len(done), len(pending))
    if not pending:
        log.info("nothing to do")
        return

    tools = dict(build_tools().items())
    log.info("tools: %s", list(tools))

    policy = make_openai_policy(
        openai.AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), max_retries=4),
        args.policy_model, tools, temperature=args.temperature,
    )
    judge_key = os.environ.get("OPENROUTER_API_KEY")
    if not judge_key:
        raise RuntimeError("OPENROUTER_API_KEY not set — needed for the judge endpoint")
    judge_fn = make_judge_fn(
        openai.AsyncOpenAI(base_url=args.judge_base_url, api_key=judge_key, max_retries=6),
        args.judge_model,
    )

    sem = asyncio.Semaphore(args.item_concurrency)
    write_lock = asyncio.Lock()
    summary_fh = summary_path.open("a")
    rollouts_fh = rollouts_path.open("a")
    histogram: Counter[int] = Counter()
    reason_hist: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    latencies: deque[float] = deque(maxlen=8000)
    done_count = 0
    # Live concurrency probe — peak in-flight tells us the sustainable rollout
    # concurrency for the RL trainer; error rate tells us where it breaks.
    m = {"inflight": 0, "peak_inflight": 0, "rollouts": 0, "errors": 0}

    def on_event(ev: str) -> None:
        if ev == "start":
            m["inflight"] += 1
            m["peak_inflight"] = max(m["peak_inflight"], m["inflight"])
        else:  # "done" | "error"
            m["inflight"] -= 1
            m["rollouts"] += 1
            if ev == "error":
                m["errors"] += 1

    async def run_one(row: dict) -> None:
        nonlocal done_count
        async with sem:
            item = await grade(
                row["query"], row["gold_answer"], policy, tools,
                k=args.k, max_turns=args.max_turns, judge_fn=judge_fn, on_event=on_event,
            )
        n_answered = sum(1 for r in item.rollouts if r.trajectory.answer is not None)
        async with write_lock:
            # Write the rollouts first, then the summary line. The summary is the
            # resume marker (`_done_sample_ids` keys off it), so writing it last
            # guarantees a requeue never skips an item whose trajectories are
            # missing — at worst it re-grades an item whose rollouts were written
            # but whose summary was not (no duplicate summary, no orphaned skip).
            for j, r in enumerate(item.rollouts):
                rollouts_fh.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "k_idx": j,
                    "pred": r.trajectory.answer,
                    "solved": r.solved,
                    "reason": r.trajectory.terminated_reason,
                    "n_turns": len(r.trajectory.turns),
                    "elapsed_s": round(r.elapsed_s, 2),
                    "error": r.error,
                    "trajectory": r.trajectory.to_dict(),
                }, ensure_ascii=False) + "\n")
                reason_hist[r.trajectory.terminated_reason] += 1
                latencies.append(r.elapsed_s)
                if r.error:
                    error_types[r.error] += 1
            rollouts_fh.flush()
            summary_fh.write(json.dumps({
                "sample_id": row["sample_id"],
                "pass_at_k": item.pass_at_k,
                "k": item.k,
                "n_answered": n_answered,
                "query": row["query"],
                "gold": row["gold_answer"],
            }, ensure_ascii=False) + "\n")
            summary_fh.flush()
            histogram[item.pass_at_k] += 1
            done_count += 1

    start = time.monotonic()

    async def monitor() -> None:
        last_r, last_t = 0, start
        while True:
            await asyncio.sleep(args.monitor_interval_s)
            now = time.monotonic()
            wall, r = now - start, m["rollouts"]
            now_rate = (r - last_r) / max(now - last_t, 1e-9) * 60
            cum_rate = r / max(wall, 1e-9) * 60
            err_pct = 100.0 * m["errors"] / max(r, 1)
            log.info(
                "[probe] %.0fs | items %d/%d | rollouts=%d inflight=%d peak=%d/%d "
                "| rate now=%.0f/min cum=%.0f/min | err=%d(%.1f%%) | lat p50=%.1fs p95=%.1fs",
                wall, done_count, len(pending), r, m["inflight"], m["peak_inflight"],
                args.item_concurrency * args.k, now_rate, cum_rate, m["errors"], err_pct,
                _pct(list(latencies), 0.5), _pct(list(latencies), 0.95),
            )
            last_r, last_t = r, now

    mon = asyncio.create_task(monitor())
    try:
        await asyncio.gather(*(run_one(r) for r in pending))
    finally:
        mon.cancel()
        summary_fh.close()
        rollouts_fh.close()

    wall = time.monotonic() - start
    lat = list(latencies)
    all_records = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    trainable = stage_split(all_records, lo=1, hi=args.k - 1)
    metrics = {
        "configured": {"item_concurrency": args.item_concurrency, "k": args.k,
                       "max_concurrent_rollouts": args.item_concurrency * args.k,
                       "max_turns": args.max_turns, "policy_model": args.policy_model},
        "wall_s": round(wall, 1),
        "items_graded_this_run": done_count,
        "rollouts": m["rollouts"],
        "peak_inflight_rollouts": m["peak_inflight"],
        "errors": m["errors"],
        "error_rate": round(m["errors"] / max(m["rollouts"], 1), 4),
        "error_types": dict(error_types),
        "rollouts_per_min": round(m["rollouts"] / max(wall, 1e-9) * 60, 1),
        "items_per_min": round(done_count / max(wall, 1e-9) * 60, 1),
        "rollout_latency_s": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                              "max": round(max(lat), 2) if lat else 0.0},
        "terminated_reason": dict(reason_hist),
        "pass_at_k_hist": dict(sorted(histogram.items())),
        "trainable_band": {p: len(v) for p, v in sorted(trainable.items())},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log.info("=== CONCURRENCY PROBE: peak_inflight=%d/%d  rollouts/min=%.0f  err=%.1f%%  lat p50/p95=%.1f/%.1fs ===",
             m["peak_inflight"], args.item_concurrency * args.k, metrics["rollouts_per_min"],
             100 * metrics["error_rate"], metrics["rollout_latency_s"]["p50"], metrics["rollout_latency_s"]["p95"])
    log.info("=== pass@%d histogram: %s | trainable[1,%d]=%d ===",
             args.k, metrics["pass_at_k_hist"], args.k - 1, sum(len(v) for v in trainable.values()))
    log.info("wrote %s, %s, %s", summary_path, rollouts_path, args.out_dir / "metrics.json")


def main(args: GradeArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(GradeArgs))
