"""Validity check: can each synthesized QA be solved by an agent in the live env?

Loads a QA jsonl, builds the live tool registry (Milvus + BGE-M3 + DuckDB +
graph), then runs ONE rollout per QA per policy model and judges whether the
agent reached the gold answer. Reports a solve rate per policy — a question no
strong agent can solve (paper unfindable / answer ungroundable / ambiguous) is a
bad training item. This is the k=1 validity filter; pass@k difficulty staging
comes later.

    uv run python -m s2cs.synthesis.run_rollout_check --qa-path data/qa/sh_gemini_v2

Needs the live env (Milvus up, a GPU for the encoder) and OPENROUTER/OPENAI creds.
"""

import asyncio
import dataclasses
import json
import logging
import os
from pathlib import Path

import openai
import tyro
from dotenv import load_dotenv

from s2cs.agent.judge import judge
from s2cs.agent.policy import make_openai_policy
from s2cs.agent.react import rollout
from s2cs.env.builder import build_tools
from s2cs.synthesis.cost import CostTrackingClient

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RolloutCheckArgs:
    qa_path: Path = Path("data/qa/sh_gemini_v2")
    policy_models: str = "google/gemini-3.1-flash-lite,deepseek/deepseek-v4-pro"
    judge_model: str = "google/gemini-3.1-flash-lite"
    base_url: str | None = None
    judge_base_url: str | None = None   # if set, judge runs on this endpoint (e.g. OpenRouter while policy is a local sglang server)
    judge_api_key: str | None = None
    max_turns: int = 12
    concurrency: int = 5
    temperature: float = 0.7
    out_dir: Path | None = None


def _load_qa(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    qas: list[dict] = []
    for f in files:
        for line in f.open():
            line = line.strip()
            if line:
                qas.append(json.loads(line))
    return qas


async def _amain(args: RolloutCheckArgs) -> None:
    load_dotenv()
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url is None:
        raise RuntimeError("OPENAI_BASE_URL not set (pass --base-url or set it in env / .env)")

    qas = _load_qa(args.qa_path)
    log.info("loaded %d QA from %s", len(qas), args.qa_path)

    tools_obj = build_tools()
    tools = dict(tools_obj.items())
    client = CostTrackingClient(
        openai.AsyncOpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", max_retries=6)
    )
    # Judge can live on a different endpoint than the policy (e.g. policy on a local
    # sglang server, judge on OpenRouter). Falls back to the policy client when unset.
    if args.judge_base_url:
        judge_client = CostTrackingClient(openai.AsyncOpenAI(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY",
            max_retries=6))
    else:
        judge_client = client
    sem = asyncio.Semaphore(args.concurrency)

    async def solve(qa: dict, policy) -> dict:
        async with sem:
            try:
                traj = await rollout(qa["question"], policy, tools, max_turns=args.max_turns)
            except Exception as exc:
                log.warning("rollout failed for %s: %s", qa.get("qa_id"), exc)
                return {"qa_id": qa.get("qa_id"), "question": qa["question"], "gold": qa["answer"],
                        "pred": None, "solved": False, "reason": "error", "n_turns": 0,
                        "trajectory": None}
            pred = traj.answer
            solved = False
            if pred is not None:
                verdict = await judge(qa["question"], qa["answer"], pred, client=judge_client, model=args.judge_model)
                solved = verdict.verdict == "Correct"
            return {"qa_id": qa.get("qa_id"), "question": qa["question"], "gold": qa["answer"],
                    "pred": pred, "solved": solved, "reason": traj.terminated_reason,
                    "n_turns": len(traj.turns), "trajectory": traj.to_dict()}

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for policy_model in [m.strip() for m in args.policy_models.split(",") if m.strip()]:
        policy = make_openai_policy(client, policy_model, tools, temperature=args.temperature)
        results = await asyncio.gather(*[solve(qa, policy) for qa in qas])
        n_solved = sum(r["solved"] for r in results)
        log.info("=== policy=%s : SOLVE RATE %d/%d = %.1f%% ===",
                 policy_model, n_solved, len(results), 100.0 * n_solved / max(len(results), 1))
        for r in results:
            log.info("  [%s] %s turns=%d reason=%s | Q=%.70s | gold=%.30s | pred=%.40s",
                     "OK  " if r["solved"] else "MISS", r["qa_id"], r["n_turns"], r["reason"],
                     r["question"], r["gold"], str(r["pred"]))
        if args.out_dir:
            tag = policy_model.replace("/", "_")
            with (args.out_dir / f"rollout_{tag}.jsonl").open("w") as fh:
                for r in results:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if client.calls_with_cost:
        log.info("policy LLM spend this run: $%.4f over %d/%d calls reporting cost",
                 client.total_cost, client.calls_with_cost, client.calls)
    if judge_client is not client and judge_client.calls_with_cost:
        log.info("judge LLM spend this run: $%.4f over %d/%d calls reporting cost",
                 judge_client.total_cost, judge_client.calls_with_cost, judge_client.calls)


def main(args: RolloutCheckArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(RolloutCheckArgs))
