"""Convert an s2cs QA-pool JSONL into a verl multiturn training parquet.

Each row becomes a verl `tool_agent` sample: a chat prompt (system + user),
the judge ground truth, and `extra_info` carrying the question, `answer_type`,
and per-tool `create_kwargs` (where `seed_paper_ids` flows through to
`S2CSTool` source-masking). System prompts are the same ones the agent trains
against (`s2cs.agent.policy`).
"""

import json
import logging

import pandas as pd

from s2cs.agent.policy import PAPER_SET_SYSTEM_PROMPT, SYSTEM_PROMPT

log = logging.getLogger(__name__)


def to_verl_row(row: dict, idx: int, tool_names: list[str]) -> dict:
    is_ps = row.get("answer_type") == "paper_set"
    system = PAPER_SET_SYSTEM_PROMPT if is_ps else SYSTEM_PROMPT

    # Always emit seed_paper_ids (default []) so the parquet struct schema is
    # consistent and never empty; S2CSTool.create treats [] as no source-mask.
    create_kwargs = {"seed_paper_ids": list(row.get("seed_paper_ids") or [])}
    tools_kwargs = {name: {"create_kwargs": dict(create_kwargs)} for name in tool_names}

    extra_info = {
        "index": idx,
        "need_tools_kwargs": True,
        "question": row["query"],
        "answer_type": "paper_set" if is_ps else "qa",
        "tools_kwargs": tools_kwargs,
    }
    if is_ps:
        extra_info["relevance"] = row["relevance"]
        extra_info["est_total_relevant"] = int(row["est_total_relevant"])

    return {
        "data_source": "s2cs_paper_set" if is_ps else "s2cs_qa",
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": row["query"]},
        ],
        "agent_name": "tool_agent",
        "reward_model": {"ground_truth": "" if is_ps else row["gold_answer"]},
        "extra_info": extra_info,
    }


def write_parquet(jsonl_path: str, out_path: str, tool_names: list[str]) -> int:
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(to_verl_row(json.loads(line), i, tool_names))
    pd.DataFrame(rows).to_parquet(out_path)
    log.info("wrote %d rows -> %s", len(rows), out_path)
    return len(rows)


def _default_tool_names() -> list[str]:
    """The full s2cs tool surface in registry order — the same tools the verl runs
    expose via `configs/trainer/tool_config.yaml`. Reuses the dummy-backend
    registry introspection from `gen_tool_config` (no live Milvus/encoder needed)."""
    from s2cs.trainer.tools.gen_tool_config import _schemas

    return list(_schemas())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="QA-pool JSONL -> verl multiturn training parquet")
    ap.add_argument("--in", dest="jsonl", required=True, help="trainer-format JSONL (mixed qa/paper_set rows ok)")
    ap.add_argument("--out", required=True, help="output verl parquet path")
    ap.add_argument(
        "--tools", nargs="+", default=None, help="tool names for per-tool create_kwargs (default: full s2cs registry)"
    )
    args = ap.parse_args()
    write_parquet(args.jsonl, args.out, args.tools or _default_tool_names())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
