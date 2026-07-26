import json
import logging
import os
from typing import Any, Callable

import openai
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef

from s2cs.agent.react import rollout
from s2cs.env.tools.submit_answer import make_submit_answer
from s2cs.eval.local_runner import make_policy as _make_policy

log = logging.getLogger(__name__)

_SURFACE_ALIGN_MODE = {"s2cs_strict": "strict", "s2cs_bodyrevive": "body_revive"}


def _surface_align_mode(tool_surface: str) -> str:
    if tool_surface not in _SURFACE_ALIGN_MODE:
        raise ValueError(f"unknown s2cs tool_surface: {tool_surface}")
    return _SURFACE_ALIGN_MODE[tool_surface]


def _traj_stamp(payload: dict, *, policy, bench, globalstep, tool_surface, run) -> dict:
    payload["policy"] = policy
    payload["bench"] = bench
    payload["globalstep"] = globalstep
    payload["tool_surface"] = tool_surface
    payload["run"] = run
    return payload


def _tool_def_to_openai_spec(td: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters.model_dump(exclude_none=True),
        },
    }


@solver
def make_astabench_solver(
    *,
    base_url: str,
    model: str = "default",
    max_turns: int = 40,
    temperature: float = 0.7,
    trajectory_dir: str | None = None,
    tool_surface: str = "asta",
    policy_label: str | None = None,
    globalstep: int | None = None,
    run_name: str | None = None,
    bench: str | None = None,
    max_retries: int = 2,
    chat_format: str = "qwen",
) -> Solver:
    """Solver that drives s2cs.agent.react.rollout over the task's tools.

    Must be `@solver`-decorated: inspect_ai's eval pipeline records the solver
    via `as_solver_spec`, which raises unless it is a registry object. Args are
    recorded in the eval log, so the policy API key is read from the
    OPENAI_API_KEY env var inside `solve` (not passed as an arg) to keep it out
    of the log.

    The intermediate ReAct turns (thought / tool call / observation) run inside
    our own `rollout`, which inspect never sees — its `.eval` log keeps only the
    input prompt and the final answer. So every rollout's full Trajectory is
    written to `trajectory_dir/<sample_id>.jsonl` when that dir is given; without
    it the turn-by-turn record is lost and post-hoc behavioral debugging is
    impossible.
    """
    submit_answer = make_submit_answer()
    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # API key comes from OPENAI_API_KEY (set in run.py) to keep it out of the
        # inspect eval log. For OpenRouter baselines base_url points at OpenRouter
        # and that env var holds the OpenRouter key; max_retries rides out its
        # transient rate-limits instead of dropping the sample.
        client = openai.AsyncOpenAI(base_url=base_url, max_retries=max_retries)
        inspect_tools: list[Tool] = list(state.tools)
        tools: dict[str, Callable] = {}
        specs: list[dict[str, Any]] = []
        if tool_surface.startswith("s2cs"):
            from s2cs.eval.astabench.tool_adapter import adapt_tools

            asta_by_name = {ToolDef(t).name: t for t in inspect_tools}
            specs, tools = adapt_tools(
                asta_by_name,
                align_mode=_surface_align_mode(tool_surface),
                question=state.input_text,
            )
        else:
            for t in inspect_tools:
                td = ToolDef(t)
                tools[td.name] = t
                specs.append(_tool_def_to_openai_spec(td))
        submit_td = ToolDef(submit_answer, name="submit_answer")
        tools["submit_answer"] = submit_answer
        specs.append(_tool_def_to_openai_spec(submit_td))

        policy = _make_policy(client, model, specs, temperature=temperature, chat_format=chat_format)
        traj = await rollout(state.input_text, policy, tools, max_turns=max_turns)

        if trajectory_dir:
            safe_id = str(state.sample_id).replace("/", "_")
            path = os.path.join(trajectory_dir, f"{safe_id}.jsonl")
            # Asta tool observations are arbitrary objects (not dataclasses), so
            # Trajectory.to_jsonl's strict encoder would raise on them. Use a
            # str-fallback encoder and never let a dump failure kill the sample.
            try:
                import dataclasses as _dc

                payload = _dc.asdict(traj)
                payload["sample_id"] = safe_id
                payload = _traj_stamp(payload, policy=policy_label, bench=bench,
                                      globalstep=globalstep, tool_surface=tool_surface, run=run_name)
                with open(path, "w") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, default=str))
            except Exception as exc:
                log.warning("trajectory dump failed for %s: %s", safe_id, exc)

        if traj.answer is not None:
            # astabench scorers (e.g. paper_finder get_model_json_output) do
            # json.loads(completion) and EvalSample requires completion to be a
            # str. The paper-finding agent submits a structured object, so
            # traj.answer can be a dict — serialise it back to a JSON string;
            # plain QA answers are already strings and pass through unchanged.
            ans = traj.answer
            state.output.completion = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
        else:
            log.warning("rollout terminated without answer: %s", traj.terminated_reason)

        return state

    return solve
