from types import SimpleNamespace

import pytest
from inspect_ai.tool import tool

from s2cs.agent.trajectory import Trajectory
from s2cs.eval.astabench import solver as solver_mod
from s2cs.eval.astabench.solver import _tool_def_to_openai_spec, make_astabench_solver
import s2cs.eval.astabench.solver as solver_mod


@tool(name="paper_search")
def _fake_paper_search():
    async def paper_search(query: str, limit: int = 10):
        """Search the corpus.

        Args:
          query: free-text query
          limit: max results
        """
        return [{"id": 1, "title": "fake"}]
    return paper_search


def test_tool_def_to_openai_spec_extracts_name_description_params():
    from inspect_ai.tool import ToolDef

    td = ToolDef(_fake_paper_search())
    spec = _tool_def_to_openai_spec(td)

    assert spec["type"] == "function"
    assert spec["function"]["name"] == "paper_search"
    assert "Search the corpus" in spec["function"]["description"]
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert "limit" in params["properties"]


def test_make_astabench_solver_returns_async_callable():
    import inspect as _inspect

    solver = make_astabench_solver(base_url="http://fake/v1", model="any", max_turns=3)
    assert _inspect.iscoroutinefunction(solver)


@pytest.mark.asyncio
async def test_solver_writes_answer_to_state_output(monkeypatch):
    captured = {}

    async def fake_rollout(query, policy, tools, *, max_turns):
        captured["query"] = query
        captured["tool_names"] = set(tools.keys())
        captured["max_turns"] = max_turns
        return Trajectory(
            query=query,
            tool_set=list(tools),
            turns=[],
            answer="42",
            terminated_reason="submit_answer",
            prompt_tokens=0,
            completion_tokens=0,
        )

    monkeypatch.setattr("s2cs.eval.astabench.solver.rollout", fake_rollout)

    solver = make_astabench_solver(base_url="http://fake/v1", model="any", max_turns=7)

    state = SimpleNamespace(
        tools=[_fake_paper_search()],
        input_text="what is the answer?",
        output=SimpleNamespace(completion=""),
    )
    out = await solver(state, generate=None)

    assert out.output.completion == "42"
    assert captured["query"] == "what is the answer?"
    assert captured["max_turns"] == 7
    assert {"paper_search", "submit_answer"} <= captured["tool_names"]


@pytest.mark.asyncio
async def test_solver_logs_warning_when_no_answer(monkeypatch, caplog):
    async def fake_rollout(query, policy, tools, *, max_turns):
        return Trajectory(
            query=query,
            tool_set=list(tools),
            turns=[],
            answer=None,
            terminated_reason="max_turns",
            prompt_tokens=0,
            completion_tokens=0,
        )

    monkeypatch.setattr("s2cs.eval.astabench.solver.rollout", fake_rollout)

    solver = make_astabench_solver(base_url="http://fake/v1", model="any", max_turns=3)
    state = SimpleNamespace(
        tools=[],
        input_text="q",
        output=SimpleNamespace(completion=""),
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="s2cs.eval.astabench.solver"):
        await solver(state, generate=None)

    assert state.output.completion == ""
    assert any("max_turns" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("surface,mode", [
    ("s2cs_strict", "strict"),
    ("s2cs_bodyrevive", "body_revive"),
])
def test_surface_to_align_mode(surface, mode):
    assert solver_mod._surface_align_mode(surface) == mode


def test_unknown_surface_rejected():
    with pytest.raises(ValueError):
        solver_mod._surface_align_mode("s2cs_bogus")
def test_traj_stamp_adds_provenance():
    out = solver_mod._traj_stamp({"sample_id": "s1"}, policy="step060",
        bench="astabench/litqa2_all", globalstep=60, tool_surface="asta", run="litqa2_verl_astanative")
    assert out["sample_id"] == "s1"
    assert out["policy"] == "step060" and out["globalstep"] == 60
    assert out["bench"] == "astabench/litqa2_all" and out["tool_surface"] == "asta"
    assert out["run"] == "litqa2_verl_astanative"
