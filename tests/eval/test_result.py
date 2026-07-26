import json

from s2cs.eval.result import BenchResult, RunResult


def test_run_result_to_json_matches_readme_shape():
    run = RunResult(
        policy="qwen3-4b-thinking-stage2-step420",
        tool_set_eval=["paper_search", "snippet_search", "submit_answer"],
        tool_set_train=["paper_search", "snippet_search", "find_in_paper", "submit_answer"],
        benches={
            "astabench/paper_finding": BenchResult(
                metrics={"accuracy": 0.41},
                n=200,
                latency_p50_s=8.2,
                latency_p95_s=31.4,
            ),
            "internal/known_item": BenchResult(
                metrics={"recall@10": 0.83},
                n=50,
            ),
        },
        total_cost_usd=0.0,
        total_runtime_s=1810.0,
    )

    parsed = json.loads(run.to_json())

    assert parsed["policy"] == "qwen3-4b-thinking-stage2-step420"
    assert parsed["benches"]["astabench/paper_finding"]["metrics"]["accuracy"] == 0.41
    assert parsed["benches"]["astabench/paper_finding"]["latency_p95_s"] == 31.4
    assert parsed["benches"]["internal/known_item"]["metrics"]["recall@10"] == 0.83
    assert parsed["benches"]["internal/known_item"]["latency_p50_s"] is None
    assert parsed["total_runtime_s"] == 1810.0


def test_bench_result_latency_defaults_to_none():
    b = BenchResult(metrics={"accuracy": 0.5}, n=10)
    assert b.latency_p50_s is None
    assert b.latency_p95_s is None


def test_run_result_cost_runtime_default_zero():
    run = RunResult(
        policy="p",
        tool_set_eval=[],
        tool_set_train=[],
        benches={},
    )
    assert run.total_cost_usd == 0.0
    assert run.total_runtime_s == 0.0


def test_runresult_carries_provenance():
    r = RunResult(
        policy="step060", tool_set_eval=["search_papers"], tool_set_train=["search_papers"],
        benches={"astabench/litqa2_all": BenchResult(metrics={"accuracy": 0.3}, n=85)},
        run="litqa2_verl_astanative", checkpoint_path="/ck/epoch0...globalstep60",
        globalstep=60, tool_surface="asta", grader_model="openrouter/openai/gpt-5.4-mini",
        model="step060", base_url="http://localhost:30102/v1",
        eval_args={"max_turns": 20}, git_sha="abc1234", git_dirty=True, created="2026-06-22T00:00:00Z",
    )
    d = json.loads(r.to_json())
    assert d["run"] == "litqa2_verl_astanative"
    assert d["globalstep"] == 60 and d["git_dirty"] is True
    assert d["benches"]["astabench/litqa2_all"]["n"] == 85


def test_runresult_backward_compatible_defaults():
    r = RunResult(policy="base", tool_set_eval=[], tool_set_train=[], benches={})
    d = json.loads(r.to_json())
    assert d["run"] is None and d["globalstep"] is None and d["git_dirty"] is None
