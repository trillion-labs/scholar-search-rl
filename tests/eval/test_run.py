from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2cs.eval.result import BenchResult
from s2cs.eval.run import (
    _ASTABENCH_TASKS,
    RunArgs,
    _extract_bench_result,
    _resolve_astabench_task,
    main,
)


def test_extract_bench_result_pulls_metrics_and_n():
    score = SimpleNamespace(
        metrics={
            "accuracy": SimpleNamespace(value=0.41),
            "recall@10": SimpleNamespace(value=0.83),
        }
    )
    elog = SimpleNamespace(
        results=SimpleNamespace(scores=[score], completed_samples=200),
    )

    br = _extract_bench_result(elog)

    assert isinstance(br, BenchResult)
    assert br.n == 200
    assert br.metrics["accuracy"] == pytest.approx(0.41)
    assert br.metrics["recall@10"] == pytest.approx(0.83)


def test_extract_bench_result_handles_none_results():
    elog = SimpleNamespace(results=None)
    br = _extract_bench_result(elog)
    assert br.n == 0
    assert br.metrics == {}


def test_all_astabench_tasks_are_importable():
    for bench, (mod, attr) in _ASTABENCH_TASKS.items():
        m = import_module(mod)
        assert hasattr(m, attr), f"{bench}: {mod}.{attr} missing"
        assert callable(getattr(m, attr))


def test_resolve_astabench_task_returns_callable():
    fn = _resolve_astabench_task("astabench/paper_finder_validation")
    assert callable(fn)


def test_main_rejects_unknown_bench(tmp_path: Path):
    args = RunArgs(bench="garbage/foo", out_path=tmp_path / "r.json")
    with pytest.raises(ValueError, match="unknown bench"):
        main(args)


def test_main_internal_not_implemented(tmp_path: Path):
    args = RunArgs(bench="internal", out_path=tmp_path / "r.json")
    with pytest.raises(NotImplementedError, match="M3.2"):
        main(args)


def test_main_transfer_not_implemented(tmp_path: Path):
    args = RunArgs(bench="transfer", out_path=tmp_path / "r.json")
    with pytest.raises(NotImplementedError, match="M3.2"):
        main(args)


def test_main_astabench_writes_results(monkeypatch, tmp_path: Path):
    score = SimpleNamespace(metrics={"accuracy": SimpleNamespace(value=0.5)})
    fake_log = SimpleNamespace(
        results=SimpleNamespace(scores=[score], completed_samples=10),
    )

    monkeypatch.setattr("s2cs.eval.run.inspect_eval", lambda **kwargs: [fake_log])
    monkeypatch.setattr(
        "s2cs.eval.run._resolve_astabench_task",
        lambda bench: (lambda: SimpleNamespace(name="fake_task")),
    )

    out_path = tmp_path / "out.json"
    args = RunArgs(
        bench="astabench/paper_finder_validation",
        policy_label="test-policy",
        out_path=out_path,
    )
    main(args)

    import json
    payload = json.loads(out_path.read_text())
    assert payload["policy"] == "test-policy"
    assert payload["benches"]["astabench/paper_finder_validation"]["n"] == 10
    assert payload["benches"]["astabench/paper_finder_validation"]["metrics"]["accuracy"] == 0.5


import json
from s2cs.eval import run as run_mod


def test_dump_sample_scores(tmp_path):
    Score = lambda v, a: SimpleNamespace(value=v, answer=a)
    log = SimpleNamespace(samples=[
        SimpleNamespace(id="s1", target="B", scores={"score_litqa2": Score({"is_correct": True, "is_sure": True}, "B")}),
        SimpleNamespace(id="s2", target="C", scores={"score_litqa2": Score({"is_correct": False, "is_sure": False}, "A")}),
        SimpleNamespace(id="s3", target="X", scores={"score_litqa2": Score({"is_correct": "I", "is_sure": False}, "Y")}),
        SimpleNamespace(id="s4", target="Z", scores={"score_litqa2": Score({"is_correct": "C", "is_sure": True}, "Z")}),
    ])
    n = run_mod._dump_sample_scores(log, tmp_path, "astabench/litqa2_all", "step060")
    assert n == 4
    p = tmp_path / "scores" / "litqa2_all" / "step060.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[0] == {"sample_id": "s1", "target": "B", "answer": "B", "is_correct": True, "is_sure": True}
    assert rows[1]["is_correct"] is False and rows[1]["answer"] == "A"
    assert rows[2]["is_correct"] is False
    assert rows[3]["is_correct"] is True


def test_git_info_returns_sha_and_dirty():
    sha, dirty = run_mod._git_info()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)
    assert dirty is None or isinstance(dirty, bool)


def test_main_writes_provenance(tmp_path, monkeypatch):
    out = tmp_path / "pf_step060.json"
    args = run_mod.RunArgs(
        bench="astabench/litqa2_all", policy_label="step060", out_path=out,
        tool_surface="asta", grader_model="g", model="step060", base_url="u",
        run_name="litqa2_verl_astanative", trial="value4k_band17_lr_seal",
        checkpoint_path="/ck/...globalstep60", globalstep=60, max_turns=20, max_samples=16,
    )
    monkeypatch.setattr(run_mod, "_run_astabench", lambda a: BenchResult(metrics={"accuracy": 0.3}, n=85))
    run_mod.main(args)
    d = json.loads(out.read_text())
    assert d["run"] == "litqa2_verl_astanative" and d["globalstep"] == 60
    assert d["tool_surface"] == "asta" and d["checkpoint_path"].endswith("globalstep60")
    assert d["eval_args"]["max_turns"] == 20
    assert isinstance(d["git_dirty"], bool) or d["git_dirty"] is None
