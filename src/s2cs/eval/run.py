import dataclasses
import json
import logging
import os
import subprocess
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import tyro
from inspect_ai import eval as inspect_eval

from s2cs.eval.astabench.solver import make_astabench_solver
from s2cs.eval.result import BenchResult, RunResult

log = logging.getLogger(__name__)


def _git_info() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return sha, (status.strip() != "")
    except Exception:
        return None, None


_DEFAULT_GRADER_MODEL = "openrouter/openai/gpt-5.4-mini"


@dataclasses.dataclass(frozen=True)
class RunArgs:
    bench: str = "astabench/paper_finder_validation"
    policy_label: str = "unknown"
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "EMPTY"
    out_path: Path = Path("eval/results.json")
    max_turns: int = 40
    temperature: float = 0.7
    limit: int | None = None
    max_samples: int = 16
    # policy-client HTTP retries (exponential backoff on 429/5xx/timeout). 0 = fail
    # fast (fine for a local sglang server); bump for remote APIs (OpenRouter) where
    # transient rate-limits would otherwise drop samples from the accuracy denominator.
    max_retries: int = 2
    trajectory_dir: Path | None = None
    grader_model: str = _DEFAULT_GRADER_MODEL
    # astabench tool surface: "asta" (native MCP tools), "s2cs_strict", or
    # "s2cs_bodyrevive" (training tool interface via tool_adapter, Asta backend).
    tool_surface: str = "asta"
    # served model's chat/tool dialect: "qwen" (default) or "glm45"
    # (</tool_call> stop + GLM-XML tool-call recovery).
    chat_format: str = "qwen"
    # provenance: identifies the training run that produced the evaluated checkpoint
    run_name: str | None = None
    trial: str | None = None
    checkpoint_path: str | None = None
    globalstep: int | None = None
    # litsearch (and other local-runner benches)
    milvus_uri: str = "http://localhost:19530"
    model_name: str = "BAAI/bge-m3"
    devices: str | None = None
    concurrency: int = 16
    top_k: int = 20
    # paper_search_qa
    retrieve_url: str = "http://localhost:8000/retrieve"
    retrieve_topk: int = 10
    split: str = "test"
    # simpleqa (web-search transfer; web surface = Serper, key from SERPER_API_KEY)
    simpleqa_csv: str = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    simpleqa_sample: int | None = None
    # browsecomp (harder web-transfer lens; same web surface as simpleqa)
    browsecomp_sample: int | None = None
    # gaia (file-less web-answerable subset; local curated jsonl, GAIA exact-match scorer)
    gaia_path: str = "data/gaia.jsonl"
    gaia_sample: int | None = None
    web_num_results: int = 10
    # internal (in-domain)
    seed: int = 0
    min_refs: int = 12


_ASTABENCH_TASKS: dict[str, tuple[str, str]] = {
    "astabench/paper_finder_validation":        ("astabench.evals.paper_finder", "paper_finder_validation"),
    "astabench/paper_finder_test":              ("astabench.evals.paper_finder", "paper_finder_test"),
    "astabench/paper_finder_litqa2_validation": ("astabench.evals.paper_finder", "paper_finder_litqa2_validation"),
    "astabench/paper_finder_litqa2_test":       ("astabench.evals.paper_finder", "paper_finder_litqa2_test"),
    "astabench/sqa_dev":                        ("astabench.evals.sqa", "sqa_dev"),
    "astabench/sqa_test":                       ("astabench.evals.sqa", "sqa_test"),
    "astabench/litqa2_validation":              ("astabench.evals.labbench", "litqa2_validation"),
    "astabench/litqa2_test":                    ("astabench.evals.labbench", "litqa2_test"),
    "astabench/litqa2_all":                     ("astabench.evals.labbench", "litqa2"),
}


_grader_model = _DEFAULT_GRADER_MODEL


def _resolve_astabench_task(bench: str) -> Any:
    """Return a zero-arg factory that builds the fully-wired inspect Task.

    The grader is wired lazily inside the factory (so merely resolving a task
    has no side effects), reading the module-level `_grader_model` that
    `_run_astabench` sets from the run args.

    - paper_finder* (incl. paper_finder_litqa2): the relevance/parse judges are
      module globals patched in place by `use_openrouter_grader`.
    - sqa*: the rubric/citation grader is a `scorer_model` arg defaulting to a
      Google model; pass our OpenRouter model instead.
    - litqa2* (LabBench): multiple-choice, scored by exact `choice` match — no
      LLM grader, nothing to redirect.
    """
    module_path, attr = _ASTABENCH_TASKS[bench]
    module = import_module(module_path)

    if bench.startswith("astabench/paper_finder"):
        def factory() -> Any:
            from s2cs.eval.astabench.grader import use_openrouter_grader

            use_openrouter_grader(_grader_model)
            return getattr(module, attr)()
        return factory

    if bench.startswith("astabench/sqa"):
        split = "test" if bench.endswith("_test") else "dev"
        return lambda: module.sqa(split=split, scorer_model=_grader_model)

    if bench == "astabench/litqa2_all":
        return lambda: module.litqa2(split="all")

    return getattr(module, attr)


def _dump_sample_scores(eval_log: Any, out_dir: Path, bench: str, policy: str) -> int:
    """Write per-sample {sample_id,target,answer,is_correct,is_sure} from an inspect
    EvalLog so trajectories can be joined to correctness. Best-effort; never raises."""
    samples = getattr(eval_log, "samples", None) or []
    short = bench.split("/")[-1]
    dest = out_dir / "scores" / short
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{policy}.jsonl"
    n = 0
    with open(path, "w") as fh:
        for s in samples:
            scores = getattr(s, "scores", {}) or {}
            sc = next(iter(scores.values()), None)
            val = getattr(sc, "value", {}) if sc is not None else {}
            ic = val.get("is_correct") if isinstance(val, dict) else None
            is_correct = (ic == "C") if isinstance(ic, str) else ic
            row = {
                "sample_id": str(getattr(s, "id", "")),
                "target": getattr(s, "target", None),
                "answer": getattr(sc, "answer", None) if sc is not None else None,
                "is_correct": is_correct,
                "is_sure": val.get("is_sure") if isinstance(val, dict) else None,
            }
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def _extract_bench_result(elog: Any) -> BenchResult:
    metrics: dict[str, float] = {}
    n = 0
    if elog.results is not None:
        n = elog.results.completed_samples
        for score in elog.results.scores:
            for name, m in score.metrics.items():
                metrics[name] = float(m.value)
    return BenchResult(metrics=metrics, n=n)


def _run_astabench(args: RunArgs) -> BenchResult:
    if args.bench not in _ASTABENCH_TASKS:
        raise ValueError(
            f"unknown astabench bench: {args.bench}. known: {sorted(_ASTABENCH_TASKS)}"
        )

    # The agent policy reads its key from OPENAI_API_KEY (the solver builds an
    # AsyncOpenAI client from base_url + this env var). Setting it here also
    # satisfies astabench's import-time get_model("openai/...") and lets the
    # OpenRouter grader client initialise (it authenticates via OPENROUTER_API_KEY).
    os.environ["OPENAI_API_KEY"] = args.api_key

    # litqa2-backed tasks (litqa2_*, paper_finder_litqa2_*) load a dataset card
    # whose feature schema needs a datasets>=3.3 type; shim it for our 3.2 pin.
    from s2cs.eval.astabench.compat import patch_datasets_list_feature

    patch_datasets_list_feature()

    global _grader_model
    _grader_model = args.grader_model
    task = _resolve_astabench_task(args.bench)()

    solver = make_astabench_solver(
        base_url=args.base_url,
        model=args.model,
        max_turns=args.max_turns,
        temperature=args.temperature,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        tool_surface=args.tool_surface,
        policy_label=args.policy_label,
        globalstep=args.globalstep,
        run_name=args.run_name,
        bench=args.bench,
        max_retries=args.max_retries,
        chat_format=args.chat_format,
    )

    logs = inspect_eval(
        tasks=task,
        solver=solver,
        model=None,
        limit=args.limit,
        max_samples=args.max_samples,
        # Don't let one bad sample abort the whole bench. A scorer that raises on a
        # model's answer (seen: paper_finder's get_model_json_output -> KeyError
        # 'output' when the policy submits JSON without the expected schema key)
        # would otherwise interrupt the task (1-2/N logged then abort). Tolerate
        # per-sample errors; they score 0 and the bench completes.
        fail_on_error=False,
    )
    if args.trajectory_dir is not None:
        try:
            out_root = args.out_path.parent.parent   # <OUT>/<bench>/pf_x.json -> <OUT>
            _dump_sample_scores(logs[0], out_root, args.bench, args.policy_label)
        except Exception as exc:
            log.warning("sample-score dump failed: %s", exc)
    return _extract_bench_result(logs[0])


def _run_litsearch(args: RunArgs) -> BenchResult:
    from s2cs.eval import litsearch

    devices = [d.strip() for d in args.devices.split(",")] if args.devices else None
    return litsearch.run(
        base_url=args.base_url,
        model=args.model,
        milvus_uri=args.milvus_uri,
        model_name=args.model_name,
        api_key=args.api_key,
        devices=devices,
        limit=args.limit,
        max_turns=args.max_turns,
        temperature=args.temperature,
        concurrency=args.concurrency,
        top_k=args.top_k,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _run_paper_search_qa(args: RunArgs) -> BenchResult:
    from s2cs.eval import paper_search_qa

    return paper_search_qa.run(
        base_url=args.base_url,
        model=args.model,
        retrieve_url=args.retrieve_url,
        api_key=args.api_key,
        split=args.split,
        retrieve_topk=args.retrieve_topk,
        limit=args.limit,
        max_turns=args.max_turns,
        temperature=args.temperature,
        concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _run_simpleqa(args: RunArgs) -> BenchResult:
    from s2cs.eval import simpleqa

    return simpleqa.run(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        csv_source=args.simpleqa_csv,
        sample=args.simpleqa_sample,
        seed=args.seed,
        limit=args.limit,
        grader_model=args.grader_model,
        web_num_results=args.web_num_results,
        max_turns=args.max_turns,
        temperature=args.temperature,
        concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _run_browsecomp(args: RunArgs) -> BenchResult:
    from s2cs.eval import browsecomp

    return browsecomp.run(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        sample=args.browsecomp_sample,
        seed=args.seed,
        limit=args.limit,
        grader_model=args.grader_model,
        web_num_results=args.web_num_results,
        max_turns=args.max_turns,
        temperature=args.temperature,
        concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _run_gaia(args: RunArgs) -> BenchResult:
    from s2cs.eval import gaia

    return gaia.run(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        path=args.gaia_path,
        sample=args.gaia_sample,
        seed=args.seed,
        limit=args.limit,
        web_num_results=args.web_num_results,
        max_turns=args.max_turns,
        temperature=args.temperature,
        concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _internal_devices(args: RunArgs) -> list[str] | None:
    return [d.strip() for d in args.devices.split(",")] if args.devices else None


def _run_internal_known_item(args: RunArgs) -> BenchResult:
    from s2cs.eval import internal

    return internal.run_known_item(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        model_name=args.model_name, devices=_internal_devices(args),
        limit=args.limit or 100, seed=args.seed, max_turns=args.max_turns,
        temperature=args.temperature, concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def _run_internal_citation_holdout(args: RunArgs) -> BenchResult:
    from s2cs.eval import internal

    return internal.run_citation_holdout(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        model_name=args.model_name, devices=_internal_devices(args),
        limit=args.limit or 100, seed=args.seed, min_refs=args.min_refs, max_turns=args.max_turns,
        temperature=args.temperature, concurrency=args.concurrency,
        trajectory_dir=str(args.trajectory_dir) if args.trajectory_dir else None,
        chat_format=args.chat_format,
    )


def main(args: RunArgs) -> None:
    start = time.monotonic()

    if args.bench.startswith("astabench/"):
        bench_result = _run_astabench(args)
    elif args.bench == "litsearch":
        bench_result = _run_litsearch(args)
    elif args.bench == "paper_search_qa":
        bench_result = _run_paper_search_qa(args)
    elif args.bench == "simpleqa":
        bench_result = _run_simpleqa(args)
    elif args.bench == "browsecomp":
        bench_result = _run_browsecomp(args)
    elif args.bench == "gaia":
        bench_result = _run_gaia(args)
    elif args.bench == "internal_known_item":
        bench_result = _run_internal_known_item(args)
    elif args.bench == "internal_citation_holdout":
        bench_result = _run_internal_citation_holdout(args)
    elif args.bench == "internal":
        raise NotImplementedError("internal eval (M3.2) — not yet implemented")
    elif args.bench == "transfer":
        raise NotImplementedError("transfer eval (M3.2) — not yet implemented")
    else:
        raise ValueError(f"unknown bench: {args.bench}")

    elapsed = time.monotonic() - start
    sha, dirty = _git_info()
    from datetime import datetime, timezone
    run = RunResult(
        policy=args.policy_label,
        tool_set_eval=[],
        tool_set_train=[],
        benches={args.bench: bench_result},
        total_cost_usd=0.0,
        total_runtime_s=elapsed,
        run=args.run_name,
        checkpoint_path=args.checkpoint_path,
        globalstep=args.globalstep,
        tool_surface=args.tool_surface,
        grader_model=args.grader_model,
        model=args.model,
        base_url=args.base_url,
        eval_args={"max_turns": args.max_turns, "max_samples": args.max_samples,
                   "limit": args.limit, "temperature": args.temperature},
        git_sha=sha,
        git_dirty=dirty,
        created=datetime.now(timezone.utc).isoformat(),
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(run.to_json())
    log.info("wrote %s", args.out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(RunArgs))
