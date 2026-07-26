import dataclasses
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class BenchResult:
    metrics: dict[str, float]
    n: int
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None


@dataclasses.dataclass(frozen=True)
class RunResult:
    policy: str
    tool_set_eval: list[str]
    tool_set_train: list[str]
    benches: dict[str, BenchResult]
    total_cost_usd: float = 0.0
    total_runtime_s: float = 0.0
    run: str | None = None
    checkpoint_path: str | None = None
    globalstep: int | None = None
    tool_surface: str | None = None
    grader_model: str | None = None
    model: str | None = None
    base_url: str | None = None
    eval_args: dict[str, Any] | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    created: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False)
