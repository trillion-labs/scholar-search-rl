import dataclasses
import json
from typing import Any


@dataclasses.dataclass
class Turn:
    thought: str
    action: dict[str, Any] | None
    observation: Any
    assistant_message: dict[str, Any] | None = None
    tool_call_id: str | None = None


@dataclasses.dataclass
class Trajectory:
    query: str
    tool_set: list[str]
    turns: list[Turn]
    answer: str | None
    terminated_reason: str
    prompt_tokens: int
    completion_tokens: int
    nudge_count: int = 0

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, default=_default)

    def to_dict(self) -> dict:
        """Serialization-safe dict view (dataclass observations flattened via _default)."""
        return json.loads(self.to_jsonl())


def _default(o: Any) -> Any:
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    raise TypeError(f"not serializable: {type(o).__name__}")
