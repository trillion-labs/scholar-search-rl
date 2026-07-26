"""verl `BaseTool` adapter over the s2cs scholar env.

One class, dispatched by `config["tool_name"]`. `execute` calls the matching
`s2cs.env.build_tools()` callable in-process; the heavy backends those callables
close over (Milvus on the retrieval host, BGE-M3 via the embed server) are already remote
services, so no new retrieval server is needed. Per-trajectory `seed_paper_ids`
(passed via verl `create_kwargs`) become the `mask_paper_ids` source-masking set.
"""

import dataclasses
import inspect
import json
import logging
import uuid
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from s2cs.env import build_tools

log = logging.getLogger(__name__)

_TOOLS_CACHE: dict[int, Any] = {}


def _get_tools():
    pid = 0
    if pid not in _TOOLS_CACHE:
        _TOOLS_CACHE[pid] = build_tools()
    return _TOOLS_CACHE[pid]


def _to_text(obj: Any) -> str:
    def enc(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return {f.name: enc(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if isinstance(o, (list, tuple, set)):
            return [enc(x) for x in o]
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        return o

    try:
        return json.dumps(enc(obj), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


class S2CSTool(BaseTool):
    def __init__(self, config: dict, tool_schema):
        if isinstance(tool_schema, dict):
            tool_schema = OpenAIFunctionToolSchema(**tool_schema)
        super().__init__(config, tool_schema)
        self.tool_name = config["tool_name"]
        self._instances: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        instance_id = instance_id or str(uuid.uuid4())
        seed = kwargs.get("seed_paper_ids")
        self._instances[instance_id] = {"mask": set(seed) if seed else None}
        return instance_id, ToolResponse(text="")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        fn = _get_tools()[self.tool_name]
        params = dict(parameters)
        mask = self._instances.get(instance_id, {}).get("mask")
        if mask is not None and "mask_paper_ids" in getattr(fn, "__code__", _NO_CODE).co_varnames:
            params["mask_paper_ids"] = mask
        try:
            result = fn(**params)
            if inspect.isawaitable(result):
                result = await result
            return ToolResponse(text=_to_text(result)), 0.0, {"tool_name": self.tool_name, "status": "ok"}
        except Exception as e:
            log.warning("[S2CSTool:%s] %s", self.tool_name, e)
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"tool_name": self.tool_name, "status": "error"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class _NoCode:
    co_varnames: tuple = ()


_NO_CODE = _NoCode()
