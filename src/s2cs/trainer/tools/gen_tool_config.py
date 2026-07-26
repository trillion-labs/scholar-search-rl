"""Emit a verl multiturn tool-config from the s2cs tool registry.

The schemas are recovered exactly as the agent sees them in training: the
registry factories only *close over* their backends (Milvus, encoder, graph,
reader) without touching them at construction, so dummy backends are enough to
introspect each tool's signature/docstring via `s2cs.agent.tools.specs`. This
is the same trick as `s2cs.eval.astabench.tool_adapter.s2cs_tool_specs`.
"""

import logging
import types

import yaml

from s2cs.agent.tools import specs as _specs
from s2cs.env.tools.registry import build_registry

log = logging.getLogger(__name__)

DEFAULT_CLASS = "s2cs.trainer.tools.s2cs_tool.S2CSTool"


def _schemas() -> dict:
    dummy_encoder = types.SimpleNamespace(encode_hybrid=None, encode_dense=None)
    registry = build_registry(
        papers=None, chunks=None, graph=None, reader=None, encoder=dummy_encoder
    )
    return {s["function"]["name"]: s for s in _specs(registry)}


def build_tool_config(tool_names: list[str], class_name: str = DEFAULT_CLASS) -> dict:
    schemas = _schemas()
    tools = []
    for name in tool_names:
        if name not in schemas:
            raise KeyError(f"unknown tool {name!r}; known={sorted(schemas)}")
        tools.append(
            {
                "class_name": class_name,
                "config": {"tool_name": name, "type": "native"},
                "tool_schema": schemas[name],
            }
        )
    return {"tools": tools}


def write_tool_config(path: str, tool_names: list[str], class_name: str = DEFAULT_CLASS) -> None:
    cfg = build_tool_config(tool_names, class_name)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log.info("wrote verl tool config: %s (%d tools)", path, len(tool_names))
