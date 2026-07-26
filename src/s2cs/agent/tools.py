import asyncio
import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin


def tool_spec(name: str, fn: Callable) -> dict[str, Any]:
    """Build an OpenAI-style tool definition from a callable's docstring + signature."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "mask_paper_ids":
            continue
        properties[pname] = _param_schema(param)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _scalar_schema(ann: Any) -> dict[str, Any]:
    if ann in (int, "int"):       return {"type": "integer"}
    if ann in (float, "float"):   return {"type": "number"}
    if ann in (bool, "bool"):     return {"type": "boolean"}
    return {"type": "string"}


def _param_schema(param: inspect.Parameter) -> dict[str, Any]:
    ann = param.annotation
    if get_origin(ann) in (Union, types.UnionType):       # X | None  ->  X
        non_none = [a for a in get_args(ann) if a is not type(None)]
        ann = non_none[0] if non_none else ann
    if get_origin(ann) in (list, tuple, set):             # list[int]  ->  array of integer
        item = get_args(ann)
        return {"type": "array", "items": _scalar_schema(item[0]) if item else {"type": "string"}}
    return _scalar_schema(ann)


def specs(tools: dict[str, Callable]) -> list[dict[str, Any]]:
    return [tool_spec(name, fn) for name, fn in tools.items()]


async def dispatch(name: str, args: dict[str, Any], tools: dict[str, Callable]) -> Any:
    if name not in tools:
        return {"error": f"unknown tool: {name}"}
    fn = tools[name]
    try:
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        return await asyncio.to_thread(fn, **args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
