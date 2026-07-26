import asyncio

from s2cs.agent.tools import dispatch, specs, tool_spec


def some_tool(query: str, limit: int = 10, year_min: int | None = None,
              mask_paper_ids: set[int] | None = None) -> list[dict]:
    """Search for stuff. First line of docstring is the description."""
    return [{"q": query, "limit": limit, "year_min": year_min}]


def test_tool_spec_top_level_shape():
    spec = tool_spec("some_tool", some_tool)
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "some_tool"
    assert "Search for stuff" in spec["function"]["description"]


def test_tool_spec_required_args():
    spec = tool_spec("some_tool", some_tool)
    required = spec["function"]["parameters"]["required"]
    assert required == ["query"]


def test_tool_spec_skips_mask_paper_ids():
    props = tool_spec("some_tool", some_tool)["function"]["parameters"]["properties"]
    assert "mask_paper_ids" not in props


def test_tool_spec_default_optionals_present():
    props = tool_spec("some_tool", some_tool)["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"query", "limit", "year_min"}


def test_specs_for_multiple_tools():
    def another(text: str) -> str:
        """Echo."""
        return text

    out = specs({"some_tool": some_tool, "another": another})
    names = {s["function"]["name"] for s in out}
    assert names == {"some_tool", "another"}


def test_dispatch_calls_sync_tool():
    def echo(x: int) -> int:
        return x * 2
    result = asyncio.run(dispatch("echo", {"x": 21}, {"echo": echo}))
    assert result == 42


def test_dispatch_awaits_async_tool():
    async def async_echo(x: int) -> int:
        return x * 3
    result = asyncio.run(dispatch("async_echo", {"x": 14}, {"async_echo": async_echo}))
    assert result == 42


def test_dispatch_async_tool_with_multiple_kwargs():
    async def combine(a: str, b: str) -> str:
        return f"{a}/{b}"
    result = asyncio.run(dispatch("combine", {"a": "x", "b": "y"}, {"combine": combine}))
    assert result == "x/y"


def test_dispatch_unknown_tool_returns_error_dict():
    result = asyncio.run(dispatch("nope", {}, {}))
    assert result == {"error": "unknown tool: nope"}


def test_dispatch_tool_error_returns_error_observation():
    def boom(x: int) -> int:
        raise ValueError("bad arg")
    result = asyncio.run(dispatch("boom", {"x": 1}, {"boom": boom}))
    assert result == {"error": "ValueError: bad arg"}


def test_param_schema_optional_int_is_integer():
    props = tool_spec("some_tool", some_tool)["function"]["parameters"]["properties"]
    assert props["year_min"] == {"type": "integer"}


def test_param_schema_list_arg_is_array():
    def with_list(query: str, paper_ids: list[int] | None = None) -> list[dict]:
        """Tool with a list-typed arg."""
        return []
    props = tool_spec("with_list", with_list)["function"]["parameters"]["properties"]
    assert props["paper_ids"] == {"type": "array", "items": {"type": "integer"}}
