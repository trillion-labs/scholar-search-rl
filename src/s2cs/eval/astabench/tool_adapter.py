"""Present the s2cs training tool surface to the model, route calls to Asta MCP.

The policy is RL-trained against the s2cs local tools (`search_papers`,
`read_paper`, `find_in_paper`, …) with their specific names, signatures and
result shapes. AstaBench tasks instead hand the model Asta MCP tools
(`search_papers_by_relevance`, `get_paper`, `snippet_search`, …). Evaluating on
the raw Asta surface measures the model on tools it never trained on.

This adapter closes that gap: it exposes the *s2cs* tool specs to the model
(identical to training, via `s2cs.agent.tools.specs`) and, when the model calls
one, translates the arguments and invokes the mapped Asta tool, returning Asta's
result unchanged. Per the agreed design the result content stays Asta-native —
read intents (read_paper/find_in_paper) route to snippet_search in body_revive
mode (and are omitted in strict mode); Asta string paperIds flow through as-is —
only the tool *interface* is aligned to training.
"""

import asyncio
import inspect
import logging
import random
import types
from typing import Any, Awaitable, Callable

from s2cs.agent.tools import specs as _specs
from s2cs.env.tools.registry import build_registry

log = logging.getLogger(__name__)

# Asta MCP throttles under load (HTTP 429 / 5xx / timeouts). The agent's dispatch
# turns any tool exception into an `{"error": ...}` observation the MODEL then
# sees — a transient rate-limit would derail the rollout. So absorb transient
# failures HERE with exponential backoff + jitter and only surface the result;
# genuine (non-transient) errors propagate immediately so the model can react.
_RETRYABLE = (
    "429", "rate limit", "rate-limit", "too many requests", "ratelimit",
    "timeout", "timed out", "503", "502", "504", "overloaded",
    "connection", "temporarily unavailable", "read error",
)
_MAX_RETRIES = 6


async def _call_with_retry(asta_tool, kwargs):
    delay = 1.0
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await asta_tool(**kwargs)
        except Exception as exc:  # noqa: BLE001 — classify by signature below
            msg = f"{type(exc).__name__}: {exc}".lower()
            if attempt == _MAX_RETRIES or not any(s in msg for s in _RETRYABLE):
                raise
            sleep = delay + random.uniform(0, delay * 0.3)
            log.warning("asta tool transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, sleep, exc)
            await asyncio.sleep(sleep)
            delay = min(delay * 2, 30.0)

# Legacy mapping (WRONG eval): read_paper/find_in_paper -> get_paper returned
# metadata only, so the trained body-reading policy never reached paper text.
# Replaced by snippet_search routing in body_revive mode; kept here as a record
# of what NOT to do.
#   "read_paper":    ("get_paper", lambda a: {"paper_id": _to_asta_id(a.get("paper_id")), "fields": _GET_PAPER_FIELDS}),
#   "find_in_paper": ("get_paper", lambda a: {"paper_id": _to_asta_id(a.get("paper_id")), "fields": _GET_PAPER_FIELDS}),

_SNIPPET_SCOPE_CANDIDATES = ("corpus_ids", "paperIds", "paper_ids", "corpusIds")

# s2cs tools use a numeric corpusId for paper_id (the id-space the model trained
# on). Asta get_paper/get_citations want a STRING id and accept "CorpusId:<n>"
# (a bare int raises). Translate so the read path works at all — without this
# every paper_info/read_paper call errored (96-100%). get_paper's `fields`
# defaults to title-only, so request abstract+tldr to give the model content.
_GET_PAPER_FIELDS = "title,abstract,tldr,year,authors,venue"


def _to_asta_id(paper_id):
    if paper_id is None:
        return None
    s = str(paper_id).strip()
    return f"CorpusId:{s}" if s.isdigit() else s  # bare corpusId -> CorpusId:; sha/prefixed pass through


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def s2cs_tool_specs() -> list[dict[str, Any]]:
    """The OpenAI tool specs the model saw in training (retrieval tools only).

    Built from the real s2cs tool callables so the schemas are byte-identical to
    training. The registry factories only close over their backends (Milvus,
    encoder, …) — they do not touch them at construction — so dummy backends are
    enough to recover the callables for signature/docstring introspection; the
    closures are never invoked here. `submit_answer` is excluded: the solver adds
    its own.
    """
    dummy_encoder = types.SimpleNamespace(encode_hybrid=None, encode_dense=None)
    registry = build_registry(
        papers=None, chunks=None, graph=None, reader=None, encoder=dummy_encoder
    )
    registry.pop("submit_answer", None)
    return _specs(registry)


def _to_snippet_corpus_id(paper_id):
    """Corpus id form for snippet_search scoping. Defaults to the bare numeric
    corpusId (string); Task 6 verifies live whether the server wants the bare id
    or the 'CorpusId:<n>' form and this helper is the single place to flip it."""
    if paper_id is None:
        return None
    return str(paper_id).strip()


def _snippet_scope_key(snippet_tool) -> str | None:
    try:
        params = inspect.signature(snippet_tool).parameters
    except (TypeError, ValueError):
        return None
    return next((c for c in _SNIPPET_SCOPE_CANDIDATES if c in params), None)


def adapt_tools(
    asta_by_name: dict[str, Callable[..., Awaitable[Any]]],
    align_mode: str = "body_revive",
    question: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Awaitable[Any]]]]:
    """Wrap Asta tools behind the s2cs interface for the given alignment mode.

    align_mode:
      "strict"      — only s2cs tools with an honest 1:1 Asta backend; the fake
                      full-text readers (read_paper/find_in_paper/find_similar)
                      are not exposed. Body text is reachable only via
                      search_snippets -> snippet_search.
      "body_revive" — full s2cs surface; read_paper/find_in_paper route to
                      snippet_search (the real body channel), find_similar ->
                      get_citations. read_paper has no query of its own, so the
                      task `question` is used as the snippet query.
    """
    if align_mode not in ("strict", "body_revive"):
        raise ValueError(f"unknown align_mode: {align_mode}")

    all_specs = {s["function"]["name"]: s for s in s2cs_tool_specs()}
    snippet_tool = asta_by_name.get("snippet_search")
    scope_key = _snippet_scope_key(snippet_tool) if snippet_tool else None

    specs: list[dict[str, Any]] = []
    tools: dict[str, Callable[..., Awaitable[Any]]] = {}

    def add(s2cs_name, asta_name, translate):
        asta_tool = asta_by_name.get(asta_name)
        if asta_tool is None or s2cs_name not in all_specs:
            return
        specs.append(all_specs[s2cs_name])
        tools[s2cs_name] = _make_adapter(asta_tool, translate)

    def _scoped(extra, paper_ids):
        out = dict(extra)
        if scope_key and paper_ids:
            out[scope_key] = [_to_snippet_corpus_id(p) for p in paper_ids]
        return _drop_none(out)

    add("search_papers", "search_papers_by_relevance",
        lambda a: _drop_none({"keyword": a.get("query"), "limit": a.get("limit", 10)}))
    add("search_snippets", "snippet_search",
        lambda a: _scoped({"query": a.get("query"), "limit": a.get("limit", 10)}, a.get("paper_ids")))
    add("paper_info", "get_paper",
        lambda a: _drop_none({"paper_id": _to_asta_id(a.get("paper_id")), "fields": _GET_PAPER_FIELDS}))
    add("list_references", "get_citations",
        lambda a: _drop_none({"paper_id": _to_asta_id(a.get("paper_id")), "limit": a.get("limit")}))
    add("list_citations", "get_citations",
        lambda a: _drop_none({"paper_id": _to_asta_id(a.get("paper_id")), "limit": a.get("limit", 20)}))

    if align_mode == "body_revive":
        # find_similar -> get_citations is not a faithful 1:1, so it is exposed
        # only in body_revive (omitted from the honest strict surface).
        add("find_similar", "get_citations",
            lambda a: _drop_none({"paper_id": _to_asta_id(a.get("paper_id")), "limit": a.get("limit", 10)}))
        add("read_paper", "snippet_search",
            lambda a: _scoped({"query": question or "", "limit": a.get("limit", 10)},
                              [a.get("paper_id")] if a.get("paper_id") is not None else None))
        add("find_in_paper", "snippet_search",
            lambda a: _scoped({"query": a.get("pattern") or "", "limit": a.get("max_hits", 20)},
                              [a.get("paper_id")] if a.get("paper_id") is not None else None))

    return specs, tools


def _make_adapter(
    asta_tool: Callable[..., Awaitable[Any]], translate: Callable[[dict], dict]
) -> Callable[..., Awaitable[Any]]:
    async def adapted(**s2cs_args: Any) -> Any:
        return await _call_with_retry(asta_tool, translate(s2cs_args))

    return adapted
