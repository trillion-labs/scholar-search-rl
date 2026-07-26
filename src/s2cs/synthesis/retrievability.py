"""Agentic retrievability check: can a capable searcher FIND the seed paper?

Unlike `filter.retrievable` (one model-derived query → search_papers/snippets), this
runs a short tool-using session: the LLM gets the question plus the full retrieval/
navigation tool surface and up to `max_turns` tool calls to surface the seed paper.
It captures multi-step findability (search → find_similar / citations → hop) and the
use of metadata filters (venue/year) — an UPPER BOUND on findability ("could a capable
agent find this in a few steps?"). It is NOT a rollout: no answer, no judge; success is
purely whether `seed_paper_id` appears in any tool's results. Difficulty against the
actual training policy is pass@8's job.
"""

import dataclasses
import json
import logging
from typing import Any, Callable

import openai

from s2cs.agent.llm import chat_json
from s2cs.agent.react import PolicyStep
from s2cs.agent.tools import dispatch, specs
from s2cs.agent.trajectory import Turn

log = logging.getLogger(__name__)

FIND_SYSTEM_PROMPT = """You are a literature-search agent. Your ONLY goal is to FIND the one specific paper the question below is about, using the search and navigation tools.

- Search by content; if the first results are off, refine the query or try a different tool.
- search_papers matches titles/abstracts and can filter by year/venue; search_snippets matches body text — use it for specific reported values, findings, or setups.
- If a search surfaces a related (but not the target) paper, follow find_similar / list_citations / list_references to hop toward the target.
- Do NOT answer the question. Just locate the paper.
Think briefly, then call a tool."""


QUERY_GEN_PROMPT = """You are locating ONE specific paper in a large scientific corpus with a search engine.
Given the question, propose up to {n} SEARCH ATTEMPTS that would surface that paper. Available tools:
- search_papers: hybrid search over titles/abstracts; can filter by year_min, year_max, venue.
- search_snippets: search over paper BODY text — use it for a specific reported value, finding, or setup.
Vary the attempts (different phrasings, keyword vs natural-language, papers vs snippets, with/without year/venue).
Use year/venue ONLY if the question states them. Do NOT put the answer in a query.

Question: {q}

Return ONLY a JSON list of up to {n} objects, no prose:
[{{"tool": "search_papers" | "search_snippets", "query": "...", "year_min": <int or null>, "year_max": <int or null>, "venue": <str or null>}}]"""


async def make_search_queries(question: str, *, client: openai.AsyncOpenAI, model: str,
                              n: int = 5, temperature: float = 0.7) -> list[dict]:
    """Phase-1 prep: deepseek proposes up to n search attempts (a list) for the question,
    WITHOUT seeing any results (non-adaptive) — so this needs no encoder/Milvus. Phase 2
    (BGE-M3) executes the attempts."""
    payload = await chat_json(client, model, [{"role": "user", "content": QUERY_GEN_PROMPT.format(q=question, n=n)}],
                              temperature=temperature)
    items = payload if isinstance(payload, list) else []
    out = []
    for it in items[:n]:
        if isinstance(it, dict) and str(it.get("query", "")).strip():
            tool = it.get("tool") if it.get("tool") in ("search_papers", "search_snippets") else "search_papers"
            out.append({"tool": tool, "query": str(it["query"]).strip(),
                        "year_min": it.get("year_min"), "year_max": it.get("year_max"),
                        "venue": (it.get("venue") or None)})
    return out


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    found: bool
    n_calls: int
    found_via: str | None          # tool that surfaced the seed
    tool_seq: list[str]            # tools called, in order


def _corpus_ids(obs: Any) -> set[int]:
    """All paper corpus_ids referenced by a tool observation (handles lists of
    dataclass hits/refs with `corpus_id` or `paper_corpus_id`)."""
    out: set[int] = set()
    items = obs if isinstance(obs, (list, tuple)) else [obs]
    for it in items:
        for attr in ("corpus_id", "paper_corpus_id"):
            v = getattr(it, attr, None)
            if isinstance(v, int):
                out.add(v)
    return out


def make_find_policy(client: openai.AsyncOpenAI, model: str, tools: dict[str, Callable],
                     *, temperature: float = 0.3) -> Callable:
    """A find-the-paper policy (native function-calling), mirroring agent.policy but
    with the FIND objective and no submit_answer."""
    tool_specs = specs(tools)

    async def policy(query: str, turns: list[Turn]) -> PolicyStep:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": FIND_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        for i, t in enumerate(turns):
            if t.action is None:
                continue
            cid = f"call_{i}"
            messages.append({"role": "assistant", "content": t.thought or "",
                             "tool_calls": [{"id": cid, "type": "function",
                                             "function": {"name": t.action["name"],
                                                          "arguments": json.dumps(t.action.get("arguments", {}))}}]})
            messages.append({"role": "tool", "tool_call_id": cid,
                             "content": json.dumps(t.observation, ensure_ascii=False,
                                                   default=lambda o: dataclasses.asdict(o) if dataclasses.is_dataclass(o) else str(o))[:6000]})
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=tool_specs, tool_choice="auto",
            temperature=temperature, stream=False,
        )
        msg = resp.choices[0].message
        action = None
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            action = {"name": tc.function.name, "arguments": args}
        return PolicyStep(thought=msg.content or "", action=action,
                          prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)

    return policy


async def find_seed(question: str, seed_id: int, *, client: openai.AsyncOpenAI, model: str,
                    tools: dict[str, Callable], max_turns: int = 5, temperature: float = 0.3) -> RetrievalResult:
    """Run a ≤max_turns find session; stop early when the seed appears in a tool result."""
    policy = make_find_policy(client, model, tools, temperature=temperature)
    turns: list[Turn] = []
    seq: list[str] = []
    via: str | None = None
    for _ in range(max_turns):
        step = await policy(question, turns)
        if step.action is None:
            break
        name = step.action["name"]
        args = step.action.get("arguments", {}) or {}
        try:
            obs = await dispatch(name, args, tools)
        except Exception as exc:
            obs = {"error": str(exc)}
        turns.append(Turn(thought=step.thought, action=step.action, observation=obs))
        seq.append(name)
        if seed_id in _corpus_ids(obs):
            via = name
            break
    return RetrievalResult(found=via is not None, n_calls=len(seq), found_via=via, tool_seq=seq)
