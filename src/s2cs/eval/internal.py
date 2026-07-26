"""Internal in-domain sanity evals (Tier S): does the policy use *our* env well?

Unlike AstaBench (Asta MCP tools, Asta corpus — a transfer measurement), these run
the agent over the **same local env tools and s2orc-cs corpus it trains on**, so a
base-vs-checkpoint gap here is in-domain learning, not transfer. This is the lens
that separates "didn't learn" from "didn't transfer".

Two sub-evals (eval/README.md contract):
- **known_item**: query = a paper's title → recover that paper's corpus_id (recall@10).
  Sanity that retrieval works at all on our corpus/tools.
- **citation_holdout**: query = a paper's title+abstract → recover a held-out sample
  of the papers it actually cites (recall@50). Tests whether the policy drives
  search + the citation-graph tools (list_references / find_similar) to surface
  genuinely related work.

Both score a ranked corpus_id list via the same recall@k machinery as LitSearch
(reused from `eval/litsearch.py`), submitted through `submit_papers`.
"""

import dataclasses
import logging
import os
import random
from pathlib import Path
from typing import Callable

import duckdb
from dotenv import load_dotenv

from s2cs.eval.litsearch import calculate_ndcg, calculate_recall, extract_ranked_ids
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult
from s2cs.eval.submit_papers import make_submit_papers

log = logging.getLogger(__name__)

RETRIEVAL_TOOLS = ["search_papers", "search_snippets", "list_references", "list_citations", "find_similar", "paper_info"]

KNOWN_ITEM_TEMPLATE = """Find the single paper in the corpus whose title is given below.

Title: {query}

Use `search_papers` to locate it, then call `submit_papers` with `paper_ids` = the corpus_ids of your best matches, most likely first (up to {top_k})."""

HOLDOUT_TEMPLATE = """Below is a paper's title and abstract. Find the papers in the corpus that this paper most likely *cites* (its references / closely related prior work).

{query}

Use `search_papers`, `find_similar`, and `list_references` / `list_citations` to gather candidates, then call `submit_papers` with `paper_ids` = the corpus_ids of the most likely cited papers, most relevant first (up to {top_k})."""


@dataclasses.dataclass(frozen=True)
class Sample:
    sample_id: str
    query: str
    gold: list[int]


def _db_path() -> str:
    load_dotenv()
    p = os.environ.get("S2CS_PAPERS_DB")
    if not p:
        raise RuntimeError("S2CS_PAPERS_DB not set (.env)")
    return p


def _edges_path() -> str:
    load_dotenv()
    p = os.environ.get("S2CS_EDGES_PATH")
    if not p:
        raise RuntimeError("S2CS_EDGES_PATH not set (.env)")
    return p


def load_known_item(n: int = 100, seed: int = 0) -> list[Sample]:
    """Deterministically sample papers with a usable title; query=title, gold=self."""
    con = duckdb.connect(_db_path(), read_only=True)
    # length>30 + reject generic boilerplate titles (Editorial / Original Article / …)
    # that many papers share — they make ambiguous known-item queries.
    rows = con.execute("""
        SELECT corpus_id, title FROM papers_text
        WHERE title IS NOT NULL AND length(title) > 30
          AND lower(title) NOT IN ('original article', 'editorial', 'introduction',
              'correction', 'erratum', 'review', 'abstract', 'preface', 'foreword')
        ORDER BY hash(corpus_id + ?) LIMIT ?
    """, [seed, n]).fetchall()
    return [Sample(sample_id=f"ki_{cid}", query=title, gold=[int(cid)]) for cid, title in rows]


def load_citation_holdout(n: int = 100, k: int = 5, min_refs: int = 12, seed: int = 0) -> list[Sample]:
    """Sample papers with >= min_refs in-corpus references; query=title+abstract,
    gold = a deterministic k-sample of those references."""
    con = duckdb.connect(_db_path(), read_only=True)
    edges = _edges_path()
    # in-corpus citation edges (both endpoints have text)
    con.execute(f"""
        CREATE TEMP VIEW refs AS
        SELECT e.src AS src, e.dst AS dst
        FROM read_parquet('{edges}') e
        WHERE e.src IN (SELECT corpus_id FROM papers_text WHERE title IS NOT NULL AND abstract IS NOT NULL)
          AND e.dst IN (SELECT corpus_id FROM papers_text)
    """)
    srcs = con.execute("""
        SELECT src FROM refs GROUP BY src HAVING count(*) >= ?
        ORDER BY hash(src + ?) LIMIT ?
    """, [min_refs, seed, n]).fetchall()

    samples: list[Sample] = []
    for (src,) in srcs:
        src = int(src)
        refs = [int(r[0]) for r in con.execute("SELECT dst FROM refs WHERE src = ?", [src]).fetchall()]
        if len(refs) < k:
            continue
        rng = random.Random(seed * 1_000_003 + src)
        gold = rng.sample(refs, k)
        meta = con.execute("SELECT title, abstract FROM papers_text WHERE corpus_id = ?", [src]).fetchone()
        title, abstract = (meta or ("", ""))
        query = f"Title: {title or ''}\nAbstract: {abstract or ''}"
        samples.append(Sample(sample_id=f"ch_{src}", query=query, gold=gold))
    return samples


def _build_local_tools(model_name: str | None, devices: list[str] | None) -> dict[str, Callable]:
    from s2cs.env import build_tools

    t = build_tools(model_name=model_name, devices=devices)
    tools = t.subset(RETRIEVAL_TOOLS)
    tools["submit_papers"] = make_submit_papers()
    return tools


def _score(trajs, samples: list[Sample], k_values: tuple[int, ...]) -> dict[str, float]:
    preds = [(extract_ranked_ids(t), s.gold) for t, s in zip(trajs, samples)]
    n = len(preds) or 1
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(calculate_recall(r[:k], g) for r, g in preds) / n
        out[f"ndcg@{k}"] = sum(calculate_ndcg(r[:k], g) for r, g in preds) / n
    return out


def _run(
    samples: list[Sample],
    template: str,
    k_values: tuple[int, ...],
    *,
    base_url: str,
    model: str,
    api_key: str,
    model_name: str | None,
    devices: list[str] | None,
    max_turns: int,
    temperature: float,
    concurrency: int,
    top_k: int,
    trajectory_dir: str | None,
    chat_format: str = "qwen",
) -> BenchResult:
    tools = _build_local_tools(model_name, devices)
    prompts = [template.format(query=s.query, top_k=top_k) for s in samples]
    ids = [s.sample_id for s in samples]

    import asyncio

    trajs = asyncio.run(run_rollouts(
        prompts, base_url=base_url, model=model, tools=tools, api_key=api_key,
        max_turns=max_turns, temperature=temperature, concurrency=concurrency,
        trajectory_dir=trajectory_dir, sample_ids=ids, chat_format=chat_format,
    ))
    return BenchResult(metrics=_score(trajs, samples, k_values), n=len(samples))


def run_known_item(*, base_url, model, api_key="EMPTY", model_name=None, devices=None,
                   limit=100, seed=0, max_turns=40, temperature=0.7, concurrency=16,
                   top_k=10, trajectory_dir=None, chat_format="qwen") -> BenchResult:
    samples = load_known_item(n=limit, seed=seed)
    log.info("known_item: %d samples", len(samples))
    return _run(samples, KNOWN_ITEM_TEMPLATE, (5, 10), base_url=base_url, model=model,
                api_key=api_key, model_name=model_name, devices=devices, max_turns=max_turns,
                temperature=temperature, concurrency=concurrency, top_k=top_k,
                trajectory_dir=trajectory_dir, chat_format=chat_format)


def run_citation_holdout(*, base_url, model, api_key="EMPTY", model_name=None, devices=None,
                         limit=100, seed=0, k=5, min_refs=12, max_turns=40, temperature=0.7,
                         concurrency=16, top_k=50, trajectory_dir=None, chat_format="qwen") -> BenchResult:
    samples = load_citation_holdout(n=limit, k=k, min_refs=min_refs, seed=seed)
    log.info("citation_holdout: %d samples (k=%d held-out refs each)", len(samples), k)
    return _run(samples, HOLDOUT_TEMPLATE, (20, 50), base_url=base_url, model=model,
                api_key=api_key, model_name=model_name, devices=devices, max_turns=max_turns,
                temperature=temperature, concurrency=concurrency, top_k=top_k,
                trajectory_dir=trajectory_dir, chat_format=chat_format)
