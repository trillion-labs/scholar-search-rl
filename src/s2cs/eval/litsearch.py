"""LitSearch adapter (EMNLP 2024, arXiv:2407.18940; princeton-nlp/LitSearch).

A paper-finding retrieval benchmark: 597 natural-language queries over a fixed
64,183-paper corpus, gold relevance given as a set of S2 `corpusid`s. We run our
agent over the corpus re-indexed into a dedicated Milvus collection
(`litsearch_papers`, built by `litsearch_corpus.py`) so the agent uses the same
`search_papers` tool it trains on, then score recall@k / ndcg@k against gold.

The corpus + gold-id scheme is faithful to the original; the recall@k / ndcg@k
metrics are ported verbatim from the repo's `utils/utils.py` (it ships the
primitives but no scoring driver). Paper headline k values: 5 and 20.
"""

import dataclasses
import logging
from typing import Any, Callable

from pymilvus import Collection

from s2cs.agent.trajectory import Trajectory
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult
from s2cs.eval.submit_papers import make_submit_papers, parse_submitted_ids

log = logging.getLogger(__name__)

LITSEARCH_COLLECTION = "litsearch_papers"
LITSEARCH_HF_ID = "princeton-nlp/LitSearch"
DEFAULT_K_VALUES = (5, 20)

TASK_TEMPLATE = """You are searching a corpus of computer-science papers to answer a literature-search query. Find the papers most relevant to the query below.

Query: {query}

Use the `search_papers` tool (you may call it several times, refining the query) to gather candidate papers. When you have enough, call `submit_papers` with `paper_ids` set to the corpus_ids of the most relevant papers, **ranked most-relevant first**, up to {top_k} ids. Submit only paper ids you actually retrieved."""


@dataclasses.dataclass(frozen=True)
class Query:
    sample_id: str
    text: str
    gold: list[int]


def calculate_recall(retrieved: list[int], relevant_docs: list[int]) -> float:
    """Ported verbatim from princeton-nlp/LitSearch utils/utils.py."""
    num_relevant_retrieved = len(set(retrieved).intersection(set(relevant_docs)))
    num_relevant = len(relevant_docs)
    return num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0


def calculate_ndcg(retrieved: list[int], relevant_docs: list[int]) -> float:
    """Ported verbatim from princeton-nlp/LitSearch utils/utils.py."""
    dcg = 0.0
    for idx, docid in enumerate(retrieved):
        if docid in relevant_docs:
            dcg += 1 / (idx + 1)
    idcg = sum(1 / (idx + 1) for idx in range(len(relevant_docs)))
    return dcg / idcg if idcg > 0 else 0.0


def load_queries(limit: int | None = None) -> list[Query]:
    from datasets import load_dataset

    ds = load_dataset(LITSEARCH_HF_ID, "query", split="full")
    queries: list[Query] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        queries.append(Query(
            sample_id=f"{row.get('query_set', 'q')}_{i}",
            text=row["query"],
            gold=[int(c) for c in row["corpusids"]],
        ))
    log.info("loaded %d LitSearch queries", len(queries))
    return queries


def extract_ranked_ids(traj: Trajectory) -> list[int]:
    """Recover the agent's ranked candidate list from a rollout.

    Primary signal is the `submit_papers` list (rides in `traj.answer` as a JSON
    array). Whether or not the agent submitted cleanly, we backfill with the
    corpus_ids it surfaced via `search_papers`, in first-seen order, so a model
    that searches well but submits sloppily still gets scored on what it found.
    """
    ranked: list[int] = []
    seen: set[int] = set()

    def add(cid: int) -> None:
        if cid not in seen:
            seen.add(cid)
            ranked.append(cid)

    for cid in parse_submitted_ids(traj.answer):
        add(cid)

    for turn in traj.turns:
        if not turn.action or turn.action.get("name") != "search_papers":
            continue
        obs = turn.observation
        if not isinstance(obs, list):
            continue
        for hit in obs:
            cid = getattr(hit, "corpus_id", None)
            if cid is None and isinstance(hit, dict):
                cid = hit.get("corpus_id")
            if cid is not None:
                add(int(cid))
    return ranked


def score(
    predictions: list[tuple[list[int], list[int]]],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Macro-averaged recall@k / ndcg@k over (retrieved, gold) pairs."""
    metrics: dict[str, float] = {}
    n = len(predictions)
    if n == 0:
        return metrics
    for k in k_values:
        recall = sum(calculate_recall(r[:k], g) for r, g in predictions) / n
        ndcg = sum(calculate_ndcg(r[:k], g) for r, g in predictions) / n
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg
    return metrics


def build_tools(
    *,
    milvus_uri: str,
    model_name: str,
    devices: list[str] | None = None,
) -> dict[str, Callable]:
    """Search surface over the LitSearch corpus + the paper-set submission."""
    from s2cs.env.encoder import BatchedEncoder
    from s2cs.env.milvus_client import MilvusEndpoint, connect
    from s2cs.env.tools.search_papers import make_search_papers

    connect(MilvusEndpoint(uri=milvus_uri))
    coll = Collection(LITSEARCH_COLLECTION)
    coll.load()

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_name, use_fp16=True, devices=devices or ["cuda:0"])
    enc = BatchedEncoder(model, max_batch=32, wait_ms=5)

    return {
        "search_papers": make_search_papers(coll, enc.encode_hybrid),
        "submit_papers": make_submit_papers(),
    }


def run(
    *,
    base_url: str,
    model: str,
    milvus_uri: str,
    model_name: str,
    api_key: str = "EMPTY",
    devices: list[str] | None = None,
    limit: int | None = None,
    max_turns: int = 40,
    temperature: float = 0.7,
    concurrency: int = 16,
    top_k: int = 20,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    trajectory_dir: str | None = None,
    chat_format: str = "qwen",
) -> BenchResult:
    queries = load_queries(limit)
    tools = build_tools(milvus_uri=milvus_uri, model_name=model_name, devices=devices)

    prompts = [TASK_TEMPLATE.format(query=q.text, top_k=top_k) for q in queries]
    sample_ids = [q.sample_id for q in queries]

    import asyncio

    trajs = asyncio.run(run_rollouts(
        prompts,
        base_url=base_url,
        model=model,
        tools=tools,
        api_key=api_key,
        max_turns=max_turns,
        temperature=temperature,
        concurrency=concurrency,
        trajectory_dir=trajectory_dir,
        sample_ids=sample_ids,
        chat_format=chat_format,
    ))

    predictions = [(extract_ranked_ids(t), q.gold) for t, q in zip(trajs, queries)]
    metrics = score(predictions, k_values)
    return BenchResult(metrics=metrics, n=len(queries))
