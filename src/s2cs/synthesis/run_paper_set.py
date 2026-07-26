"""Drive paper_set QA synthesis over the corpus, one anchor per run.

Unlike single-hop generation (generator + DuckDB only), paper_set freezes its gold
set against the **live env**: it searches the corpus for each generated question and
judges candidates against the criteria. So this driver wires `env.build_tools()`
(Milvus + BGE-M3 encoder + reader) and runs on a GPU node with Milvus up — like
`run_difficulty` / `run_rollout_check`, not like `run_single_hop`.

Seeds are drawn from the **well-connected tail** of the citation graph — papers with
at least `--min-out-degree` in-corpus citations — so each seed comes with a real
topical family (its cited papers) to ground the criterion in. The citation graph is
sparse overall (~35% coverage) but the high-degree tail is plentiful (~52k papers
cite ≥5 in-corpus works). Per seed: generate (question + criteria, conditioned on the
seed's cited papers) -> freeze (search + cited-union + judge -> gold set) ->
size-gate -> write. Streaming + resumable per item.

    uv run python -m s2cs.synthesis.run_paper_set --n-papers 200
    uv run python -m s2cs.synthesis.run_paper_set --min-out-degree 8 --out-dir data/qa/ps_d8

Needs `OPENAI_BASE_URL` (+ `OPENAI_API_KEY` if required) for the generator/judge,
and the standard env keys (`S2CS_MILVUS_URI`, `S2CS_PAPERS_DB`, `S2CS_EDGES_PATH`).
"""

import asyncio
import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path

import duckdb
import openai
import tyro
from dotenv import load_dotenv

from s2cs.env import build_tools
from s2cs.synthesis.cost import CostTrackingClient
from s2cs.synthesis.paper_set import build_gold_set, make_paper_set_synth
from s2cs.synthesis.run_single_hop import DROP_CLASS_RE, _attach_cited

log = logging.getLogger(__name__)

QA_FILE = "paper_set.jsonl"
ATTEMPTED_FILE = "paper_set.attempted"


@dataclasses.dataclass(frozen=True)
class PaperSetSynthArgs:
    papers_db: Path | None = None
    edges_path: Path | None = None
    out_dir: Path = Path("data/qa/paper_set")
    model: str = "deepseek-v4-pro"     # generator + relevance judge
    query_model: str | None = None     # candidate-gather query model (defaults to --model)
    anchor: str = "content_conjunction"  # content_conjunction | context_conjunction
    temperature: float = 0.7
    sample_seed: int | None = None     # set -> hash-shuffle the seed list; None -> highest-degree first
    base_offset: int = 0
    base_url: str | None = None
    milvus_uri: str | None = None
    devices: list[str] | None = None   # encoder GPUs, e.g. ["cuda:0"]
    n_papers: int | None = None
    n_qa: int | None = None
    drop_review_survey: bool = True
    min_out_degree: int = 5            # seed must cite >= this many in-corpus papers (the family)
    cited_limit: int = 12             # in-corpus cited papers shown to the generator / unioned into candidates
    min_body_chars: int = 2_000
    max_body_chars: int | None = 24_000
    page_size: int = 200
    concurrency: int = 2               # seeds in flight (each fans out to many freeze judge calls)
    # freeze knobs
    n_queries: int = 5
    gather_k: int = 15
    pool_cap: int = 60
    judge_concurrency: int = 8
    min_size: int = 3
    max_size: int = 20
    max_retries: int = 8


def _done_ids(out_dir: Path, qa_file: str = QA_FILE, attempted_file: str = ATTEMPTED_FILE) -> set[int]:
    # qa_file/attempted_file are parameterized so run_multi_hop can resume off its OWN
    # files (multi_hop.jsonl/.attempted) instead of paper_set's defaults.
    done: set[int] = set()
    qa = out_dir / qa_file
    if qa.exists():
        with qa.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.update(json.loads(line).get("seed_paper_ids", []))
    att = out_dir / attempted_file
    if att.exists():
        with att.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(int(line))
    return done


def _seed_ids_by_degree(edges_path: Path, *, min_out_degree: int, sample_seed: int | None) -> list[int]:
    """corpus_ids that cite >= min_out_degree in-corpus papers, ordered highest-degree
    first (or hash-shuffled if sample_seed is set, for reproducible random sampling)."""
    order = "hash(src + ?)" if sample_seed is not None else "d DESC"
    params: list = [min_out_degree]
    if sample_seed is not None:
        params.append(sample_seed)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='duckdb_s2cs_')}'")
    try:
        rows = con.execute(
            f"SELECT src FROM (SELECT src, count(*) AS d FROM read_parquet('{edges_path}') "
            f"GROUP BY src HAVING count(*) >= ?) ORDER BY {order}",
            params,
        ).fetchall()
    finally:
        con.close()
    return [int(r[0]) for r in rows]


def _load_seed_rows(db: Path, ids: list[int], *, min_body_chars: int, drop_review_survey: bool) -> list[dict]:
    """Load paper rows for the given corpus_ids (body-bearing, non-review), preserving
    the input order."""
    if not ids:
        return []
    class_filter = (
        f" AND NOT regexp_matches(lower(coalesce(m.classification, '')), '{DROP_CLASS_RE}')"
        if drop_review_survey else ""
    )
    placeholders = ",".join("?" * len(ids))
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='duckdb_s2cs_')}'")
    try:
        rows = con.execute(
            f"SELECT t.corpus_id, t.title, t.abstract, t.summary, t.body, m.year, m.venue "
            f"FROM papers_text t JOIN papers_meta m ON t.corpus_id = m.corpus_id "
            f"WHERE t.corpus_id IN ({placeholders}) AND length(t.body) >= ?{class_filter}",
            ids + [min_body_chars],
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    by_id = {int(r[0]): dict(zip(cols, r)) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def _amain(args: PaperSetSynthArgs) -> None:
    load_dotenv()
    db = args.papers_db or Path(os.environ["S2CS_PAPERS_DB"])
    edges_path = args.edges_path or Path(os.environ["S2CS_EDGES_PATH"])
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url is None:
        raise RuntimeError("OPENAI_BASE_URL not set (pass --base-url or set it in env / .env)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = CostTrackingClient(openai.AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        max_retries=args.max_retries,
    ))
    if args.devices == ["cpu"]:
        # Slurm-less path: freeze is LLM-bound, not encoder-bound, so a CPU BGE-M3 is
        # fine and avoids contending for a GPU. build_tools' default encoder hardcodes
        # fp16 (GPU-only), so build a CPU one (fp16 off) and inject it.
        from FlagEmbedding import BGEM3FlagModel

        from s2cs.env.encoder import BatchedEncoder
        log.info("loading BGE-M3 on CPU (slurm-less run) ...")
        enc = BatchedEncoder(
            BGEM3FlagModel(os.environ.get("S2CS_MODEL", "BAAI/bge-m3"), use_fp16=False, devices=["cpu"]),
            max_batch=8, wait_ms=5,
        )
        tools = build_tools(encoder=enc, papers_db=db, edges_path=edges_path, milvus_uri=args.milvus_uri)
    else:
        tools = build_tools(milvus_uri=args.milvus_uri, papers_db=db, edges_path=edges_path, devices=args.devices)
    synth = make_paper_set_synth(
        client, args.model, anchor=args.anchor,
        max_body_chars=args.max_body_chars, temperature=args.temperature,
    )
    sem = asyncio.Semaphore(args.concurrency)

    done = _done_ids(args.out_dir)
    seed_ids = _seed_ids_by_degree(edges_path, min_out_degree=args.min_out_degree, sample_seed=args.sample_seed)
    log.info("resume: %d already attempted; seed pool: %d papers cite >= %d in-corpus works",
             len(done), len(seed_ids), args.min_out_degree)
    kept = 0

    with (args.out_dir / QA_FILE).open("a") as qa_fh, (args.out_dir / ATTEMPTED_FILE).open("a") as att_fh:

        async def handle(paper: dict) -> None:
            nonlocal kept
            cid = int(paper["corpus_id"])
            cited_ids = [int(c["corpus_id"]) for c in paper.get("cited", [])]
            async with sem:
                try:
                    qa = await synth(paper)
                    if qa is not None:
                        qa = await build_gold_set(
                            qa,
                            search_papers=tools.search_papers,
                            search_snippets=tools.search_snippets,
                            paper_info=tools.paper_info,
                            client=client, model=args.model, query_model=args.query_model,
                            seed_candidate_ids=cited_ids,
                            n_queries=args.n_queries, k=args.gather_k, pool_cap=args.pool_cap,
                            concurrency=args.judge_concurrency,
                            min_size=args.min_size, max_size=args.max_size,
                        )
                except Exception as exc:
                    log.warning("paper_set synth failed for corpus_id=%s: %s", cid, exc)
                    qa = None
            if qa is not None:
                qa_fh.write(json.dumps(qa.to_record(), ensure_ascii=False) + "\n")
                qa_fh.flush()
                kept += 1
            att_fh.write(f"{cid}\n")
            att_fh.flush()

        pos = args.base_offset
        scanned = 0
        while args.n_papers is None or scanned < args.n_papers:
            take = args.page_size
            if args.n_papers is not None:
                take = min(take, args.n_papers - scanned)
            page_ids = seed_ids[pos:pos + take]
            if not page_ids:
                break
            pos += len(page_ids)
            scanned += len(page_ids)
            new_ids = [i for i in page_ids if i not in done]
            rows = _load_seed_rows(db, new_ids, min_body_chars=args.min_body_chars,
                                   drop_review_survey=args.drop_review_survey)
            rows = _attach_cited(db, edges_path, rows, limit=args.cited_limit)
            await asyncio.gather(*[handle(p) for p in rows])
            log.info("page @%d: %d ids (%d new, %d usable) -> %d paper_set QA kept this run",
                     pos - len(page_ids), len(page_ids), len(new_ids), len(rows), kept)
            if args.n_qa is not None and kept >= args.n_qa:
                break

    log.info("done: %d paper_set QA kept this run -> %s", kept, args.out_dir / QA_FILE)
    if client.calls_with_cost:
        log.info("LLM API spend this run: $%.4f over %d/%d calls reporting cost",
                 client.total_cost, client.calls_with_cost, client.calls)


def main(args: PaperSetSynthArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(PaperSetSynthArgs))
