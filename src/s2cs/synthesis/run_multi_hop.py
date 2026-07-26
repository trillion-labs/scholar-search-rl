"""Drive multi-hop citation-bridge QA synthesis over the corpus, one answer_type per run.

SELECT picks a citing passage in A and the target B from A's in-corpus cited papers;
GROUND loads B's text and writes the question + B-grounded answer; the anti-shortcut PROBE
then drops any QA whose question already surfaces B via the agent's own retrieval (so the
hop can't be skipped). Generation is DuckDB + LLM only, but the probe uses the live env
(Milvus + BGE-M3) like run_paper_set's freeze — pass `--devices cpu` for a slurm-less CPU
run, or `--no-probe` to skip it (DuckDB + LLM only). Seeds (= the intermediate paper A) are
drawn from the well-connected tail of the citation graph (out-degree >= --min-out-degree),
reusing the paper_set seed selector. Streaming + resumable per seed A.

    uv run python -m s2cs.synthesis.run_multi_hop --answer-type value --n-papers 200 --devices cpu

Needs OPENAI_BASE_URL (+ OPENAI_API_KEY if required), S2CS_PAPERS_DB / S2CS_EDGES_PATH, and
(when probing) S2CS_MILVUS_URI.
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
from s2cs.synthesis.chain import build_chain
from s2cs.synthesis.cost import CostTrackingClient
from s2cs.synthesis.edge_store import append_edges, load_edges
from s2cs.synthesis.multi_hop import (
    ANSWER_TYPES,
    ground_chain,
    make_anchor,
    make_edge_discoverer,
    probe_chain,
)
from s2cs.synthesis.run_paper_set import _done_ids, _load_seed_rows, _seed_ids_by_degree

log = logging.getLogger(__name__)

QA_FILE = "multi_hop.jsonl"
ATTEMPTED_FILE = "multi_hop.attempted"


def _attach_cited_with_abstract(db: Path, edges_path: Path, papers: list[dict], *, limit: int) -> list[dict]:
    """For each seed A, attach `cited`: its in-corpus cited papers as
    {corpus_id, title, abstract}, lowest-citationcount first, top `limit`. Drop seeds
    with no in-corpus cited paper. (Like run_single_hop._attach_cited but adds abstract,
    which the SELECT stage needs to match the citing passage to a cited work.)"""
    if not papers:
        return []
    ids = [int(p["corpus_id"]) for p in papers]
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='duckdb_s2cs_')}'")
    try:
        rows = con.execute(
            f"SELECT e.src, e.dst, t.title, t.abstract "
            f"FROM read_parquet('{edges_path}') e "
            f"JOIN papers_text t ON t.corpus_id = e.dst "
            f"JOIN papers_meta m ON m.corpus_id = e.dst "
            f"WHERE e.src IN ({','.join('?' * len(ids))}) "
            f"AND t.title IS NOT NULL AND length(t.title) > 0 "
            f"AND t.abstract IS NOT NULL AND length(t.abstract) > 0 "
            f"ORDER BY e.src, m.citationcount ASC",
            ids,
        ).fetchall()
    finally:
        con.close()
    by_src: dict[int, list[dict]] = {}
    for src, dst, title, abstract in rows:
        lst = by_src.setdefault(int(src), [])
        if len(lst) < limit:
            lst.append({"corpus_id": int(dst), "title": title, "abstract": abstract})
    out = []
    for p in papers:
        cited = by_src.get(int(p["corpus_id"]))
        if cited:
            out.append({**p, "cited": cited})
    return out


def _fetch_b_text(db: Path, cid: int) -> tuple[str | None, str | None, str] | None:
    """Load (title, abstract, body) for a target paper B; None if absent or body-less."""
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute(
            "SELECT title, abstract, body FROM papers_text WHERE corpus_id = ?", [cid]
        ).fetchone()
    finally:
        con.close()
    if not row or not (row[2] or "").strip():
        return None
    return row[0], row[1], row[2]


@dataclasses.dataclass(frozen=True)
class MultiHopSynthArgs:
    papers_db: Path | None = None
    edges_path: Path | None = None
    out_dir: Path = Path("data/qa/multi_hop")
    model: str = "deepseek-v4-pro"
    answer_type: str = "value"          # value | abstract | identity
    a_anchor: str = "detail_cue"        # how to identify A: named | detail_cue | paraphrastic | content_conjunction
    hops: int = 2                       # chain length in NODES (K); 2 = single citation bridge (current)
    min_hops: int = 2                   # keep chains of at least this many nodes (mixed depth if hops > min_hops)
    edge_store: Path | None = None      # reusable edge store; defaults to out_dir/edges.jsonl
    temperature: float = 0.7
    sample_seed: int | None = None
    base_offset: int = 0
    base_url: str | None = None
    n_papers: int | None = None
    n_qa: int | None = None
    drop_review_survey: bool = False    # surveys are GOOD intermediates A (gold = a cited B)
    min_out_degree: int = 8             # A must cite >= this many in-corpus papers
    max_out_degree: int = 60            # ... and <= this many (bounds SELECT context + trajectory length)
    cited_limit: int = 20               # in-corpus cited papers shown to SELECT
    min_body_chars: int = 2_000
    max_body_chars: int | None = 24_000
    max_b_body_chars: int | None = 16_000
    page_size: int = 200
    concurrency: int = 4
    max_retries: int = 8
    # anti-shortcut probe (faithful: the agent's own Milvus retrieval) — needs live Milvus + encoder
    probe: bool = True                  # drop QA whose question directly surfaces B (shortcuttable hop)
    probe_n_queries: int = 5            # question->query reformulations to try
    probe_k: int = 15                   # hits retrieved per query (for A reachability + B rank)
    probe_b_rank: int = 3               # b_found only if B lands at best rank <= this (top hit = real shortcut)
    query_model: str | None = None      # probe query-gen model (defaults to --model)
    milvus_uri: str | None = None
    devices: list[str] | None = None    # encoder GPUs, e.g. ["cuda:0"]; ["cpu"] for a slurm-less CPU run


async def _amain(args: MultiHopSynthArgs) -> None:
    load_dotenv()
    if args.answer_type not in ANSWER_TYPES:
        raise ValueError(f"--answer-type must be one of {sorted(ANSWER_TYPES)}")
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
    discover_edge = make_edge_discoverer(
        client, args.model, max_body_chars=args.max_body_chars, temperature=args.temperature)
    anchor = make_anchor(
        client, args.model, a_anchor=args.a_anchor,
        max_body_chars=args.max_body_chars, temperature=args.temperature)
    query_model = args.query_model or args.model

    store_path = args.edge_store or (args.out_dir / "edges.jsonl")
    store = load_edges(store_path)
    seen_edges = {(f, e.to_id) for f, lst in store.items() for e in lst}

    def _node_paper(node_id: int) -> dict | None:
        rows = _load_seed_rows(db, [node_id], min_body_chars=args.min_body_chars, drop_review_survey=False)
        rows = _attach_cited_with_abstract(db, edges_path, rows, limit=args.cited_limit)
        return rows[0] if rows else None

    async def discover(node_id: int):
        paper = await asyncio.to_thread(_node_paper, node_id)
        return await discover_edge(paper) if paper else None

    # The probe uses the agent's real retrieval, so it needs the live env (Milvus + encoder),
    # like run_paper_set's freeze. Build it only when probing; --no-probe stays DuckDB+LLM only.
    tools = None
    if args.probe:
        if args.devices == ["cpu"]:
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

    sem = asyncio.Semaphore(args.concurrency)
    done = _done_ids(args.out_dir, QA_FILE, ATTEMPTED_FILE)
    seed_ids = _seed_ids_by_degree(edges_path, min_out_degree=args.min_out_degree, sample_seed=args.sample_seed)
    log.info("resume: %d already attempted; seed pool: %d papers cite >= %d in-corpus works",
             len(done), len(seed_ids), args.min_out_degree)
    kept = 0

    with (args.out_dir / QA_FILE).open("a") as qa_fh, (args.out_dir / ATTEMPTED_FILE).open("a") as att_fh:

        async def handle(paper: dict) -> None:
            nonlocal kept
            cid = int(paper["corpus_id"])
            async with sem:
                try:
                    chain = await build_chain(cid, hops=args.hops, min_hops=args.min_hops,
                                              store=store, discover=discover)
                    qa = None
                    if chain:
                        a_cue = await anchor(paper)
                        b = _fetch_b_text(db, chain[-1].to_id)  # terminal text
                        if a_cue and b is not None:
                            b_title, b_abstract, b_body = b
                            qa = await ground_chain(
                                a_cue, chain, b_title=b_title, b_abstract=b_abstract, b_body=b_body,
                                answer_type=args.answer_type, client=client, model=args.model,
                                max_body_chars=args.max_b_body_chars, temperature=args.temperature,
                            )
                            if qa is not None:
                                qa = dataclasses.replace(qa, anchor=args.a_anchor)
                except Exception as exc:
                    log.warning("multi_hop synth failed for corpus_id=%s: %s", cid, exc)
                    qa = None
            if qa is not None and tools is not None:
                try:
                    start_found, terminal_found = await probe_chain(
                        qa.question, qa.path[0], qa.path[-1],
                        search_papers=tools.search_papers, search_snippets=tools.search_snippets,
                        client=client, query_model=query_model,
                        n_queries=args.probe_n_queries, k=args.probe_k, b_rank=args.probe_b_rank,
                    )
                except Exception as exc:
                    log.warning("retrievability check failed for %s (keeping QA): %s", qa.qa_id, exc)
                    start_found, terminal_found = True, False
                if terminal_found or not start_found:
                    log.info("multi_hop %s dropped: start_found=%s terminal_found=%s (need start findable, terminal not)",
                             qa.qa_id, start_found, terminal_found)
                    qa = None
            if qa is not None:
                qa_fh.write(json.dumps(qa.to_record(), ensure_ascii=False) + "\n")
                qa_fh.flush()
                append_edges(store_path, qa.edges, seen_edges)  # persist discovered edges for reuse
                for e in qa.edges:
                    store.setdefault(e.from_id, []).append(e)
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
            rows = _attach_cited_with_abstract(db, edges_path, rows, limit=args.cited_limit)
            await asyncio.gather(*[handle(p) for p in rows])
            log.info("page @%d: %d ids (%d new, %d usable) -> %d multi_hop QA kept this run",
                     pos - len(page_ids), len(page_ids), len(new_ids), len(rows), kept)
            if args.n_qa is not None and kept >= args.n_qa:
                break

    log.info("done: %d multi_hop QA kept this run -> %s", kept, args.out_dir / QA_FILE)
    if client.calls_with_cost:
        log.info("LLM API spend this run: $%.4f over %d/%d calls reporting cost",
                 client.total_cost, client.calls_with_cost, client.calls)


def main(args: MultiHopSynthArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(MultiHopSynthArgs))
