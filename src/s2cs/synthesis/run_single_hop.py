"""Drive single-hop QA synthesis over the corpus, one format mode per run.

Reads body-bearing papers from `papers.duckdb` and generates QA + evidence with
`google/gemini-3.1-flash-lite` (override with --model) for one (--anchor,
--answer-type) mode. Run once per mode (own --out-dir).

**Streaming + resumable per item.** Each QA is appended to `single_hop.jsonl` and
flushed the moment it is generated; every attempted `corpus_id` is appended to
`single_hop.attempted`. On restart, both files (and any legacy `single_hop_*.jsonl`
shards) are read to skip already-done papers — so an interrupted run resumes with
no LLM re-spend and at most the in-flight items lost. Papers are *loaded* in pages
only to bound memory and amortize the DuckDB sort; the page size is NOT the
checkpoint granularity (checkpointing is per item).

This is **pure generation** — policy-agnostic. Closed-book + pass@8 difficulty are
*policy-relative* and run later at the training-prep stage against the actual
policy (not here, not with the generator). Intrinsic quality (rubric) is its own
pass over the generated jsonl.

    uv run python -m s2cs.synthesis.run_single_hop --n-papers 200
    uv run python -m s2cs.synthesis.run_single_hop --anchor describe --answer-type identity --out-dir data/qa/sh_describe_identity

Needs `OPENAI_BASE_URL` (and `OPENAI_API_KEY` if the endpoint requires one) in
the environment or `.env`; `S2CS_PAPERS_DB` points at the DuckDB.
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

from s2cs.synthesis.cost import CostTrackingClient
from s2cs.synthesis.single_hop import CITATION_ANCHORS, make_single_hop_synth

log = logging.getLogger(__name__)

QA_FILE = "single_hop.jsonl"
ATTEMPTED_FILE = "single_hop.attempted"

# Survey/review/editorial/opinion papers report third-party facts by design → their
# QA breach the "own-work" property. Exclude them as seeds (papers_meta.classification).
DROP_CLASS_RE = r"review|survey|editorial|opinion|comment|perspective|position|correction|erratum|errata|retraction"


@dataclasses.dataclass(frozen=True)
class SingleHopSynthArgs:
    papers_db: Path | None = None
    edges_path: Path | None = None   # required for relational_conjunction / full_span
    out_dir: Path = Path("data/qa/single_hop")
    model: str = "google/gemini-3.1-flash-lite"
    anchor: str = "named"        # named | describe | conjunction (one mode per run / out_dir)
    answer_type: str = "value"   # value | identity
    temperature: float = 0.3     # generation temp; downstream judges stay at 0.0
    sample_seed: int | None = None  # if set, ORDER BY hash(corpus_id + seed) = reproducible shuffle
    base_offset: int = 0         # start offset into the (shuffled) list — set per mode for disjoint samples
    base_url: str | None = None
    n_papers: int | None = None
    n_qa: int | None = None      # stop once this many QA are kept THIS run (with n_papers as a safety cap)
    drop_review_survey: bool = True   # exclude survey/review/editorial/opinion seeds (own-work hazard)
    min_body_chars: int = 2_000
    max_body_chars: int | None = None
    page_size: int = 500         # papers loaded per DuckDB query (memory + sort amortization), NOT a checkpoint unit
    concurrency: int = 4
    max_retries: int = 8


def _done_ids(out_dir: Path) -> set[int]:
    """corpus_ids already attempted — from the QA jsonl, the .attempted log, and
    any legacy shard files. Resume skips these (no LLM re-spend)."""
    done: set[int] = set()
    for path in [out_dir / QA_FILE, *sorted(out_dir.glob("single_hop_*.jsonl"))]:
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        done.update(json.loads(line).get("seed_paper_ids", []))
    att = out_dir / ATTEMPTED_FILE
    if att.exists():
        with att.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(int(line))
    return done


def _load_page(db: Path, *, min_body_chars: int, offset: int, limit: int, sample_seed: int | None,
               drop_review_survey: bool) -> list[dict]:
    # sample_seed → deterministic shuffle (hash order). Same seed + disjoint offsets
    # across modes ⇒ disjoint random samples. None → sequential corpus_id order.
    order = "hash(t.corpus_id + ?)" if sample_seed is not None else "t.corpus_id"
    params: list = [min_body_chars]
    if sample_seed is not None:
        params.append(sample_seed)
    params += [limit, offset]
    class_filter = (
        f" AND NOT regexp_matches(lower(coalesce(m.classification, '')), '{DROP_CLASS_RE}')"
        if drop_review_survey else ""
    )
    con = duckdb.connect(str(db), read_only=True)
    # The hash-order sort spills to a temp dir; the read-only DB's default is shared
    # (`<db>.tmp/` beside the file), so concurrent runs on one node collide there. Give
    # each process its own temp dir.
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='duckdb_s2cs_')}'")
    try:
        rows = con.execute(
            f"SELECT t.corpus_id, t.title, t.abstract, t.summary, t.body, m.year, m.venue "
            f"FROM papers_text t JOIN papers_meta m ON t.corpus_id = m.corpus_id "
            f"WHERE length(t.body) >= ?{class_filter} ORDER BY {order} LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, row)) for row in rows]


def _attach_cited(db: Path, edges_path: Path, papers: list[dict], *, limit: int) -> list[dict]:
    """For each seed, attach `cited`: its in-corpus cited papers (lowest citationcount
    first, top `limit`, titled only). Drop seeds with no in-corpus cited paper — the
    two citation anchors (T2c/T2d) cannot form a citation cue without one."""
    if not papers:
        return []
    ids = [int(p["corpus_id"]) for p in papers]
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='duckdb_s2cs_')}'")
    try:
        rows = con.execute(
            f"SELECT e.src, e.dst, t.title, m.year "
            f"FROM read_parquet('{edges_path}') e "
            f"JOIN papers_text t ON t.corpus_id = e.dst "
            f"JOIN papers_meta m ON m.corpus_id = e.dst "
            f"WHERE e.src IN ({','.join('?' * len(ids))}) "
            f"AND t.title IS NOT NULL AND length(t.title) > 0 "
            f"ORDER BY e.src, m.citationcount ASC",
            ids,
        ).fetchall()
    finally:
        con.close()
    by_src: dict[int, list[dict]] = {}
    for src, dst, title, year in rows:
        lst = by_src.setdefault(int(src), [])
        if len(lst) < limit:
            lst.append({"corpus_id": int(dst), "title": title, "year": year})
    out = []
    for p in papers:
        cited = by_src.get(int(p["corpus_id"]))
        if cited:
            out.append({**p, "cited": cited})
    return out


async def _amain(args: SingleHopSynthArgs) -> None:
    load_dotenv()
    db = args.papers_db or Path(os.environ["S2CS_PAPERS_DB"])
    edges_path = None
    if args.anchor in CITATION_ANCHORS:
        edges_path = args.edges_path or (Path(os.environ["S2CS_EDGES_PATH"]) if os.environ.get("S2CS_EDGES_PATH") else None)
        if edges_path is None:
            raise RuntimeError(f"anchor {args.anchor!r} needs --edges-path or S2CS_EDGES_PATH")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url is None:
        raise RuntimeError("OPENAI_BASE_URL not set (pass --base-url or set it in env / .env)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = CostTrackingClient(openai.AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        max_retries=args.max_retries,
    ))
    synth = make_single_hop_synth(
        client, args.model, anchor=args.anchor, answer_type=args.answer_type,
        max_body_chars=args.max_body_chars, temperature=args.temperature,
    )
    sem = asyncio.Semaphore(args.concurrency)

    done = _done_ids(args.out_dir)
    log.info("resume: %d papers already attempted in %s", len(done), args.out_dir)
    kept = 0

    with (args.out_dir / QA_FILE).open("a") as qa_fh, (args.out_dir / ATTEMPTED_FILE).open("a") as att_fh:

        async def handle(paper: dict) -> None:
            nonlocal kept
            cid = int(paper["corpus_id"])
            async with sem:
                try:
                    qas = await synth(paper)
                except Exception as exc:
                    log.warning("generation failed for corpus_id=%s: %s", cid, exc)
                    qas = []
            for qa in qas:
                qa_fh.write(json.dumps(qa.to_record(), ensure_ascii=False) + "\n")
            if qas:
                qa_fh.flush()
                kept += len(qas)
            att_fh.write(f"{cid}\n")
            att_fh.flush()

        offset = args.base_offset
        while args.n_papers is None or (offset - args.base_offset) < args.n_papers:
            limit = args.page_size
            if args.n_papers is not None:
                limit = min(limit, args.n_papers - (offset - args.base_offset))
            page = _load_page(db, min_body_chars=args.min_body_chars,
                              offset=offset, limit=limit, sample_seed=args.sample_seed,
                              drop_review_survey=args.drop_review_survey)
            if not page:
                break
            # Advance offset by rows actually READ — _attach_cited drops non-citing seeds,
            # so its (shorter) output must not drive paging or the n_papers scan cap.
            offset += len(page)
            seeds = _attach_cited(db, edges_path, page, limit=8) if edges_path is not None else page
            todo = [p for p in seeds if int(p["corpus_id"]) not in done]
            await asyncio.gather(*[handle(p) for p in todo])
            log.info("page @%d: %d read, %d seeds (%d new) -> %d QA kept this run",
                     offset - len(page), len(page), len(seeds), len(todo), kept)
            if args.n_qa is not None and kept >= args.n_qa:
                break

    log.info("done: %d QA kept this run -> %s", kept, args.out_dir / QA_FILE)
    if client.calls_with_cost:
        log.info("LLM API spend this run: $%.4f over %d/%d calls reporting cost",
                 client.total_cost, client.calls_with_cost, client.calls)


def main(args: SingleHopSynthArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(SingleHopSynthArgs))
