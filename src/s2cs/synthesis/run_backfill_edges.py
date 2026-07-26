"""Seed the edge store from existing 2-hop QA pools. Old records persist
`evidence_a` + (intermediate, gold) but not `pointer_label`, so regenerate it
from the citing sentence + the cited paper's title/abstract.

    uv run python -m s2cs.synthesis.run_backfill_edges --pools data/qa/mh100_detail_cue \
        --edge-store data/qa/edges_mh/edges.jsonl

Needs OPENAI_BASE_URL (+ key if required) and S2CS_PAPERS_DB.
"""

import asyncio
import dataclasses
import json
import logging
import os
from pathlib import Path

import duckdb
import openai
import tyro
from dotenv import load_dotenv

from s2cs.agent.llm import chat_json
from s2cs.synthesis.edge_store import Edge, append_edges, load_edges

log = logging.getLogger(__name__)

POINTER_PROMPT = """A sentence from paper A cites another work B (below). Write a back-reference to B THROUGH A's relationship to it — the ROLE B plays for A in this passage (e.g. 'the baseline A compares against', 'the dataset A reuses', 'the method A adapts'). Do NOT restate B's own task, method, or findings, and do NOT use B's title or name.

[SENTENCE FROM A]
{citing_evidence}

[B]
Title: {to_title}
Abstract: {to_abstract}

Return ONLY a JSON object (no prose): {{"pointer_label": "..."}}"""


async def regenerate_pointer_label(client, model, *, citing_evidence, to_title, to_abstract):
    out = await chat_json(client, model, [{"role": "user", "content": POINTER_PROMPT.format(
        citing_evidence=citing_evidence, to_title=to_title or "",
        to_abstract=(to_abstract or "")[:600])}], temperature=0.0)
    if not isinstance(out, dict):
        return None
    return str(out.get("pointer_label", "")).strip() or None


@dataclasses.dataclass(frozen=True)
class BackfillArgs:
    pools: list[Path]                    # pool dirs holding multi_hop.jsonl
    edge_store: Path = Path("data/qa/edges_mh/edges.jsonl")
    papers_db: Path | None = None
    model: str = "glm-5.2-nvfp4"
    base_url: str | None = None
    concurrency: int = 6
    max_retries: int = 8


def _title_abstract(db: Path, cid: int) -> tuple[str | None, str | None]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute("SELECT title, abstract FROM papers_text WHERE corpus_id = ?", [cid]).fetchone()
    finally:
        con.close()
    return (row[0], row[1]) if row else (None, None)


async def _amain(args: BackfillArgs) -> None:
    load_dotenv()
    db = args.papers_db or Path(os.environ["S2CS_PAPERS_DB"])
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url is None:
        raise RuntimeError("OPENAI_BASE_URL not set (pass --base-url or set it in env / .env)")
    client = openai.AsyncOpenAI(base_url=base_url,
                                api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                max_retries=args.max_retries)
    args.edge_store.parent.mkdir(parents=True, exist_ok=True)
    seen = {(f, e.to_id) for f, lst in load_edges(args.edge_store).items() for e in lst}

    # collect candidate (from, to, citing_evidence) from the pools, skipping already-stored edges
    cands: list[tuple[int, int, str]] = []
    cand_keys: set[tuple[int, int]] = set()
    for pool in args.pools:
        f = pool / "multi_hop.jsonl"
        if not f.exists():
            log.warning("no multi_hop.jsonl in %s", pool)
            continue
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            frm, to = int(r["intermediate_paper_id"]), int(r["gold_paper_id"])
            ev = (r.get("evidence_a") or "").strip()
            key = (frm, to)
            if not ev or key in seen or key in cand_keys:
                continue
            cand_keys.add(key)
            cands.append((frm, to, ev))

    log.info("backfill: %d new candidate edges from %d pool(s)", len(cands), len(args.pools))
    sem = asyncio.Semaphore(args.concurrency)

    async def make_edge(frm: int, to: int, ev: str) -> Edge | None:
        async with sem:
            title, abstract = await asyncio.to_thread(_title_abstract, db, to)
            label = await regenerate_pointer_label(client, args.model,
                                                   citing_evidence=ev, to_title=title, to_abstract=abstract)
        return Edge(frm, to, ev, label) if label else None

    edges = [e for e in await asyncio.gather(*[make_edge(*c) for c in cands]) if e is not None]
    n = append_edges(args.edge_store, edges, seen)
    log.info("backfill done: wrote %d edges -> %s", n, args.edge_store)


def main(args: BackfillArgs) -> None:
    asyncio.run(_amain(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BackfillArgs))
