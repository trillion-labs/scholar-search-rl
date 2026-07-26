"""Build the `litsearch_papers` Milvus collection from the LitSearch corpus.

One-off ETL for the LitSearch eval (`litsearch.py`). Loads the 64,183-paper
`corpus_clean` config of `princeton-nlp/LitSearch`, embeds each paper's
title+abstract with BGE-M3 (dense + learned sparse, same as the live corpus),
and inserts into a dedicated collection keyed by the paper's `corpusid` so the
agent's `search_papers` tool retrieves over it unchanged.

64k papers fit comfortably in direct `Collection.insert` batches — no need for
the MinIO bulk_insert path the 1.12M-paper corpus uses.

Run via slurm: `sbatch slurm/build_litsearch_corpus.sbatch`.
"""

import dataclasses
import logging

import tyro
from pymilvus import Collection

from s2cs.eval.litsearch import LITSEARCH_COLLECTION, LITSEARCH_HF_ID
from s2cs.env.milvus_client import (
    HNSW_INDEX,
    SPARSE_INDEX,
    MilvusEndpoint,
    build_indexes,
    connect,
    ensure_collection,
    load,
    papers_schema,
)

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildArgs:
    milvus_uri: str = "http://localhost:19530"
    model_name: str = "BAAI/bge-m3"
    devices: str = "cuda:0"
    embed_batch: int = 256
    insert_batch: int = 1000


def _paper_text(row: dict) -> str:
    return f"Title: {row.get('title') or ''}\nAbstract: {row.get('abstract') or ''}"


def _sparse(weights: dict) -> dict[int, float]:
    """BGE-M3 lexical_weights → {token_id: weight}; Milvus rejects empties."""
    d = {int(k): float(v) for k, v in weights.items()}
    return d or {0: 1e-9}


def _embed_and_insert(coll: Collection, args: BuildArgs) -> None:
    from datasets import load_dataset
    from FlagEmbedding import BGEM3FlagModel

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    model = BGEM3FlagModel(args.model_name, use_fp16=True, devices=devices or ["cuda:0"])

    ds = load_dataset(LITSEARCH_HF_ID, "corpus_clean", split="full")
    log.info("embedding %d LitSearch corpus papers", len(ds))

    pending: list[dict] = []
    total = 0
    for start in range(0, len(ds), args.embed_batch):
        batch = ds[start : start + args.embed_batch]
        cids = batch["corpusid"]
        texts = [
            _paper_text({"title": t, "abstract": a})
            for t, a in zip(batch["title"], batch["abstract"])
        ]
        out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        for j, cid in enumerate(cids):
            pending.append({
                "corpus_id": int(cid),
                "embedding": out["dense_vecs"][j].tolist(),
                "sparse_embedding": _sparse(out["lexical_weights"][j]),
                "year": 0,
                "venue": "",
                "citationcount": 0,
                "classification": "",
            })
        while len(pending) >= args.insert_batch:
            coll.insert(pending[: args.insert_batch])
            total += args.insert_batch
            del pending[: args.insert_batch]
            log.info("inserted %d / %d", total, len(ds))
    if pending:
        coll.insert(pending)
        total += len(pending)
    coll.flush()
    log.info("insert complete: %d entities", total)


def main(args: BuildArgs) -> None:
    connect(MilvusEndpoint(uri=args.milvus_uri))
    coll = ensure_collection(LITSEARCH_COLLECTION, papers_schema())
    if coll.num_entities > 0:
        log.info("%s already has %d entities; skipping insert", LITSEARCH_COLLECTION, coll.num_entities)
    else:
        _embed_and_insert(coll, args)
    build_indexes(coll, {"embedding": HNSW_INDEX, "sparse_embedding": SPARSE_INDEX})
    load(coll)
    log.info("%s ready (num_entities=%d)", LITSEARCH_COLLECTION, coll.num_entities)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BuildArgs))
