"""Ingest paper + chunk embeddings into Milvus via the bulk_insert API.

Milvus's bulk_insert is the only way to import 34M chunks in reasonable time —
sequential gRPC `Collection.insert` saturates at ~9 MB/s on the standalone
deployment, projecting to ~4h for chunks alone. bulk_insert lets the server
read finalised parquet segments from MinIO in parallel.

Two things make bulk_insert fragile, both handled here:

1. Parquet column types must match Milvus's loader expectations EXACTLY:
   - FloatVector  → list<float32>   (NOT fixed_size_list — see _fixed_size_list_to_list)
   - SparseFloatVector → string column of JSON dicts   (NOT map<int,float>)
   See env/etl/README.md → "The required parquet schema for bulk_insert".

2. BGE-M3 sometimes emits an empty sparse dict ({}) for short inputs; Milvus
   rejects empty sparse vectors. We substitute {"0": 1e-9} as a placeholder.

The transformation is fully vectorised in PyArrow — we explicitly avoid
pymilvus.RemoteBulkWriter because its row-by-row append_row API caps at
~800 rows/sec, ~12h for our chunk volume.
"""
import dataclasses
import json
import logging
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from minio import Minio
from pymilvus import BulkInsertState, Collection, utility

from s2cs.env.milvus_client import (
    CHUNKS_COLLECTION,
    DIM,
    HNSW_INDEX,
    PAPERS_COLLECTION,
    SPARSE_INDEX,
    MilvusEndpoint,
    build_indexes,
    chunks_schema,
    connect,
    ensure_collection,
    load,
    papers_schema,
)

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class IngestMilvusArgs:
    milvus_uri: str = "http://localhost:19530"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "a-bucket"
    embeddings_paper_dir: Path = Path("data/processed/embeddings_paper")
    embeddings_chunk_dir: Path = Path("data/processed/embeddings_chunk")
    papers_meta_path: Path = Path("data/processed/papers_meta.parquet")
    stage_dir: Path = Path("/tmp/s2cs_bulk_stage")
    s3_prefix: str = "s2cs"
    poll_interval_s: int = 5
    max_concurrent_tasks: int = 8


def _minio(args: IngestMilvusArgs) -> Minio:
    return Minio(
        args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=False,
    )


def _fixed_size_list_to_list(col: pa.ChunkedArray, dim: int) -> pa.ListArray:
    """Cast fixed_size_list<float32, dim> → list<float32>.

    Milvus's bulk_insert refuses fixed-size-list columns for FloatVector
    fields — it wants a plain variable-length list with all elements having
    `dim` floats. PyArrow has no direct cast for this; we flatten the values
    and rebuild a ListArray with regular `i*dim` offsets, which is fully
    vectorised and avoids touching individual rows in Python.
    """
    fsl = col.combine_chunks()
    values = fsl.values
    n = len(fsl)
    offsets = pa.array(list(range(0, (n + 1) * dim, dim)), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, values)


def _map_to_sparse_json(col: pa.ChunkedArray) -> pa.Array:
    """Convert map<int64, float32> → JSON-string column for SparseFloatVector.

    Milvus bulk_insert wants sparse vectors as a string column where each cell
    is a JSON dict {"<token_id>": weight, ...}. See bulk_writer/buffer.py in
    pymilvus for the on-disk format.

    BGE-M3 sometimes emits an empty lexical_weights dict for short or
    degenerate inputs; Milvus rejects empty sparse vectors with
    'empty sparse vector is not allowed', so we inject a placeholder
    {"0": 1e-9} that has no practical effect on ranking.
    """
    strings: list[str] = []
    for entry in col.combine_chunks().to_pylist():
        if not entry:
            strings.append('{"0": 1e-9}')
            continue
        items = entry.items() if isinstance(entry, dict) else entry
        d = {str(int(k)): float(v) for k, v in items}
        strings.append(json.dumps(d))
    return pa.array(strings, type=pa.string())


def _convert_paper_shard(shard_path: Path, meta_by_id: dict) -> pa.Table:
    tbl = pq.read_table(shard_path)
    cids = tbl["corpus_id"].to_pylist()
    keep = [c in meta_by_id for c in cids]

    emb_list = _fixed_size_list_to_list(tbl["embedding"], DIM)
    sparse_json = _map_to_sparse_json(tbl["sparse_embedding"])

    years = pa.array([int(meta_by_id[c][1]) if k else 0 for c, k in zip(cids, keep)], type=pa.int32())
    venues = pa.array([meta_by_id[c][2] if k else "" for c, k in zip(cids, keep)], type=pa.string())
    cnts = pa.array([int(meta_by_id[c][3]) if k else 0 for c, k in zip(cids, keep)], type=pa.int32())
    classes = pa.array([meta_by_id[c][4] if k else "" for c, k in zip(cids, keep)], type=pa.string())

    new_tbl = pa.table({
        "corpus_id": tbl["corpus_id"],
        "embedding": emb_list,
        "sparse_embedding": sparse_json,
        "year": years,
        "venue": venues,
        "citationcount": cnts,
        "classification": classes,
    })
    if not all(keep):
        new_tbl = new_tbl.filter(pa.array(keep, type=pa.bool_()))
    return new_tbl


def _convert_chunk_shard(shard_path: Path) -> pa.Table:
    tbl = pq.read_table(shard_path)
    emb_list = _fixed_size_list_to_list(tbl["embedding"], DIM)
    return pa.table({
        "chunk_id": tbl["chunk_id"],
        "paper_corpus_id": tbl["paper_corpus_id"],
        "section_idx": tbl["section_idx"].cast(pa.int32()),
        "embedding": emb_list,
    })


def _stage_and_upload(table: pa.Table, kind: str, shard_name: str,
                      args: IngestMilvusArgs, mc: Minio) -> str:
    local = args.stage_dir / kind / shard_name
    local.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, local, compression="zstd")
    s3_path = f"{args.s3_prefix}/{kind}/{shard_name}"
    mc.fput_object(args.minio_bucket, s3_path, str(local))
    local.unlink()
    return s3_path


def _bulk_import(collection_name: str, s3_paths: list[str], args: IngestMilvusArgs) -> None:
    queue = list(s3_paths)
    in_flight: dict[int, str] = {}
    completed = 0
    while queue or in_flight:
        while queue and len(in_flight) < args.max_concurrent_tasks:
            sp = queue.pop(0)
            tid = utility.do_bulk_insert(collection_name=collection_name, files=[sp])
            in_flight[tid] = sp
            log.info("submitted task %s for %s (in_flight=%d, queue=%d)",
                     tid, sp, len(in_flight), len(queue))
        time.sleep(args.poll_interval_s)
        for tid in list(in_flight):
            st = utility.get_bulk_insert_state(task_id=tid)
            if st.state == BulkInsertState.ImportCompleted:
                del in_flight[tid]
                completed += 1
                log.info("task %s done (%s rows). completed=%d / %d",
                         tid, st.row_count, completed, len(s3_paths))
            elif st.state in (BulkInsertState.ImportFailed, BulkInsertState.ImportFailedAndCleaned):
                raise RuntimeError(f"bulk_insert task {tid} failed: {st.failed_reason}")


def _ingest_papers(coll: Collection, args: IngestMilvusArgs, mc: Minio) -> None:
    if coll.num_entities > 0:
        log.info("papers collection already has %d entities; skipping insert", coll.num_entities)
        return
    con = duckdb.connect()
    meta = con.execute(f"""
        SELECT corpus_id,
               CAST(COALESCE(year, 0) AS INT) AS year,
               COALESCE(venue, '') AS venue,
               CAST(COALESCE(citationcount, 0) AS INT) AS citationcount,
               COALESCE(classification, '') AS classification
        FROM read_parquet('{args.papers_meta_path}')
    """).fetchall()
    meta_by_id = {row[0]: row for row in meta}
    log.info("loaded %d papers_meta rows", len(meta_by_id))

    shards = sorted(args.embeddings_paper_dir.glob("shard_*.parquet"))
    log.info("staging %d paper shards", len(shards))
    s3_paths: list[str] = []
    for shard in shards:
        tbl = _convert_paper_shard(shard, meta_by_id)
        sp = _stage_and_upload(tbl, "papers", shard.name, args, mc)
        s3_paths.append(sp)
        log.info("  staged %s -> %s (%d rows)", shard.name, sp, tbl.num_rows)
    log.info("paper staging complete; firing bulk_insert")
    _bulk_import(PAPERS_COLLECTION, s3_paths, args)
    coll.flush()
    log.info("papers bulk_insert done, num_entities=%d", coll.num_entities)


def _ingest_chunks(coll: Collection, args: IngestMilvusArgs, mc: Minio) -> None:
    if coll.num_entities > 0:
        log.info("chunks collection already has %d entities; skipping insert", coll.num_entities)
        return
    shards = sorted(args.embeddings_chunk_dir.glob("shard_*.parquet"))
    log.info("staging %d chunk shards", len(shards))
    s3_paths: list[str] = []
    for i, shard in enumerate(shards, 1):
        tbl = _convert_chunk_shard(shard)
        sp = _stage_and_upload(tbl, "chunks", shard.name, args, mc)
        s3_paths.append(sp)
        if i % 20 == 0 or i == len(shards):
            log.info("  staged %d / %d chunk shards", i, len(shards))
    log.info("chunk staging complete; firing bulk_insert")
    _bulk_import(CHUNKS_COLLECTION, s3_paths, args)
    coll.flush()
    log.info("chunks bulk_insert done, num_entities=%d", coll.num_entities)


def main(args: IngestMilvusArgs) -> None:
    connect(MilvusEndpoint(uri=args.milvus_uri))
    mc = _minio(args)
    if not mc.bucket_exists(args.minio_bucket):
        mc.make_bucket(args.minio_bucket)
    args.stage_dir.mkdir(parents=True, exist_ok=True)

    papers = ensure_collection(PAPERS_COLLECTION, papers_schema())
    _ingest_papers(papers, args, mc)
    build_indexes(papers, {"embedding": HNSW_INDEX, "sparse_embedding": SPARSE_INDEX})
    load(papers)

    chunks = ensure_collection(CHUNKS_COLLECTION, chunks_schema())
    _ingest_chunks(chunks, args, mc)
    build_indexes(chunks, {"embedding": HNSW_INDEX})
    load(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(IngestMilvusArgs))
