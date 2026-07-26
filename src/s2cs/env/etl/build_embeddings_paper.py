import dataclasses
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from FlagEmbedding import BGEM3FlagModel

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildPaperEmbeddingsArgs:
    raw_dir: Path = Path("data/raw/s2orc")
    out_dir: Path = Path("data/processed/embeddings_paper")
    model: str = "BAAI/bge-m3"
    batch_size: int = 256
    max_len: int = 8192
    shard_size: int = 50_000
    fp16: bool = True


def _format_input(title: str | None, abstract: str | None, summary: str | None) -> str:
    return "\n\n".join(p for p in (title, abstract, summary) if p)


def _existing_shard_count(out_dir: Path, rank: int) -> int:
    return len(list(out_dir.glob(f"shard_{rank:02d}_*.parquet")))


def _flush_shard(
    out_dir: Path, rank: int, shard_idx: int,
    ids: list[int], dense: np.ndarray, sparse: list[dict[int, float]],
) -> None:
    final = out_dir / f"shard_{rank:02d}_{shard_idx:04d}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    table = pa.table({
        "corpus_id": pa.array(ids, type=pa.int64()),
        "embedding": pa.array(list(dense), type=pa.list_(pa.float32(), 1024)),
        "sparse_embedding": pa.array(sparse, type=pa.map_(pa.int64(), pa.float32())),
    })
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(final)
    log.info("[rank %d] wrote %s (%d rows)", rank, final.name, len(ids))


def _to_sparse_dict(weights: dict) -> dict[int, float]:
    return {int(k): float(v) for k, v in weights.items()}


def main(args: BuildPaperEmbeddingsArgs) -> None:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    log.info("rank=%d world_size=%d gpu_visible=%s", rank, world_size,
             os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.raw_dir.glob("*.parquet"))
    my_files = [str(p) for i, p in enumerate(all_files) if i % world_size == rank]
    log.info("[rank %d] reading %d / %d parquet files", rank, len(my_files), len(all_files))

    con = duckdb.connect()
    con.execute("SET threads = 8")
    rows = con.execute(f"""
        SELECT corpus_id, parsed_title, abstract, summary
        FROM read_parquet({my_files!r}, union_by_name=true)
        ORDER BY corpus_id
    """).fetchall()

    done_shards = _existing_shard_count(args.out_dir, rank)
    skip = done_shards * args.shard_size
    log.info("[rank %d] %d rows; resuming from shard %d (skip %d)", rank, len(rows), done_shards, skip)
    rows = rows[skip:]

    model = BGEM3FlagModel(args.model, use_fp16=args.fp16, devices=["cuda:0"])

    shard_idx = done_shards
    for start in range(0, len(rows), args.shard_size):
        batch = rows[start : start + args.shard_size]
        ids = [int(c) for c, *_ in batch]
        texts = [_format_input(t, a, s) for _, t, a, s in batch]
        out = model.encode(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_len,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"].astype(np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True).clip(min=1e-12)
        sparse = [_to_sparse_dict(w) for w in out["lexical_weights"]]
        _flush_shard(args.out_dir, rank, shard_idx, ids, dense, sparse)
        shard_idx += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BuildPaperEmbeddingsArgs))
