import dataclasses
import hashlib
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
class BuildChunkEmbeddingsArgs:
    raw_dir: Path = Path("data/raw/s2orc")
    out_dir: Path = Path("data/processed/embeddings_chunk")
    model: str = "BAAI/bge-m3"
    batch_size: int = 1024
    window_chars: int = 1600
    stride_chars: int = 200
    min_chars: int = 100
    max_len: int = 1024
    shard_size: int = 200_000
    fp16: bool = True


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: int
    paper_corpus_id: int
    section_idx: int
    char_start: int
    char_end: int
    text: str


def _chunk_id(paper_corpus_id: int, section_idx: int, char_start: int, char_end: int) -> int:
    h = hashlib.blake2b(f"{paper_corpus_id}:{section_idx}:{char_start}:{char_end}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big", signed=True)


def _slide(text: str, window: int, stride: int, min_chars: int) -> list[tuple[int, int]]:
    if not text or len(text) < min_chars:
        return []
    if len(text) <= window:
        return [(0, len(text))]
    spans = []
    start = 0
    step = max(1, window - stride)
    while start < len(text):
        end = min(start + window, len(text))
        if end - start >= min_chars:
            spans.append((start, end))
        if end == len(text):
            break
        start += step
    return spans


def _chunk_paper(
    corpus_id: int,
    abstract: str | None,
    sections: list[dict] | None,
    *,
    window: int, stride: int, min_chars: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    if abstract:
        for cs, ce in _slide(abstract, window, stride, min_chars):
            out.append(Chunk(
                chunk_id=_chunk_id(corpus_id, -1, cs, ce),
                paper_corpus_id=corpus_id, section_idx=-1,
                char_start=cs, char_end=ce, text=abstract[cs:ce],
            ))
    for i, sec in enumerate(sections or []):
        content = sec.get("content") if isinstance(sec, dict) else None
        if not content:
            continue
        for cs, ce in _slide(content, window, stride, min_chars):
            out.append(Chunk(
                chunk_id=_chunk_id(corpus_id, i, cs, ce),
                paper_corpus_id=corpus_id, section_idx=i,
                char_start=cs, char_end=ce, text=content[cs:ce],
            ))
    return out


def _existing_shard_count(out_dir: Path, rank: int) -> int:
    return len(list(out_dir.glob(f"shard_{rank:02d}_*.parquet")))


def _flush_shard(out_dir: Path, rank: int, shard_idx: int, chunks: list[Chunk], vecs: np.ndarray) -> None:
    final = out_dir / f"shard_{rank:02d}_{shard_idx:04d}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    table = pa.table({
        "chunk_id":        pa.array([c.chunk_id for c in chunks], type=pa.int64()),
        "paper_corpus_id": pa.array([c.paper_corpus_id for c in chunks], type=pa.int64()),
        "section_idx":     pa.array([c.section_idx for c in chunks], type=pa.int32()),
        "char_start":      pa.array([c.char_start for c in chunks], type=pa.int32()),
        "char_end":        pa.array([c.char_end for c in chunks], type=pa.int32()),
        "embedding":       pa.array(list(vecs), type=pa.list_(pa.float32(), 1024)),
    })
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(final)
    log.info("[rank %d] wrote %s (%d chunks)", rank, final.name, len(chunks))


def main(args: BuildChunkEmbeddingsArgs) -> None:
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
        SELECT corpus_id, abstract, sections
        FROM read_parquet({my_files!r}, union_by_name=true)
        ORDER BY corpus_id
    """).fetchall()

    done_shards = _existing_shard_count(args.out_dir, rank)
    skip_chunks = done_shards * args.shard_size
    log.info("[rank %d] %d papers; resuming from shard %d (skip %d chunks)",
             rank, len(rows), done_shards, skip_chunks)

    model = BGEM3FlagModel(args.model, use_fp16=args.fp16)

    shard_idx = done_shards
    chunks_buf: list[Chunk] = []
    emitted = 0

    def flush() -> None:
        nonlocal chunks_buf, shard_idx
        texts = [c.text for c in chunks_buf]
        out = model.encode(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_len,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"].astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        _flush_shard(args.out_dir, rank, shard_idx, chunks_buf, vecs)
        shard_idx += 1
        chunks_buf = []

    for corpus_id, abstract, sections in rows:
        for c in _chunk_paper(
            int(corpus_id), abstract, sections,
            window=args.window_chars, stride=args.stride_chars, min_chars=args.min_chars,
        ):
            if emitted < skip_chunks:
                emitted += 1
                continue
            chunks_buf.append(c)
            emitted += 1
            if len(chunks_buf) >= args.shard_size:
                flush()

    if chunks_buf:
        flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BuildChunkEmbeddingsArgs))
