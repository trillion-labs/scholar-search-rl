"""High-level factory: get the 9 tool callables as named attributes.

Hides Milvus, MinIO, DuckDB, and S2Graph wiring behind a single
`build_tools(...)` that reads configuration from a `.env` file (committed at
repo root) and returns a Tools dataclass — call the tools by name:

    from s2cs.env import build_tools
    tools = build_tools()
    hits = tools.search_papers("transformer", limit=10)
    info = tools.paper_info(hits[0].corpus_id)

Pin endpoints explicitly when needed:

    tools = build_tools(milvus_uri="http://127.0.0.1:19530")

Config precedence: explicit kwarg → process env var → `.env` file. Every
path / URI is in `.env`; this module ships no path constants.
"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv
from pymilvus import Collection

from s2cs.env.encoder import BatchedEncoder
from s2cs.env.graph import S2Graph
from s2cs.env.milvus_client import (
    CHUNKS_COLLECTION,
    PAPERS_COLLECTION,
    MilvusEndpoint,
    connect,
)
from s2cs.env.reader import PaperReader
from s2cs.env.tools.registry import build_registry

log = logging.getLogger(__name__)

REQUIRED_KEYS = ("S2CS_MILVUS_URI", "S2CS_PAPERS_DB", "S2CS_EDGES_PATH", "S2CS_MODEL")


@dataclasses.dataclass(frozen=True)
class Tools:
    """Bundle of the nine s2cs env callables exposed as named attributes.

    Use them by name — `tools.search_papers(...)`, `tools.paper_info(...)`.
    Dict-style access (`tools["search_papers"]`) and `.items()` / `.keys()`
    are also available for code that wants to iterate over the surface.
    """
    search_papers: Callable
    search_snippets: Callable
    paper_info: Callable
    read_paper: Callable
    find_in_paper: Callable
    list_references: Callable
    list_citations: Callable
    find_similar: Callable
    submit_answer: Callable

    def __getitem__(self, name: str) -> Callable:
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in dataclasses.fields(self))

    def keys(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self)]

    def items(self) -> list[tuple[str, Callable]]:
        return [(f.name, getattr(self, f.name)) for f in dataclasses.fields(self)]

    def subset(self, names: list[str]) -> dict[str, Callable]:
        missing = [n for n in names if n not in self.keys()]
        if missing:
            raise KeyError(f"unknown tools: {missing}")
        return {n: getattr(self, n) for n in names}


def _require(value: Any, env_key: str) -> str:
    if value is not None:
        return str(value)
    env_value = os.environ.get(env_key)
    if not env_value:
        raise RuntimeError(
            f"{env_key} is not set. Expected in `.env` at the repo root, "
            f"or in the process environment, or passed as a kwarg to build_tools()."
        )
    return env_value


def _default_encoder(model_name: str, devices: list[str] | None) -> BatchedEncoder:
    from FlagEmbedding import BGEM3FlagModel
    log.info("loading %s ...", model_name)
    model = BGEM3FlagModel(model_name, use_fp16=True, devices=devices or ["cuda:0"])
    return BatchedEncoder(model, max_batch=32, wait_ms=5)


def build_tools(
    *,
    env_file: str | Path | None = None,
    milvus_uri: str | None = None,
    papers_db: str | Path | None = None,
    edges_path: str | Path | None = None,
    encoder: BatchedEncoder | None = None,
    model_name: str | None = None,
    devices: list[str] | None = None,
) -> Tools:
    """Return the nine env tool callables as a named bundle.

    Config is loaded from `.env` at the repo root (or `env_file=` if given),
    then overlaid with process env vars, then overlaid with kwargs.

    Required keys:
        S2CS_MILVUS_URI    e.g. http://127.0.0.1:19530
        S2CS_PAPERS_DB     path to papers.duckdb
        S2CS_EDGES_PATH    path to edges.parquet
        S2CS_MODEL         e.g. BAAI/bge-m3
    Optional:
        S2CS_DEVICES       comma-separated, e.g. cuda:0,cuda:1
        S2CS_EMBED_URL     comma-separated embed-server URL(s); when set (and no
                           explicit `encoder=`), queries are embedded remotely via
                           RemoteBGEM3 instead of a local BGE-M3 on this GPU.
        S2CS_EMBED_OPENAI_URL  OpenAI-compatible /v1 base URL for dense embeddings
                           (e.g. https://openrouter.ai/api/v1); GPU-free, dense-only.
                           Pair with S2CS_EMBED_OPENAI_MODEL (default baai/bge-m3) and
                           S2CS_EMBED_OPENAI_KEY. Takes precedence over S2CS_EMBED_URL.

    Pass `encoder=` to skip loading BGE-M3 and reuse your own.
    """
    load_dotenv(env_file)

    milvus_uri = _require(milvus_uri, "S2CS_MILVUS_URI")
    papers_db = Path(_require(papers_db, "S2CS_PAPERS_DB"))
    edges_path = Path(_require(edges_path, "S2CS_EDGES_PATH"))
    model_name = _require(model_name, "S2CS_MODEL")
    if devices is None and (env_dev := os.environ.get("S2CS_DEVICES")):
        devices = [d.strip() for d in env_dev.split(",")]

    log.info("milvus=%s  papers_db=%s  edges=%s", milvus_uri, papers_db, edges_path)
    if not papers_db.exists():
        raise FileNotFoundError(f"papers DuckDB not found: {papers_db}")
    if not edges_path.exists():
        raise FileNotFoundError(f"edges parquet not found: {edges_path}")

    connect(MilvusEndpoint(uri=milvus_uri))
    papers = Collection(PAPERS_COLLECTION); papers.load()
    chunks = Collection(CHUNKS_COLLECTION); chunks.load()

    reader = PaperReader(papers_db)
    graph = S2Graph(edges_path)
    if encoder is not None:
        enc = encoder
    elif openai_url := os.environ.get("S2CS_EMBED_OPENAI_URL"):
        # Dense-only embeddings over an OpenAI-compatible /v1/embeddings endpoint
        # (e.g. OpenRouter baai/bge-m3) — fully GPU-free, no local model and no embed
        # server. Returns no learned sparse, so search_papers falls back to dense-only.
        from s2cs.env.encoder import OpenAIEmbeddings

        emb_model = os.environ.get("S2CS_EMBED_OPENAI_MODEL", "baai/bge-m3")
        log.info("using OpenAI-compatible dense embeddings: %s @ %s", emb_model, openai_url)
        enc = BatchedEncoder(
            OpenAIEmbeddings(openai_url, emb_model, os.environ.get("S2CS_EMBED_OPENAI_KEY", "")),
            max_batch=32, wait_ms=5,
        )
    elif embed_url := os.environ.get("S2CS_EMBED_URL"):
        # Remote BGE-M3: call an embedding server (e.g. on an idle GPU) over HTTP
        # instead of loading a local model on this process's GPU — keeps the encoder
        # off the rollout/sglang GPUs. Comma-separated URLs are round-robined.
        from s2cs.env.encoder import RemoteBGEM3

        urls = [u.strip() for u in embed_url.split(",") if u.strip()]
        log.info("using remote BGE-M3 encoder(s): %s", urls)
        enc = BatchedEncoder(RemoteBGEM3(urls), max_batch=32, wait_ms=5)
    else:
        enc = _default_encoder(model_name, devices)

    registry = build_registry(
        papers=papers, chunks=chunks, graph=graph, reader=reader, encoder=enc,
    )
    return Tools(**registry)
