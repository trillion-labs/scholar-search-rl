"""In-process batched encoder for BGE-M3.

The tools (search_papers, search_snippets) call encode_hybrid / encode_dense
per query. When many rollouts run concurrently the GPU serialises the
individual forward passes, wasting capacity. BatchedEncoder coalesces
concurrent calls: each caller's query goes on a queue, a worker thread
collects up to max_batch items (waiting at most wait_ms after the first),
runs one model.encode([q1, q2, ..., qK], ...), and scatters results back
through Futures.

The caller-facing API is unchanged — encode_hybrid(q) still returns
(dense_vec, sparse_dict) — so the tool code does not need to know that
batching exists.
"""

import logging
import math
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np

log = logging.getLogger(__name__)


class BatchedEncoder:
    def __init__(self, model: Any, *, max_batch: int = 16, wait_ms: int = 5) -> None:
        self.model = model
        self.max_batch = max_batch
        self.wait_ms = wait_ms
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="BatchedEncoderWorker")
        self._worker.start()

    def encode_hybrid(self, q: str) -> tuple[list[float], dict[int, float]]:
        fut: Future = Future()
        self._queue.put(("hybrid", q, fut))
        return fut.result()

    def encode_dense(self, q: str) -> list[float]:
        fut: Future = Future()
        self._queue.put(("dense", q, fut))
        return fut.result()

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.time() + self.wait_ms / 1000
            while len(batch) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._process(batch)

    def _process(self, batch: list[tuple[str, str, Future]]) -> None:
        hybrid = [(q, fut) for mode, q, fut in batch if mode == "hybrid"]
        dense = [(q, fut) for mode, q, fut in batch if mode == "dense"]

        if hybrid:
            qs = [q for q, _ in hybrid]
            try:
                out = self.model.encode(qs, return_dense=True, return_sparse=True, return_colbert_vecs=False)
                for i, (_, fut) in enumerate(hybrid):
                    fut.set_result(
                        (
                            out["dense_vecs"][i].tolist(),
                            {int(k): float(v) for k, v in out["lexical_weights"][i].items()},
                        )
                    )
            except Exception as exc:
                log.exception("hybrid batch failed (size=%d)", len(hybrid))
                for _, fut in hybrid:
                    fut.set_exception(exc)

        if dense:
            qs = [q for q, _ in dense]
            try:
                out = self.model.encode(qs, return_dense=True, return_sparse=False, return_colbert_vecs=False)
                for i, (_, fut) in enumerate(dense):
                    fut.set_result(out["dense_vecs"][i].tolist())
            except Exception as exc:
                log.exception("dense batch failed (size=%d)", len(dense))
                for _, fut in dense:
                    fut.set_exception(exc)


class RemoteBGEM3:
    """HTTP client for one or more `scripts/embed_server.py` instances.

    Implements the slice of `BGEM3FlagModel.encode` that `BatchedEncoder` uses, so
    `build_tools` can wrap it identically: `BatchedEncoder(RemoteBGEM3(urls), ...)`.
    `encode(texts, ...)` returns `{"dense_vecs": [np.ndarray, ...],
    "lexical_weights": [dict, ...]}` matching the local model's output.

    Each call's texts are split into contiguous chunks — one per server — and the
    chunks are POSTed concurrently, so N servers embed in parallel (the client-side
    `BatchedEncoder` worker is single-threaded, so a single server would otherwise
    serialise). Single-text calls round-robin across servers for balance. Results
    are reassembled in the original text order.
    """

    def __init__(self, urls: list[str], *, timeout: float = 30.0) -> None:
        self._urls = [u.rstrip("/") for u in urls]
        if not self._urls:
            raise ValueError("RemoteBGEM3 needs at least one url")
        self._client = httpx.Client(timeout=timeout)
        self._pool = ThreadPoolExecutor(max_workers=len(self._urls), thread_name_prefix="remote-bge")
        self._rr = 0
        self._rr_lock = threading.Lock()

    def _post(self, url: str, texts: list[str], return_sparse: bool) -> dict:
        resp = self._client.post(f"{url}/encode", json={"texts": texts, "return_sparse": return_sparse})
        resp.raise_for_status()
        return resp.json()

    def encode(self, texts, *, return_dense=True, return_sparse=False, return_colbert_vecs=False) -> dict:
        texts = list(texts)
        if not texts:
            return {"dense_vecs": [], "lexical_weights": []}
        n = len(self._urls)
        n_chunks = min(len(texts), n)
        size = math.ceil(len(texts) / n_chunks)
        chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
        with self._rr_lock:
            base = self._rr
            self._rr += len(chunks)
        futures = [
            self._pool.submit(self._post, self._urls[(base + j) % n], chunk, return_sparse)
            for j, chunk in enumerate(chunks)
        ]
        dense_vecs: list = []
        lexical: list = []
        for fut in futures:  # submission order == chunk order == original text order
            resp = fut.result()
            dense_vecs.extend(np.asarray(v, dtype=np.float32) for v in resp["dense"])
            if return_sparse:
                lexical.extend(resp.get("sparse", []))
        return {"dense_vecs": dense_vecs, "lexical_weights": lexical}


class OpenAIEmbeddings:
    """Dense-only encoder backed by an OpenAI-compatible `/v1/embeddings` endpoint.

    Implements the slice of `BGEM3FlagModel.encode` that `BatchedEncoder` uses, so
    it drops in via `build_tools(encoder=BatchedEncoder(OpenAIEmbeddings(...)))` and
    runs query embedding fully off-GPU. The intended backend is OpenRouter's
    `baai/bge-m3`, whose dense vectors are identical (cos≈1.0) to the local model's.

    The API returns DENSE vectors only — there is no learned-sparse output — so
    `lexical_weights` come back empty and hybrid callers (`search_papers`) fall back
    to dense-only retrieval. That costs ~9pp seed-recall@10 on value4k vs hybrid.
    """

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def encode(self, texts, *, return_dense=True, return_sparse=False, return_colbert_vecs=False) -> dict:
        texts = list(texts)
        if not texts:
            return {"dense_vecs": [], "lexical_weights": []}
        resp = self._client.post(self._url, json={"model": self._model, "input": texts})
        resp.raise_for_status()
        # OpenAI returns each item with an "index"; sort to restore input order.
        data = sorted(resp.json()["data"], key=lambda e: e.get("index", 0))
        dense_vecs = [np.asarray(e["embedding"], dtype=np.float32) for e in data]
        # No sparse head over HTTP -> empty dicts; search_papers reads this as dense-only.
        return {"dense_vecs": dense_vecs, "lexical_weights": [{} for _ in texts]}
