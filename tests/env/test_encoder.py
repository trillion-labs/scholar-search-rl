import threading
import time
from unittest.mock import MagicMock

import numpy as np

from s2cs.env.encoder import BatchedEncoder


class FakeM3:
    """Stand-in for FlagEmbedding.BGEM3FlagModel.encode."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def encode(self, sentences, *, return_dense=True, return_sparse=False, return_colbert_vecs=False):
        with self.lock:
            self.calls.append({"n": len(sentences), "hybrid": return_sparse})
        n = len(sentences)
        out = {}
        if return_dense:
            out["dense_vecs"] = np.arange(n * self.dim, dtype=np.float32).reshape(n, self.dim)
        if return_sparse:
            out["lexical_weights"] = [{i: float(i) + 0.5} for i in range(n)]
        return out


def test_single_call_returns_dense_and_sparse():
    enc = BatchedEncoder(FakeM3(dim=4), max_batch=8, wait_ms=2)
    dense, sparse = enc.encode_hybrid("q")
    assert len(dense) == 4
    assert isinstance(sparse, dict) and sparse


def test_single_dense_call_returns_list():
    enc = BatchedEncoder(FakeM3(dim=4), max_batch=8, wait_ms=2)
    out = enc.encode_dense("q")
    assert isinstance(out, list) and len(out) == 4


def test_concurrent_hybrid_calls_get_batched():
    fake = FakeM3(dim=4)
    enc = BatchedEncoder(fake, max_batch=8, wait_ms=30)

    results: list = []
    threads = [threading.Thread(target=lambda i=i: results.append(enc.encode_hybrid(f"q{i}"))) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(results) == 8
    # All 8 should have been folded into one (or very few) model.encode calls
    hybrid_calls = [c for c in fake.calls if c["hybrid"]]
    assert hybrid_calls, "no hybrid calls recorded"
    assert max(c["n"] for c in hybrid_calls) > 1, "expected batched call, got all size-1"


def test_concurrent_mixed_modes_split_into_two_batched_calls():
    fake = FakeM3(dim=4)
    enc = BatchedEncoder(fake, max_batch=8, wait_ms=30)

    def call_hybrid(i: int): enc.encode_hybrid(f"h{i}")
    def call_dense(i: int):  enc.encode_dense(f"d{i}")

    threads = (
        [threading.Thread(target=call_hybrid, args=(i,)) for i in range(4)]
        + [threading.Thread(target=call_dense, args=(i,)) for i in range(4)]
    )
    for t in threads: t.start()
    for t in threads: t.join()

    hybrid = [c for c in fake.calls if c["hybrid"]]
    dense = [c for c in fake.calls if not c["hybrid"]]
    assert hybrid and dense, "both modes should have triggered a call"
    assert sum(c["n"] for c in hybrid) == 4
    assert sum(c["n"] for c in dense) == 4


def test_exception_propagates_to_all_in_batch():
    fake = MagicMock()
    fake.encode.side_effect = RuntimeError("kaboom")
    enc = BatchedEncoder(fake, max_batch=8, wait_ms=20)

    errors: list = []
    def call():
        try: enc.encode_hybrid("q")
        except RuntimeError as e: errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(errors) == 4
    assert all("kaboom" in str(e) for e in errors)


def test_wait_ms_does_not_delay_single_call_indefinitely():
    enc = BatchedEncoder(FakeM3(dim=4), max_batch=16, wait_ms=10)
    t0 = time.time()
    enc.encode_dense("q")
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"single call took too long: {elapsed:.3f}s"
