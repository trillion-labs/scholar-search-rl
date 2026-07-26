import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

from s2cs.env.encoder import BatchedEncoder


@dataclasses.dataclass
class FakeEntity:
    fields: dict[str, Any]

    def get(self, name: str) -> Any:
        return self.fields.get(name)


@dataclasses.dataclass
class FakeHit:
    entity: FakeEntity
    distance: float


@pytest.fixture
def encode_dense_fake():
    def _encode(_q: str) -> list[float]:
        return [0.0] * 1024
    return _encode


@pytest.fixture
def encode_hybrid_fake():
    def _encode(_q: str) -> tuple[list[float], dict[int, float]]:
        return [0.0] * 1024, {1: 0.5, 2: 0.3}
    return _encode


@pytest.fixture
def fake_encoder():
    """A BatchedEncoder wrapped around a deterministic mock model.

    Wired into the live tool path the same way the production code expects;
    safe to use anywhere a registry needs an `encoder` argument.
    """
    class _FakeModel:
        def encode(self, sentences, *, return_dense=True, return_sparse=False, return_colbert_vecs=False):
            import numpy as np
            n = len(sentences)
            out: dict[str, Any] = {}
            if return_dense:
                out["dense_vecs"] = np.zeros((n, 1024), dtype=np.float32)
            if return_sparse:
                out["lexical_weights"] = [{1: 0.5, 2: 0.3} for _ in range(n)]
            return out

    enc = BatchedEncoder(_FakeModel(), max_batch=8, wait_ms=2)
    yield enc
    enc.close()


@pytest.fixture
def mock_papers_collection():
    coll = MagicMock()
    hits = [
        FakeHit(FakeEntity({"corpus_id": 1, "year": 2024, "venue": "NeurIPS", "citationcount": 42}), 0.9),
        FakeHit(FakeEntity({"corpus_id": 2, "year": 2023, "venue": "ICLR", "citationcount": 13}), 0.8),
        FakeHit(FakeEntity({"corpus_id": 3, "year": 2022, "venue": "ICML", "citationcount": 7}), 0.7),
    ]
    coll.search.return_value = [hits]
    coll.hybrid_search.return_value = [hits]
    coll.query.return_value = [{"embedding": [0.0] * 1024}]
    return coll


@pytest.fixture
def mock_chunks_collection():
    coll = MagicMock()
    hits = [
        FakeHit(FakeEntity({"chunk_id": 100, "paper_corpus_id": 1, "section_idx": 0}), 0.95),
        FakeHit(FakeEntity({"chunk_id": 101, "paper_corpus_id": 2, "section_idx": 1}), 0.85),
    ]
    coll.search.return_value = [hits]
    return coll


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.references.return_value = [10, 11, 12]
    g.cited_by.return_value = [20, 21]
    return g


@pytest.fixture
def mock_reader():
    r = MagicMock()
    from s2cs.env.reader import Match, PaperMeta, ReadResult, SectionRef
    r.meta.return_value = PaperMeta(
        corpus_id=1, title="Hello", abstract="abs", summary="sum",
        year=2024, venue="NeurIPS", citation_count=42,
        classification="Methods Paper", reference_count=10,
        sections=[
            SectionRef(section_idx=0, title="Intro", n_lines=12),
            SectionRef(section_idx=1, title="Methods", n_lines=40),
        ],
    )
    r.read.return_value = ReadResult(
        paper_id=1, section_idx=None, offset=0, limit=200, total_lines=3,
        text="line one\nline two\nline three",
    )
    r.find_in_paper.return_value = [
        Match(line_number=1, line="match here", context_before=["before"], context_after=["after"]),
    ]
    return r
