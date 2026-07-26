from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2cs.env.graph import S2Graph


@pytest.fixture
def edges_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "edges.parquet"
    table = pa.table({
        "src": pa.array([1, 1, 1, 2, 2, 3], type=pa.int64()),
        "dst": pa.array([10, 11, 12, 11, 13, 13], type=pa.int64()),
    })
    pq.write_table(table, path)
    return path


def test_references_returns_outgoing(edges_parquet):
    g = S2Graph(edges_parquet)
    assert sorted(g.references(1)) == [10, 11, 12]


def test_cited_by_returns_incoming(edges_parquet):
    g = S2Graph(edges_parquet)
    assert sorted(g.cited_by(13)) == [2, 3]


def test_missing_node_returns_empty(edges_parquet):
    g = S2Graph(edges_parquet)
    assert g.references(999) == []
    assert g.cited_by(999) == []


def test_limit_truncates(edges_parquet):
    g = S2Graph(edges_parquet)
    assert len(g.references(1, limit=2)) == 2
    assert len(g.cited_by(11, limit=1)) == 1
