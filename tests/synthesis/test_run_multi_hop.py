import duckdb
import pytest

from s2cs.synthesis.run_multi_hop import (
    ATTEMPTED_FILE,
    QA_FILE,
    MultiHopSynthArgs,
    _attach_cited_with_abstract,
)
from s2cs.synthesis.run_paper_set import _done_ids


def test_args_expose_hops_defaulting_to_two():
    a = MultiHopSynthArgs()
    assert a.hops == 2 and a.min_hops == 2


def test_done_ids_reads_multi_hop_files(tmp_path):
    # resume must key off multi_hop's OWN files, not paper_set's (the bug: _done_ids
    # was called with paper_set's default filenames against a multi_hop out_dir).
    (tmp_path / QA_FILE).write_text('{"seed_paper_ids": [42]}\n')
    (tmp_path / ATTEMPTED_FILE).write_text("42\n99\n")
    assert _done_ids(tmp_path, QA_FILE, ATTEMPTED_FILE) == {42, 99}
    # with paper_set's default filenames the multi_hop files are invisible -> empty
    assert _done_ids(tmp_path) == set()


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "papers.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE papers_text (corpus_id BIGINT, title VARCHAR, abstract VARCHAR, body VARCHAR)")
    con.execute("CREATE TABLE papers_meta (corpus_id BIGINT, citationcount BIGINT)")
    con.executemany("INSERT INTO papers_text VALUES (?, ?, ?, ?)", [
        (7, "Dense Passage Retrieval", "A dual-encoder for QA.", "body7"),
        (11, "Wavelet spindles", "Signal processing.", "body11"),
    ])
    con.executemany("INSERT INTO papers_meta VALUES (?, ?)", [(7, 500), (11, 3)])
    edges = tmp_path / "edges.parquet"
    con.execute("CREATE TABLE e (src BIGINT, dst BIGINT)")
    con.executemany("INSERT INTO e VALUES (?, ?)", [(100, 7), (100, 11)])
    con.execute(f"COPY e TO '{edges}' (FORMAT PARQUET)")
    con.close()
    return p, edges


def test_attach_cited_with_abstract(db):
    papers_db, edges = db
    out = _attach_cited_with_abstract(papers_db, edges, [{"corpus_id": 100}], limit=12)
    assert len(out) == 1
    cited = out[0]["cited"]
    # lowest citationcount first (matches single_hop ordering); abstracts present
    assert [c["corpus_id"] for c in cited] == [11, 7]
    assert cited[1]["title"] == "Dense Passage Retrieval"
    assert cited[1]["abstract"] == "A dual-encoder for QA."


def test_attach_drops_seed_without_cited(db):
    papers_db, edges = db
    out = _attach_cited_with_abstract(papers_db, edges, [{"corpus_id": 999}], limit=12)
    assert out == []
