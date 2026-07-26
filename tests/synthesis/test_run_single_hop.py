import json
from pathlib import Path

import duckdb

from s2cs.synthesis.run_single_hop import _attach_cited, _done_ids


def _make_db(tmp_path) -> Path:
    db = tmp_path / "papers.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE papers_text(corpus_id BIGINT, title VARCHAR)")
    con.execute("CREATE TABLE papers_meta(corpus_id BIGINT, year BIGINT, citationcount BIGINT)")
    con.executemany("INSERT INTO papers_text VALUES (?, ?)", [
        (10, "Famous prior work"), (11, "Niche prior work"), (12, "Untitled"),
    ])
    con.execute("UPDATE papers_text SET title = NULL WHERE corpus_id = 12")
    con.executemany("INSERT INTO papers_meta VALUES (?, ?, ?)", [
        (10, 2017, 5000), (11, 2019, 12), (12, 2020, 3),
    ])
    con.close()
    return db


def _make_edges(tmp_path) -> Path:
    p = tmp_path / "edges.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE e(src BIGINT, dst BIGINT)")
    # seed 1 cites 10, 11, 12 ; seed 2 cites nothing in-corpus
    con.executemany("INSERT INTO e VALUES (?, ?)", [(1, 10), (1, 11), (1, 12)])
    con.execute(f"COPY e TO '{p}' (FORMAT parquet)")
    con.close()
    return p


def test_attach_cited_orders_by_citationcount_and_drops_untitled(tmp_path):
    db = _make_db(tmp_path)
    edges = _make_edges(tmp_path)
    papers = [{"corpus_id": 1, "title": "Seed", "body": "x", "year": 2021, "venue": "V"}]
    out = _attach_cited(db, edges, papers, limit=8)
    assert len(out) == 1
    cited = out[0]["cited"]
    # corpus_id 12 dropped (null title); 11 (cc=12) before 10 (cc=5000)
    assert [c["corpus_id"] for c in cited] == [11, 10]
    assert cited[0]["title"] == "Niche prior work"


def test_attach_cited_drops_seed_with_no_in_corpus_cites(tmp_path):
    db = _make_db(tmp_path)
    edges = _make_edges(tmp_path)
    papers = [{"corpus_id": 2, "title": "Seed2", "body": "x"}]  # cites nothing
    assert _attach_cited(db, edges, papers, limit=8) == []


def test_done_ids_unions_qa_attempted_and_legacy_shards(tmp_path):
    # QA jsonl: papers that produced a QA
    (tmp_path / "single_hop.jsonl").write_text(
        json.dumps({"qa_id": "a", "seed_paper_ids": [1]}) + "\n"
        + json.dumps({"qa_id": "b", "seed_paper_ids": [2]}) + "\n"
    )
    # .attempted: every attempted paper (incl. ones that yielded no QA, e.g. 3)
    (tmp_path / "single_hop.attempted").write_text("1\n2\n3\n")
    # a legacy shard file from the old shard-based format
    (tmp_path / "single_hop_0000.jsonl").write_text(
        json.dumps({"qa_id": "c", "seed_paper_ids": [9]}) + "\n"
    )
    assert _done_ids(tmp_path) == {1, 2, 3, 9}


def test_done_ids_empty_dir(tmp_path):
    assert _done_ids(tmp_path) == set()
