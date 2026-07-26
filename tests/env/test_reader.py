from pathlib import Path

import duckdb
import pytest

from s2cs.env.reader import PaperReader


@pytest.fixture
def papers_db(tmp_path: Path) -> Path:
    path = tmp_path / "papers.duckdb"
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE papers_text(
            corpus_id BIGINT,
            title VARCHAR,
            abstract VARCHAR,
            summary VARCHAR,
            body VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO papers_text VALUES
            (1, 'Attention is all you need', 'self-attention transformer', 'sum1', 'line one\nline two\nline three\nFlashAttention appears here\nline five'),
            (2, 'Other paper', 'abs2', NULL, NULL),
            (3, 'Empty body', NULL, NULL, '')
    """)
    con.close()
    return path


def test_meta_present(papers_db):
    r = PaperReader(papers_db)
    m = r.meta(1)
    assert m is not None
    assert m.corpus_id == 1
    assert m.title == "Attention is all you need"
    assert m.abstract == "self-attention transformer"


def test_meta_missing(papers_db):
    r = PaperReader(papers_db)
    assert r.meta(999) is None


def test_read_full(papers_db):
    r = PaperReader(papers_db)
    res = r.read(1, offset=0, limit=200)
    assert res.total_lines == 5
    assert "line one" in res.text
    assert "FlashAttention" in res.text


def test_read_with_offset(papers_db):
    r = PaperReader(papers_db)
    res = r.read(1, offset=2, limit=2)
    assert res.text.split("\n")[0] == "line three"
    assert res.total_lines == 5


def test_read_empty_body(papers_db):
    r = PaperReader(papers_db)
    res = r.read(3, offset=0, limit=10)
    assert res.total_lines == 0
    assert res.text == ""


def test_find_in_paper_matches(papers_db):
    r = PaperReader(papers_db)
    matches = r.find_in_paper(1, "FlashAttention")
    assert len(matches) == 1
    assert matches[0].line_number == 3
    assert "FlashAttention" in matches[0].line


def test_find_in_paper_with_context(papers_db):
    r = PaperReader(papers_db)
    matches = r.find_in_paper(1, "FlashAttention", context=1)
    assert matches[0].context_before == ["line three"]
    assert matches[0].context_after == ["line five"]


def test_find_in_paper_no_match(papers_db):
    r = PaperReader(papers_db)
    assert r.find_in_paper(1, "nonexistent-token-xyz") == []


def test_find_in_paper_case_insensitive(papers_db):
    r = PaperReader(papers_db)
    assert len(r.find_in_paper(1, "flashattention")) == 1


def test_find_in_paper_max_hits(papers_db):
    r = PaperReader(papers_db)
    matches = r.find_in_paper(1, "line", max_hits=2)
    assert len(matches) == 2
