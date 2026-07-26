from s2cs.env.etl.build_embeddings_chunk import (
    _chunk_id,
    _chunk_paper,
    _slide,
)


def test_slide_empty():
    assert _slide("", 100, 10, 5) == []


def test_slide_shorter_than_min():
    assert _slide("abc", 100, 10, 5) == []


def test_slide_under_window_returns_single_span():
    text = "x" * 50
    spans = _slide(text, window=100, stride=10, min_chars=5)
    assert spans == [(0, 50)]


def test_slide_exact_window():
    text = "x" * 100
    assert _slide(text, window=100, stride=10, min_chars=5) == [(0, 100)]


def test_slide_long_text_emits_strided_windows():
    text = "a" * 350
    spans = _slide(text, window=100, stride=20, min_chars=10)
    starts = [s for s, _ in spans]
    assert starts[0] == 0
    assert all(b - a == 80 for a, b in zip(starts, starts[1:]))
    assert spans[-1][1] == 350


def test_slide_drops_tail_below_min_chars():
    text = "a" * 105
    spans = _slide(text, window=50, stride=10, min_chars=20)
    assert spans[-1][1] == 105
    assert all(e - s >= 20 for s, e in spans)


def test_chunk_id_deterministic():
    a = _chunk_id(123, 0, 0, 100)
    b = _chunk_id(123, 0, 0, 100)
    assert a == b


def test_chunk_id_distinct_by_corpus():
    assert _chunk_id(1, 0, 0, 100) != _chunk_id(2, 0, 0, 100)


def test_chunk_id_distinct_by_section():
    assert _chunk_id(1, 0, 0, 100) != _chunk_id(1, 1, 0, 100)


def test_chunk_id_distinct_by_span():
    assert _chunk_id(1, 0, 0, 100) != _chunk_id(1, 0, 50, 150)


def test_chunk_paper_abstract_only():
    chunks = _chunk_paper(42, "x" * 200, None, window=100, stride=10, min_chars=5)
    assert all(c.section_idx == -1 for c in chunks)
    assert all(c.paper_corpus_id == 42 for c in chunks)
    assert len(chunks) >= 2


def test_chunk_paper_sections_only():
    sections = [{"content": "y" * 250, "title": "Intro"}, {"content": "z" * 50, "title": "Body"}]
    chunks = _chunk_paper(42, None, sections, window=100, stride=10, min_chars=10)
    section_indices = {c.section_idx for c in chunks}
    assert section_indices == {0, 1}


def test_chunk_paper_skips_empty_section_content():
    sections = [{"content": None, "title": "X"}, {"content": "abc", "title": "Y"}]
    chunks = _chunk_paper(42, None, sections, window=100, stride=10, min_chars=2)
    assert all(c.section_idx == 1 for c in chunks)


def test_chunk_paper_none_inputs():
    assert _chunk_paper(42, None, None, window=100, stride=10, min_chars=5) == []
