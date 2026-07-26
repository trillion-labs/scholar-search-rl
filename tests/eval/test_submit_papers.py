from s2cs.eval.submit_papers import make_submit_papers, parse_submitted_ids


def test_submit_papers_roundtrips_ranked_ids():
    submit = make_submit_papers()
    sub = submit([3, 1, 2])
    assert parse_submitted_ids(sub.answer) == [3, 1, 2]


def test_parse_salvages_array_from_prose():
    assert parse_submitted_ids("top matches: [10, 20, 30] — done") == [10, 20, 30]


def test_parse_handles_string_ids():
    assert parse_submitted_ids('["100", "200"]') == [100, 200]


def test_parse_empty_or_garbage_returns_empty():
    assert parse_submitted_ids(None) == []
    assert parse_submitted_ids("") == []
    assert parse_submitted_ids("no ids here") == []
