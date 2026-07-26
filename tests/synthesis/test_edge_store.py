from s2cs.synthesis.edge_store import Edge, append_edges, load_edges


def test_append_dedups_and_load_indexes(tmp_path):
    p = tmp_path / "edges.jsonl"
    seen: set[tuple[int, int]] = set()
    e1 = Edge(1, 2, "A cites B here.", "the baseline it compares against")
    e2 = Edge(1, 3, "A also cites C.", "the dataset it reuses")
    assert append_edges(p, [e1, e2, e1], seen) == 2  # duplicate e1 dropped
    assert append_edges(p, [e1], seen) == 0  # already seen across calls
    idx = load_edges(p)
    assert {e.to_id for e in idx[1]} == {2, 3}
    assert idx[1][0].citing_evidence == "A cites B here."
    assert load_edges(tmp_path / "missing.jsonl") == {}  # absent file -> empty
