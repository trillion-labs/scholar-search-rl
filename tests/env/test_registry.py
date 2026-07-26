import pytest

from s2cs.env.tools.registry import build_registry, compose


def test_build_registry_names(mock_papers_collection, mock_chunks_collection,
                               mock_graph, mock_reader, fake_encoder):
    reg = build_registry(
        papers=mock_papers_collection,
        chunks=mock_chunks_collection,
        graph=mock_graph,
        reader=mock_reader,
        encoder=fake_encoder,
    )
    expected = {
        "search_papers", "search_snippets", "read_paper", "find_in_paper",
        "list_references", "list_citations", "find_similar", "paper_info",
        "submit_answer",
    }
    assert set(reg.keys()) == expected


def test_compose_subset(mock_papers_collection, mock_chunks_collection,
                        mock_graph, mock_reader, fake_encoder):
    reg = build_registry(
        papers=mock_papers_collection, chunks=mock_chunks_collection,
        graph=mock_graph, reader=mock_reader, encoder=fake_encoder,
    )
    subset = compose(reg, ["search_papers", "submit_answer"])
    assert set(subset.keys()) == {"search_papers", "submit_answer"}


def test_compose_unknown_raises(mock_papers_collection, mock_chunks_collection,
                                 mock_graph, mock_reader, fake_encoder):
    reg = build_registry(
        papers=mock_papers_collection, chunks=mock_chunks_collection,
        graph=mock_graph, reader=mock_reader, encoder=fake_encoder,
    )
    with pytest.raises(KeyError):
        compose(reg, ["search_papers", "nonexistent"])
