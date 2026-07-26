from s2cs.env.tools.find_in_paper import make_find_in_paper
from s2cs.env.tools.find_similar import make_find_similar
from s2cs.env.tools.list_citations import make_list_citations
from s2cs.env.tools.list_references import PaperRef, make_list_references
from s2cs.env.tools.paper_info import make_paper_info
from s2cs.env.tools.read_paper import make_read_paper
from s2cs.env.tools.search_papers import PaperHit, make_search_papers
from s2cs.env.tools.search_snippets import SnippetHit, make_search_snippets
from s2cs.env.tools.submit_answer import AnswerSubmission, make_submit_answer


def test_search_papers_hybrid(mock_papers_collection, encode_hybrid_fake):
    search_papers = make_search_papers(mock_papers_collection, encode_hybrid_fake)
    hits = search_papers("attention is all you need", limit=2)
    assert len(hits) == 2
    assert all(isinstance(h, PaperHit) for h in hits)
    assert hits[0].corpus_id == 1


def test_search_papers_mask(mock_papers_collection, encode_hybrid_fake):
    search_papers = make_search_papers(mock_papers_collection, encode_hybrid_fake)
    hits = search_papers("q", limit=2, mask_paper_ids={1})
    assert all(h.corpus_id != 1 for h in hits)


def test_search_papers_dense_only_when_no_sparse(mock_papers_collection):
    # An encoder with no learned-sparse (e.g. OpenAIEmbeddings) returns an empty
    # sparse dict -> search_papers must take the plain dense path, not hybrid_search.
    search_papers = make_search_papers(mock_papers_collection, lambda q: ([0.0] * 1024, {}))
    hits = search_papers("q", limit=2)
    assert [h.corpus_id for h in hits] == [1, 2]
    mock_papers_collection.search.assert_called_once()
    mock_papers_collection.hybrid_search.assert_not_called()


def test_search_snippets(mock_chunks_collection, encode_dense_fake):
    search_snippets = make_search_snippets(mock_chunks_collection, encode_dense_fake)
    hits = search_snippets("flash attention", limit=2)
    assert len(hits) == 2
    assert all(isinstance(h, SnippetHit) for h in hits)


def test_read_paper_body(mock_reader):
    read_paper = make_read_paper(mock_reader)
    r = read_paper(1, offset=0, limit=10)
    assert r.paper_id == 1
    assert r.total_lines == 3
    mock_reader.read.assert_called_with(1, section_idx=None, offset=0, limit=10)


def test_read_paper_with_section(mock_reader):
    read_paper = make_read_paper(mock_reader)
    read_paper(1, section_idx=2, offset=0, limit=50)
    mock_reader.read.assert_called_with(1, section_idx=2, offset=0, limit=50)


def test_find_in_paper(mock_reader):
    find_in_paper = make_find_in_paper(mock_reader)
    hits = find_in_paper(1, "match")
    assert len(hits) == 1
    assert hits[0].line_number == 1


def test_list_references(mock_graph):
    list_references = make_list_references(mock_graph)
    refs = list_references(1)
    assert refs == [PaperRef(10), PaperRef(11), PaperRef(12)]


def test_list_citations(mock_graph):
    list_citations = make_list_citations(mock_graph)
    citers = list_citations(1, limit=20)
    assert citers == [PaperRef(20), PaperRef(21)]


def test_find_similar_semantic(mock_papers_collection, mock_graph):
    find_similar = make_find_similar(mock_papers_collection, mock_graph)
    hits = find_similar(1, kind="semantic", limit=2)
    assert all(h.corpus_id != 1 for h in hits)
    assert all(isinstance(h, PaperHit) for h in hits)


def test_find_similar_co_citation(mock_papers_collection, mock_graph):
    mock_graph.cited_by.side_effect = lambda pid: {1: [100, 101]}.get(pid, [])
    mock_graph.references.side_effect = lambda pid: {100: [1, 2, 3], 101: [1, 2, 4]}.get(pid, [])
    find_similar = make_find_similar(mock_papers_collection, mock_graph)
    hits = find_similar(1, kind="co_citation", limit=10)
    cids = [h.corpus_id for h in hits]
    assert 2 in cids
    assert 1 not in cids


def test_find_similar_bibliographic_coupling(mock_papers_collection, mock_graph):
    mock_graph.references.side_effect = lambda pid: {1: [10, 11]}.get(pid, [])
    mock_graph.cited_by.side_effect = lambda pid: {10: [1, 2, 3], 11: [1, 2]}.get(pid, [])
    find_similar = make_find_similar(mock_papers_collection, mock_graph)
    hits = find_similar(1, kind="bibliographic_coupling", limit=10)
    cids = [h.corpus_id for h in hits]
    assert 2 in cids
    assert 1 not in cids


def test_paper_info(mock_reader):
    paper_info = make_paper_info(mock_reader)
    m = paper_info(1)
    assert m is not None
    assert m.corpus_id == 1
    assert m.classification == "Methods Paper"
    assert len(m.sections) == 2


def test_submit_answer():
    submit_answer = make_submit_answer()
    result = submit_answer("42")
    assert isinstance(result, AnswerSubmission)
    assert result.answer == "42"
