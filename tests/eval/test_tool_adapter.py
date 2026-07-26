import pytest

from s2cs.eval.astabench.tool_adapter import adapt_tools, s2cs_tool_specs


def _fake_asta():
    async def search_papers_by_relevance(keyword=None, limit=10): return {"keyword": keyword, "limit": limit}
    async def snippet_search(query=None, limit=10, corpus_ids=None): return {"query": query, "corpus_ids": corpus_ids}
    async def get_paper(paper_id=None, fields=None): return {"paper_id": paper_id, "fields": fields}
    async def get_citations(paper_id=None, limit=None): return {"paper_id": paper_id, "limit": limit}
    return {
        "search_papers_by_relevance": search_papers_by_relevance,
        "snippet_search": snippet_search,
        "get_paper": get_paper,
        "get_citations": get_citations,
    }


def test_common_tools_present_both_modes():
    for mode in ("strict", "body_revive"):
        specs, tools = adapt_tools(_fake_asta(), align_mode=mode)
        names = {s["function"]["name"] for s in specs}
        assert {"search_papers", "search_snippets", "paper_info", "list_citations"} <= names
        assert set(tools) == names


def _spec_names(specs):
    return {s["function"]["name"] for s in specs}


def _params(specs, name):
    s = next(s for s in specs if s["function"]["name"] == name)
    return set(s["function"]["parameters"]["properties"])


def test_s2cs_specs_match_training_surface():
    names = _spec_names(s2cs_tool_specs())
    # the training retrieval tool surface (submit_answer is added by the solver)
    assert {"search_papers", "search_snippets", "read_paper", "find_in_paper",
            "list_references", "list_citations", "find_similar", "paper_info"} <= names
    assert "submit_answer" not in names
    # s2cs param names (not Asta's): search_papers takes `query`, read_paper takes `paper_id`
    assert "query" in _params(s2cs_tool_specs(), "search_papers")
    assert "paper_id" in _params(s2cs_tool_specs(), "read_paper")


@pytest.fixture
def fake_asta():
    calls = []

    async def search_papers_by_relevance(keyword, limit=10, **_):
        calls.append(("search_papers_by_relevance", {"keyword": keyword, "limit": limit}))
        return [{"paperId": "p1", "title": keyword}]

    async def snippet_search(query, limit=10, corpus_ids=None, **_):
        calls.append(("snippet_search", {"query": query, "limit": limit, "corpus_ids": corpus_ids}))
        return [{"snippet": f"about {query}"}]

    async def get_paper(paper_id, **_):
        calls.append(("get_paper", {"paper_id": paper_id}))
        return {"paperId": paper_id, "text": "FULL BODY TEXT"}

    async def get_citations(paper_id, limit=20, **_):
        calls.append(("get_citations", {"paper_id": paper_id, "limit": limit}))
        return [{"paperId": "c1"}]

    by_name = {
        "search_papers_by_relevance": search_papers_by_relevance,
        "snippet_search": snippet_search,
        "get_paper": get_paper,
        "get_citations": get_citations,
    }
    return by_name, calls


def test_model_sees_s2cs_specs_for_available_tools(fake_asta):
    by_name, _ = fake_asta
    specs, _tools = adapt_tools(by_name)
    names = _spec_names(specs)
    # tools backed by an available Asta tool are exposed with their s2cs names
    assert {"search_papers", "search_snippets", "read_paper", "find_in_paper", "paper_info"} <= names
    assert "search_papers_by_relevance" not in names  # the Asta name is hidden


@pytest.mark.asyncio
async def test_search_papers_routes_to_asta_with_translated_args(fake_asta):
    by_name, calls = fake_asta
    _specs, tools = adapt_tools(by_name)
    out = await tools["search_papers"](query="gliogenic switch", limit=5)
    assert calls[-1] == ("search_papers_by_relevance", {"keyword": "gliogenic switch", "limit": 5})
    assert out == [{"paperId": "p1", "title": "gliogenic switch"}]  # Asta result passed through


@pytest.mark.asyncio
async def test_read_paper_routes_to_snippet_search(fake_asta):
    # body_revive mode: read_paper -> snippet_search (the real body channel)
    by_name, calls = fake_asta
    _specs, tools = adapt_tools(by_name, align_mode="body_revive", question="what is attention?")
    out = await tools["read_paper"](paper_id="abc123")
    assert calls[-1][0] == "snippet_search"
    # task question used as the snippet query when read_paper has no query of its own
    assert calls[-1][1]["query"] == "what is attention?"
    assert calls[-1][1]["corpus_ids"] == ["abc123"]


@pytest.mark.asyncio
async def test_find_in_paper_routes_to_snippet_search(fake_asta):
    # body_revive mode: find_in_paper -> snippet_search with pattern as query
    by_name, calls = fake_asta
    _specs, tools = adapt_tools(by_name, align_mode="body_revive")
    out = await tools["find_in_paper"](paper_id="abc123", pattern="loop")
    assert calls[-1][0] == "snippet_search"
    assert calls[-1][1]["query"] == "loop"
    assert calls[-1][1]["corpus_ids"] == ["abc123"]


def test_unavailable_asta_backend_tool_is_dropped():
    # only get_paper available in body_revive mode → read_paper (needs snippet_search)
    # is NOT exposed; paper_info (backed by get_paper) IS exposed
    async def get_paper(paper_id, **_):
        return {"paperId": paper_id}
    specs, tools = adapt_tools({"get_paper": get_paper})
    names = _spec_names(specs)
    assert "paper_info" in names
    assert "read_paper" not in names  # snippet_search unavailable → read_paper dropped
    assert "search_papers" not in names


@pytest.mark.asyncio
async def test_search_snippets_scopes_when_paper_ids_given():
    # search_snippets(paper_ids=...) -> snippet_search(corpus_ids=...), ids stringified
    _specs, tools = adapt_tools(_fake_asta(), align_mode="body_revive")
    out = await tools["search_snippets"](query="x", paper_ids=[1, 2])
    assert out["corpus_ids"] == ["1", "2"]


@pytest.mark.asyncio
async def test_snippet_unscoped_when_tool_has_no_scope_param():
    # snippet tool without a corpus-scope param -> no scope key injected, no crash
    fake = _fake_asta()
    async def snippet_no_scope(query=None, limit=10):
        return {"query": query}
    fake["snippet_search"] = snippet_no_scope
    _specs, tools = adapt_tools(fake, align_mode="body_revive", question="q")
    out = await tools["read_paper"](paper_id=5)
    assert out == {"query": "q"}


def test_strict_mode_omits_fake_readers():
    # strict mode exposes no fake full-text readers; body channel is search_snippets only
    _specs, tools = adapt_tools(_fake_asta(), align_mode="strict")
    for absent in ("read_paper", "find_in_paper", "find_similar"):
        assert absent not in tools
    assert "search_snippets" in tools
