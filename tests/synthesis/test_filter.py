import asyncio

from s2cs.env.tools.search_papers import PaperHit
from s2cs.env.tools.search_snippets import SnippetHit
from s2cs.synthesis.filter import make_search_query, requires_retrieval, retrievable, rubric_verify
from s2cs.synthesis.single_hop import SingleHopQA
from tests.agent.conftest import make_fake_resp

QA = SingleHopQA(
    qa_id="single_hop_42_named_value",
    question="What learning rate trains the model?",
    answer="3e-4",
    seed_paper_id=42,
    evidence="learning rate of 3e-4",
    anchor="named",
    answer_type="value",
)


def test_drops_when_closed_book_correct(fake_openai_client):
    # 1st create() = closed-book answer; 2nd = judge verdict
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="3e-4"),
        make_fake_resp(content='{"reasoning": "matches", "judgment": "Correct"}'),
    ]
    keep = asyncio.run(requires_retrieval(QA, client=fake_openai_client, model="m"))
    assert keep is False


def test_keeps_when_closed_book_wrong(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="I don't know"),
        make_fake_resp(content='{"reasoning": "no match", "judgment": "Incorrect"}'),
    ]
    keep = asyncio.run(requires_retrieval(QA, client=fake_openai_client, model="m"))
    assert keep is True


def test_rubric_verify_all_pass(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='{"grounded": true, "closed_form": true, "unique": true, "own_work": true, "findable": true, "reasoning": "ok"}'
    )
    v = asyncio.run(rubric_verify(QA, "body with learning rate of 3e-4", client=fake_openai_client, model="m"))
    assert v.passed is True


def test_rubric_verify_one_fail_blocks_pass(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='{"grounded": true, "closed_form": false, "unique": true, "own_work": true, "findable": true, "reasoning": "descriptive"}'
    )
    v = asyncio.run(rubric_verify(QA, "body", client=fake_openai_client, model="m"))
    assert v.passed is False
    assert v.closed_form is False


def test_rubric_verify_json_failure_fails_closed(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="not json"),
        make_fake_resp(content="still not"),
        make_fake_resp(content="nope"),
    ]
    v = asyncio.run(rubric_verify(QA, "body", client=fake_openai_client, model="m"))
    assert v.passed is False


def _ph(cid):
    return PaperHit(corpus_id=cid, year=None, venue=None, citation_count=0, score=0.0)


def _sh(pid, chunk=1, sec=0):
    return SnippetHit(chunk_id=chunk, paper_corpus_id=pid, section_idx=sec, score=0.0)


def test_make_search_query_strips_surrounding_quotes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content='"ReachMedia accelerometer"')
    q = asyncio.run(make_search_query("In the ReachMedia system, what sampling rate?",
                                       client=fake_openai_client, model="m"))
    assert q == "ReachMedia accelerometer"


def test_retrievable_found_in_paper_search(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content="ReachMedia")
    paper_hits = [_ph(99), _ph(42)]  # seed=42 at rank 2
    r = asyncio.run(retrievable(QA, lambda q, limit=10: paper_hits[:limit],
                                 lambda q, limit=10: [],
                                 client=fake_openai_client, query_model="m", k=10))
    assert r.found is True
    assert r.paper_rank == 2
    assert r.snippet_rank is None
    assert r.query == "ReachMedia"


def test_retrievable_found_via_snippet_with_paper_dedup(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content="x")
    # 99 appears twice (dedup -> 1 paper), then 42 -> dedup rank 2
    snippet_hits = [_sh(99, chunk=1), _sh(99, chunk=2), _sh(42, chunk=3)]
    r = asyncio.run(retrievable(QA, lambda q, limit=10: [],
                                 lambda q, limit=10: snippet_hits[:limit],
                                 client=fake_openai_client, query_model="m", k=10))
    assert r.found is True
    assert r.paper_rank is None
    assert r.snippet_rank == 2


def test_retrievable_not_found_when_seed_absent(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content="x")
    r = asyncio.run(retrievable(QA, lambda q, limit=10: [_ph(7), _ph(99)],
                                 lambda q, limit=10: [_sh(7), _sh(99)],
                                 client=fake_openai_client, query_model="m", k=10))
    assert r.found is False
    assert r.paper_rank is None
    assert r.snippet_rank is None
