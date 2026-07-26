import asyncio
import json

import pytest

from s2cs.synthesis.multi_hop import (
    MultiHopDraft,
    MultiHopQA,
    make_multi_hop_select,
    ground_multi_hop_answer,
)
from tests.agent.conftest import make_fake_resp


def _resp(obj):
    return make_fake_resp(content=json.dumps(obj))


def test_to_record_schema():
    from s2cs.synthesis.edge_store import Edge
    qa = MultiHopQA(
        qa_id="multi_hop_100_7_value",
        question="In the paper reporting 75.3 accuracy for model X, what does the cited work it builds on report?",
        answer="performance dropped on the legal domain",
        answer_type="value",
        path=(100, 7),
        edges=(Edge(100, 7, "we reproduce model X (75.3 accuracy) following [3]",
                    "the open-domain QA work A builds its baseline on"),),
        evidence_b="after training on the search domain, accuracy on the legal domain dropped",
    )
    rec = qa.to_record()
    assert rec["source"] == "multi_hop"
    assert rec["hops"] == 2 and rec["path"] == [100, 7] and len(rec["edges"]) == 1
    assert rec["seed_paper_ids"] == [100]
    assert rec["intermediate_paper_id"] == 100 and rec["gold_paper_id"] == 7
    assert rec["evidence_a"] == "we reproduce model X (75.3 accuracy) following [3]"
    assert rec["question"] and rec["answer"]  # judged-reward needs both
    assert rec["anchor"] == "detail_cue" and rec["answer_type"] == "value"
    assert set(rec) == {
        "qa_id", "source", "question", "answer", "answer_type", "anchor",
        "hops", "path", "edges", "seed_paper_ids", "intermediate_paper_id",
        "gold_paper_id", "evidence_a", "evidence_b", "rubric_pass", "stage", "pass_at_8",
    }


A_BODY = (
    "We reproduce the strong baseline of 75.3 accuracy reported for the dual-encoder "
    "retriever, following the in-batch-negatives recipe of the open-domain QA work we "
    "build on. Our analysis extends that setup to three new domains."
)
EVIDENCE_A = "following the in-batch-negatives recipe of the open-domain QA work we build on"

CITED = [
    {"corpus_id": 11, "title": "Continuous wavelet transform for spindle detection",
     "abstract": "A signal-processing method for sleep spindles."},
    {"corpus_id": 7, "title": "Dense Passage Retrieval for open-domain QA",
     "abstract": "A dual-encoder trained with in-batch negatives for retrieval."},
]

A_PAPER = {
    "corpus_id": 100,
    "title": "Domain transfer for neural retrieval",
    "abstract": "We study how retrievers transfer across domains.",
    "summary": "Cross-domain retrieval analysis.",
    "body": A_BODY,
    "cited": CITED,
}


def _run_select(client, paper=None, **kw):
    sel = make_multi_hop_select(client, "m", **kw)
    return asyncio.run(sel(paper if paper is not None else A_PAPER))


def test_discover_edge_returns_structural_edge(fake_openai_client):
    from s2cs.synthesis.edge_store import Edge
    from s2cs.synthesis.multi_hop import make_edge_discoverer

    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2), _resp(_PASSAGE_OK), _resp(_VERIFY_YES)]  # no anchor call
    discover = make_edge_discoverer(fake_openai_client, "m")
    edge = asyncio.run(discover(A_PAPER))
    assert isinstance(edge, Edge)
    assert edge.from_id == 100 and edge.to_id == 7
    assert edge.citing_evidence == EVIDENCE_A
    assert edge.pointer_label == _PASSAGE_OK["b_context"]


def test_make_anchor_returns_cue(fake_openai_client):
    from s2cs.synthesis.multi_hop import make_anchor

    fake_openai_client.chat.completions.create.return_value = _resp(_ANCHOR_OK)
    anchor = make_anchor(fake_openai_client, "m", a_anchor="detail_cue")
    assert asyncio.run(anchor(A_PAPER)) == _ANCHOR_OK["a_cue"]


# SELECT is B-first: CHOOSE (C-label) -> PASSAGE (ground in named B) -> VERIFY (bridge judge) -> A-anchor.
_CHOOSE_C2 = {"cited_label": "C2"}
_PASSAGE_OK = {"evidence_a": EVIDENCE_A,
               "b_context": "the open-domain QA retrieval work A builds its 75.3 baseline on"}
_VERIFY_YES = {"match": True, "reason": "DPR matches the in-batch-negatives QA retriever"}
_VERIFY_NO = {"match": False, "reason": "passage is about a different work"}
_ANCHOR_OK = {"a_cue": "the paper reporting a 75.3 dual-encoder baseline on domain transfer"}


def test_select_passes_and_resolves_b(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2), _resp(_PASSAGE_OK), _resp(_VERIFY_YES), _resp(_ANCHOR_OK)]
    draft = _run_select(fake_openai_client)
    assert draft is not None
    assert draft.intermediate_paper_id == 100
    assert draft.gold_paper_id == 7          # C2 -> CITED[1].corpus_id
    assert draft.evidence_a == EVIDENCE_A
    assert draft.a_cue == _ANCHOR_OK["a_cue"]
    assert draft.a_anchor == "detail_cue"    # default tier


def test_select_rejects_when_verify_no(fake_openai_client):
    # B chosen + passage grounded, but the bridge verifier says evidence_a is NOT about B
    # (PASSAGE fabricated) -> drop. This is the gate that makes B-first safe.
    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2), _resp(_PASSAGE_OK), _resp(_VERIFY_NO)]
    assert _run_select(fake_openai_client) is None


def test_select_named_anchor_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2), _resp(_PASSAGE_OK), _resp(_VERIFY_YES),
        _resp({"a_cue": "the Domain transfer for neural retrieval paper"})]
    draft = _run_select(fake_openai_client, a_anchor="named")
    assert draft is not None and draft.a_anchor == "named"


def test_select_unknown_a_anchor_raises():
    with pytest.raises(ValueError):
        make_multi_hop_select(object(), "m", a_anchor="bogus")


def test_select_rejects_when_a_cue_empty(fake_openai_client):
    # CHOOSE + PASSAGE + verify pass, but A-anchor yields no cue -> drop
    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2), _resp(_PASSAGE_OK), _resp(_VERIFY_YES), _resp({})]
    assert _run_select(fake_openai_client) is None


def test_select_rejects_evidence_absent_from_a_body(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        _resp(_CHOOSE_C2),
        _resp({"evidence_a": "a quote that never appears in A's body", "b_context": "the QA work"})]
    assert _run_select(fake_openai_client) is None


def test_select_rejects_out_of_range_label(fake_openai_client):
    # CHOOSE returns an out-of-range C-number -> reject before PASSAGE
    fake_openai_client.chat.completions.create.return_value = _resp({"cited_label": "C9"})
    assert _run_select(fake_openai_client) is None


def test_select_skips_paper_without_cited(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp(_CHOOSE_C2)
    paper = {**A_PAPER}
    paper.pop("cited")
    assert _run_select(fake_openai_client, paper=paper) is None


def test_select_rejects_when_passage_empty(fake_openai_client):
    # B chosen, but PASSAGE can't ground it -> {} -> clean miss (no broken bridge)
    fake_openai_client.chat.completions.create.side_effect = [_resp(_CHOOSE_C2), _resp({})]
    assert _run_select(fake_openai_client) is None


def test_select_returns_none_on_json_failure(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="not json"),
        make_fake_resp(content="still not"),
        make_fake_resp(content="nope"),
    ]
    assert _run_select(fake_openai_client) is None


B_TITLE = "Dense Passage Retrieval for open-domain QA"
B_ABSTRACT = "A dual-encoder trained with in-batch negatives for retrieval."
B_BODY = (
    "We train on the search domain. After domain transfer, accuracy on the legal "
    "domain dropped by nine points, the largest regression we observed."
)
DRAFT = MultiHopDraft(
    intermediate_paper_id=100, gold_paper_id=7,
    evidence_a=EVIDENCE_A,
    b_label="the open-domain QA retrieval work A builds its 75.3 baseline on",
    a_cue="the paper reporting a 75.3 dual-encoder baseline on domain transfer",
    a_anchor="detail_cue",
)


def test_ground_chain_two_edges_composes_three_hop(fake_openai_client):
    from s2cs.synthesis.edge_store import Edge
    from s2cs.synthesis.multi_hop import ground_chain

    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "In the start work, via the baseline it compares against, of the corpus that work "
                    "in turn reuses, how many documents does that corpus contain?",
        "answer": "twelve",
        "evidence": "The corpus contains twelve documents.",
    })
    edges = [Edge(1, 2, "ev12", "the baseline it compares against"),
             Edge(2, 3, "ev23", "the corpus it reuses")]
    qa = asyncio.run(ground_chain("the start work", edges,
                                  b_title="Zeta Lexicon Resource", b_abstract="ab",
                                  b_body="The corpus contains twelve documents.",
                                  answer_type="value", client=fake_openai_client, model="m"))
    assert qa is not None
    assert qa.hops == 3 and qa.path == (1, 2, 3)
    assert qa.qa_id == "multi_hop_1_2_3_value"
    rec = qa.to_record()
    assert rec["intermediate_paper_id"] == 1 and rec["gold_paper_id"] == 3
    assert rec["seed_paper_ids"] == [1, 2] and len(rec["edges"]) == 2


def _run_ground(client, *, answer_type="value", **kw):
    return asyncio.run(ground_multi_hop_answer(
        DRAFT, b_title=B_TITLE, b_abstract=B_ABSTRACT, b_body=B_BODY,
        answer_type=answer_type, client=client, model="m", **kw,
    ))


def test_ground_value_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "In the paper reporting a 75.3 accuracy baseline for a dual-encoder, "
                    "consider the open-domain QA work it builds on: after training on the "
                    "search domain, which domain's accuracy dropped most?",
        "answer": "the legal domain",
        "evidence": "accuracy on the legal domain dropped by nine points",
    })
    qa = _run_ground(fake_openai_client, answer_type="value")
    assert qa is not None
    assert qa.qa_id == "multi_hop_100_7_value"
    assert qa.gold_paper_id == 7 and qa.answer_type == "value"
    assert qa.edges[0].citing_evidence == EVIDENCE_A


def test_ground_rejects_b_title_in_question(fake_openai_client):
    # pure-bridge violated: the question names B directly
    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "In Dense Passage Retrieval for open-domain QA, which domain dropped?",
        "answer": "the legal domain",
        "evidence": "accuracy on the legal domain dropped by nine points",
    })
    assert _run_ground(fake_openai_client, answer_type="value") is None


def test_ground_rejects_answer_leaked_into_question(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "Did the legal domain drop in the work A builds on?",
        "answer": "the legal domain",
        "evidence": "accuracy on the legal domain dropped by nine points",
    })
    assert _run_ground(fake_openai_client, answer_type="value") is None


def test_ground_rejects_evidence_absent_from_b_body(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "After training on search, which domain dropped in the work A builds on?",
        "answer": "the legal domain",
        "evidence": "a sentence that is not in B's body at all",
    })
    assert _run_ground(fake_openai_client, answer_type="value") is None


def test_ground_abstract_checks_against_b_abstract(fake_openai_client):
    # answer/evidence must be in B's ABSTRACT for answer_type=abstract; a body-only fact fails
    fake_openai_client.chat.completions.create.return_value = _resp({
        "question": "What kind of encoder is the work A builds on?",
        "answer": "a dual-encoder",
        "evidence": "A dual-encoder trained with in-batch negatives for retrieval.",
    })
    qa = _run_ground(fake_openai_client, answer_type="abstract")
    assert qa is not None and qa.answer_type == "abstract"


def test_ground_rejects_unknown_answer_type():
    with pytest.raises(ValueError):
        asyncio.run(ground_multi_hop_answer(
            DRAFT, b_title=B_TITLE, b_abstract=B_ABSTRACT, b_body=B_BODY,
            answer_type="bogus", client=object(), model="m",
        ))


# ---------- two-part retrievability check (A findable, B not) ----------

import s2cs.synthesis.multi_hop as mh


def _hop(monkeypatch, a_ranks, b_ranks, b_rank=3):
    # the A-check and B-check use different prompts; route the fake by which prompt is passed.
    # a_ranks / b_ranks map corpus_id -> best 1-indexed rank (what _surfaced_ranks returns).
    async def fake_ranks(question, prompt_tmpl, **k):
        return a_ranks if prompt_tmpl is mh.A_QUERY_PROMPT else b_ranks
    monkeypatch.setattr(mh, "_surfaced_ranks", fake_ranks)
    return asyncio.run(mh.check_hop(
        "q", 100, 7, search_papers=None, search_snippets=None, client=None, query_model="m", b_rank=b_rank,
    ))


def test_hop_valid_when_a_found_and_b_absent(monkeypatch):
    # A(100) surfaces from the A-prompt, B(7) does NOT surface from the B-prompt -> keep
    assert _hop(monkeypatch, {100: 1, 5: 2}, {5: 1, 9: 2}) == (True, False)


def test_hop_b_top_hit_is_shortcut(monkeypatch):
    # B(7) is the rank-1 hit from the adversarial B-prompt -> shortcuttable -> b_found True
    assert _hop(monkeypatch, {100: 1}, {7: 1, 9: 2}) == (True, True)


def test_hop_b_present_but_buried_is_not_shortcut(monkeypatch):
    # B(7) surfaces but only at rank 8 (> b_rank=3) -> topical neighbor, not pinnable -> keep
    assert _hop(monkeypatch, {100: 1}, {9: 1, 7: 8}) == (True, False)


def test_hop_a_not_findable(monkeypatch):
    # A(100) does not surface -> can't reach the entry point -> a_found False
    assert _hop(monkeypatch, {5: 1, 9: 2}, {9: 1}) == (False, False)


def test_probe_chain_start_terminal(monkeypatch):
    # probe_chain is the public name over (start, terminal); start(100) found, terminal(7) buried -> keep
    async def fake_ranks(question, prompt_tmpl, **k):
        return {100: 1} if prompt_tmpl is mh.A_QUERY_PROMPT else {7: 9}
    monkeypatch.setattr(mh, "_surfaced_ranks", fake_ranks)
    out = asyncio.run(mh.probe_chain("q", 100, 7, search_papers=None, search_snippets=None,
                                     client=None, query_model="m"))
    assert out == (True, False)
