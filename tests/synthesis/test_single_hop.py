import asyncio
import json

import pytest

from s2cs.synthesis.single_hop import make_single_hop_synth
from tests.agent.conftest import make_fake_resp

PAPER = {
    "corpus_id": 42,
    "title": "Efficient Transformers",
    "abstract": "We study attention efficiency.",
    "summary": "A method for efficient attention.",
    "body": (
        "In our experiments we train FlashConv with the AdamW optimizer at a learning rate "
        "of 3e-4 for 100 epochs on the WikiText-103 dataset."
    ),
}

# a real prose sentence from the body, for value evidence
PROSE = "In our experiments we train FlashConv with the AdamW optimizer at a learning rate of 3e-4"

CITED = [
    {"corpus_id": 7, "title": "Continuous wavelet transform for spindle detection", "year": 2015},
    {"corpus_id": 9, "title": "Crowdsourced spindle benchmarking", "year": 2014},
]


def _resp(obj):
    return make_fake_resp(content=json.dumps(obj))


def _run(client, **kw):
    synth = make_single_hop_synth(client, "m", **kw)
    return asyncio.run(synth(PAPER))


def _run_cited(client, paper_extra, **kw):
    synth = make_single_hop_synth(client, "m", **kw)
    return asyncio.run(synth({**PAPER, **paper_extra}))


# ---------- named × value ----------

def test_named_value_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What learning rate trains the efficient-attention model on WikiText-103?",
         "answer": "3e-4", "evidence": PROSE},
    ])
    qas = _run(fake_openai_client)
    assert len(qas) == 1
    qa = qas[0]
    assert qa.qa_id == "single_hop_42_named_value_0"
    assert qa.answer == "3e-4"
    assert qa.anchor == "named" and qa.answer_type == "value"


def test_rejects_evidence_absent_from_body(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Q?", "answer": "x", "evidence": "a quote that never appears in the body of this paper"},
    ])
    assert _run(fake_openai_client) == []


def test_rejects_missing_fields(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Q?", "answer": "", "evidence": PROSE},
    ])
    assert _run(fake_openai_client) == []


def test_value_rejects_non_prose_evidence(fake_openai_client):
    # evidence is in the body but is a bare token (table cell / number) — not prose
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "How many epochs?", "answer": "100", "evidence": "100"},
    ])
    assert _run(fake_openai_client) == []


def test_declined_empty_array(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([])
    assert _run(fake_openai_client) == []


def test_declined_empty_object(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp({})
    assert _run(fake_openai_client) == []


def test_json_failure_returns_empty(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="not json"),
        make_fake_resp(content="still not json"),
        make_fake_resp(content="nope"),
    ]
    assert _run(fake_openai_client) == []


def test_to_record_schema(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Q?", "answer": "3e-4", "evidence": PROSE},
    ])
    rec = _run(fake_openai_client)[0].to_record()
    assert rec["source"] == "single_hop"
    assert rec["seed_paper_ids"] == [42]
    assert rec["anchor"] == "named" and rec["answer_type"] == "value"
    assert set(rec) == {
        "qa_id", "source", "question", "answer", "seed_paper_ids",
        "evidence", "anchor", "answer_type", "cited_paper_ids", "rubric_pass", "stage", "pass_at_8",
    }


# ---------- up to 3 per call ----------

def test_three_distinct_per_call(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What learning rate is used?", "answer": "3e-4", "evidence": PROSE},
        {"question": "How many epochs are trained?", "answer": "100", "evidence": PROSE},
        {"question": "Which dataset is used?", "answer": "WikiText-103", "evidence": PROSE},
    ])
    qas = _run(fake_openai_client)
    assert [qa.qa_id for qa in qas] == [
        "single_hop_42_named_value_0", "single_hop_42_named_value_1", "single_hop_42_named_value_2",
    ]


def test_dedup_identical_questions(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What learning rate is used?", "answer": "3e-4", "evidence": PROSE},
        {"question": "What learning rate is used?", "answer": "3e-4", "evidence": PROSE},
    ])
    assert len(_run(fake_openai_client)) == 1


# ---------- cell whitelist ----------

def test_named_identity_is_rejected():
    with pytest.raises(ValueError):
        make_single_hop_synth(object(), "m", anchor="named", answer_type="identity")


def test_paraphrastic_identity_is_rejected():
    with pytest.raises(ValueError):
        make_single_hop_synth(object(), "m", anchor="paraphrastic", answer_type="identity")


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError):
        make_single_hop_synth(object(), "m", anchor="bogus")


# ---------- anchor anti-leak ----------

def test_non_named_anchor_rejects_title_in_question(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "In Efficient Transformers, what learning rate is used?", "answer": "3e-4", "evidence": PROSE},
    ])
    assert _run(fake_openai_client, anchor="detail_cue", answer_type="value") == []


# ---------- detail_cue × identity ----------

def test_detail_cue_identity_coined_name_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Which method, trained on WikiText-103 at a 3e-4 learning rate, makes attention efficient?",
         "answer": "FlashConv", "evidence": "In our experiments we train FlashConv with the AdamW optimizer"},
    ])
    qas = _run(fake_openai_client, anchor="detail_cue", answer_type="identity")
    assert len(qas) == 1
    assert qas[0].qa_id == "single_hop_42_detail_cue_identity_0"
    assert qas[0].answer == "FlashConv"


def test_detail_cue_identity_title_allowed(fake_openai_client):
    # identity now permits the title (the L3 anchor is disjoint from it — no leak)
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Which work, trained on WikiText-103 at a 3e-4 rate, studies making attention efficient?",
         "answer": "Efficient Transformers", "evidence": "We study attention efficiency"},
    ])
    qas = _run(fake_openai_client, anchor="detail_cue", answer_type="identity")
    assert len(qas) == 1
    assert qas[0].answer == "Efficient Transformers"


def test_identity_answer_leaked_into_question_is_rejected(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What does FlashConv achieve on WikiText-103?", "answer": "FlashConv",
         "evidence": "In our experiments we train FlashConv with the AdamW optimizer"},
    ])
    assert _run(fake_openai_client, anchor="detail_cue", answer_type="identity") == []


# ---------- context_conjunction (T2b) needs a venue ----------

def test_context_conjunction_skips_paper_without_venue(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "In a study on attention efficiency, what learning rate is used?",
         "answer": "3e-4", "evidence": PROSE},
    ])
    # PAPER carries no venue → T2b skips it before the LLM call (fills count from venue-bearing papers)
    assert _run(fake_openai_client, anchor="context_conjunction", answer_type="value") == []


def test_context_conjunction_with_venue_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "In a 2020 study on attention efficiency, what learning rate is used?",
         "answer": "3e-4", "evidence": PROSE},
    ])
    synth = make_single_hop_synth(fake_openai_client, "m", anchor="context_conjunction", answer_type="value")
    qas = asyncio.run(synth({**PAPER, "venue": "ICML", "year": 2020}))
    assert len(qas) == 1


# ---------- abstract ----------

def test_named_abstract_passes(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What efficiency aspect does the FlashConv paper study?",
         "answer": "attention efficiency", "evidence": "We study attention efficiency."},
    ])
    qas = _run(fake_openai_client, anchor="named", answer_type="abstract")
    assert len(qas) == 1
    assert qas[0].qa_id == "single_hop_42_named_abstract_0"
    assert qas[0].answer_type == "abstract"


def test_abstract_answer_must_be_in_abstract(fake_openai_client):
    # answer that is a body-only fact (not in PUBLIC) is wrong for the abstract rung
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "What learning rate is used?", "answer": "3e-4", "evidence": "We study attention efficiency."},
    ])
    assert _run(fake_openai_client, anchor="named", answer_type="abstract") == []


# ---------- T2c / T2d cell whitelist ----------

def test_relational_conjunction_cells_constructible():
    for at in ("value", "abstract", "identity"):
        make_single_hop_synth(object(), "m", anchor="relational_conjunction", answer_type=at)


def test_full_span_cells_constructible():
    for at in ("value", "abstract", "identity"):
        make_single_hop_synth(object(), "m", anchor="full_span", answer_type=at)


def test_relational_conjunction_passes_and_records_cited(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "A study citing the continuous wavelet transform spindle-detection work trains at what rate "
                     "on WikiText-103?",
         "answer": "3e-4", "evidence": PROSE, "cited_label": "C1"},
    ])
    qas = _run_cited(fake_openai_client, {"cited": CITED},
                     anchor="relational_conjunction", answer_type="value")
    assert len(qas) == 1
    assert qas[0].cited_paper_ids == (7,)
    assert qas[0].anchor == "relational_conjunction"


def test_full_span_passes_with_metadata_and_citation(fake_openai_client):
    # full_span combines the metadata path (needs venue) with the citation path
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Which 2020 ICML study, citing the crowdsourced spindle benchmarking work, "
                     "trains at a 3e-4 rate on WikiText-103?",
         "answer": "3e-4", "evidence": PROSE, "cited_label": "C2"},
    ])
    qas = _run_cited(fake_openai_client, {"cited": CITED, "venue": "ICML", "year": 2020},
                     anchor="full_span", answer_type="value")
    assert len(qas) == 1
    assert qas[0].cited_paper_ids == (9,)  # C2 -> CITED[1].corpus_id
    assert qas[0].anchor == "full_span"


def test_full_span_skips_paper_without_venue(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Which study citing the crowdsourced spindle benchmarking work trains at 3e-4?",
         "answer": "3e-4", "evidence": PROSE, "cited_label": "C2"},
    ])
    # PAPER carries no venue → full_span skips it before the LLM call
    assert _run_cited(fake_openai_client, {"cited": CITED},
                      anchor="full_span", answer_type="value") == []


def test_relational_conjunction_rejects_dangling_citation(fake_openai_client):
    # question mentions NO word from the cited title "Continuous wavelet transform..." -> dangling cue
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "A study that builds on a cited prior method trains at what rate on WikiText-103?",
         "answer": "3e-4", "evidence": PROSE, "cited_label": "C1"},
    ])
    assert _run_cited(fake_openai_client, {"cited": CITED},
                      anchor="relational_conjunction", answer_type="value") == []


def test_relational_conjunction_skips_paper_without_cited(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "Q?", "answer": "3e-4", "evidence": PROSE, "cited_label": "C1"},
    ])
    # no "cited" key on the paper -> skip before the LLM call
    assert _run(fake_openai_client, anchor="relational_conjunction", answer_type="value") == []


def test_relational_conjunction_rejects_out_of_range_label(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp([
        {"question": "A study citing the wavelet spindle work trains at what rate?",
         "answer": "3e-4", "evidence": PROSE, "cited_label": "C9"},  # out of range
    ])
    assert _run_cited(fake_openai_client, {"cited": CITED},
                      anchor="relational_conjunction", answer_type="value") == []


def test_to_record_includes_cited_paper_ids():
    from s2cs.synthesis.single_hop import SingleHopQA
    qa = SingleHopQA(
        qa_id="single_hop_42_relational_conjunction_value_0",
        question="Q?", answer="3e-4", seed_paper_id=42, evidence="x",
        anchor="relational_conjunction", answer_type="value", cited_paper_ids=(7,),
    )
    rec = qa.to_record()
    assert rec["cited_paper_ids"] == [7]
    assert set(rec) == {
        "qa_id", "source", "question", "answer", "seed_paper_ids",
        "evidence", "anchor", "answer_type", "cited_paper_ids",
        "rubric_pass", "stage", "pass_at_8",
    }
