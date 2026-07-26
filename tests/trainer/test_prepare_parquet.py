from s2cs.trainer.data.prepare_parquet import to_verl_row, write_parquet


def test_qa_row():
    r = to_verl_row(
        {"query": "what color?", "gold_answer": "blue", "sample_id": "q1"},
        0,
        ["search_papers", "submit_answer"],
    )
    assert r["data_source"] == "s2cs_qa"
    assert r["agent_name"] == "tool_agent"
    assert r["reward_model"]["ground_truth"] == "blue"
    assert r["prompt"][0]["role"] == "system"
    assert r["prompt"][-1]["role"] == "user"
    assert r["prompt"][-1]["content"] == "what color?"
    assert r["extra_info"]["question"] == "what color?"
    assert r["extra_info"]["answer_type"] == "qa"
    assert "search_papers" in r["extra_info"]["tools_kwargs"]


def test_paper_set_row_carries_relevance_and_seed_mask():
    r = to_verl_row(
        {
            "query": "find papers",
            "answer_type": "paper_set",
            "relevance": {"1": 3},
            "est_total_relevant": 1,
            "seed_paper_ids": [9],
        },
        1,
        ["search_papers", "submit_answer"],
    )
    assert r["data_source"] == "s2cs_paper_set"
    assert r["reward_model"]["ground_truth"] == ""
    assert r["extra_info"]["answer_type"] == "paper_set"
    assert r["extra_info"]["relevance"] == {"1": 3}
    assert r["extra_info"]["est_total_relevant"] == 1
    assert r["extra_info"]["tools_kwargs"]["search_papers"]["create_kwargs"]["seed_paper_ids"] == [9]


def test_qa_and_paper_set_use_different_system_prompts():
    qa = to_verl_row({"query": "q", "gold_answer": "a"}, 0, ["submit_answer"])
    ps = to_verl_row({"query": "q", "answer_type": "paper_set", "relevance": {}, "est_total_relevant": 0}, 1, ["submit_answer"])
    assert qa["prompt"][0]["content"] != ps["prompt"][0]["content"]


def test_write_parquet_roundtrip(tmp_path):
    import json

    import pandas as pd

    src = tmp_path / "pool.jsonl"
    src.write_text(
        json.dumps({"query": "q1", "gold_answer": "a1"}) + "\n" + json.dumps({"query": "q2", "gold_answer": "a2"}) + "\n"
    )
    out = tmp_path / "out.parquet"
    n = write_parquet(str(src), str(out), ["search_papers", "submit_answer"])
    assert n == 2
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["data_source"]) == {"s2cs_qa"}
