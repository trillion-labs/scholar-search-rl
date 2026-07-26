from s2cs.agent.judge import Verdict
from s2cs.trainer.reward import s2cs_reward


def test_extract_answer():
    assert s2cs_reward._extract_answer("foo <answer>42</answer> bar") == "42"
    # No submit_answer tool call and no <answer> tag => never submitted => None
    # (the value4k Bug-1 fix: do NOT fall back to the whole transcript).
    assert s2cs_reward._extract_answer("no tags here") is None
    tool_call = '<tool_call>{"name": "submit_answer", "arguments": {"answer": "blue"}}</tool_call>'
    assert s2cs_reward._extract_answer(tool_call) == "blue"
    # GLM-style XML tool-call dialect (not hermes JSON) — must parse too.
    glm = "<tool_call>submit_answer\n<arg_key>answer</arg_key>\n<arg_value>green</arg_value>\n</tool_call>"
    assert s2cs_reward._extract_answer(glm) == "green"
    # a GLM trajectory that searches but never submits stays no_answer (no prose fallback).
    no_sub = "<think>search</think>\n<tool_call>search_papers\n<arg_key>query</arg_key>\n<arg_value>x</arg_value>\n</tool_call>"
    assert s2cs_reward._extract_answer(no_sub) is None


def test_qa_routes_to_judge(monkeypatch):
    async def fake_judge(question, gold, prediction, *, client, model):
        return Verdict(verdict="Correct" if prediction == gold else "Incorrect", reasoning="")

    monkeypatch.setattr(s2cs_reward, "judge", fake_judge)
    monkeypatch.setattr(s2cs_reward, "_judge_client", lambda: ("client", "model"))

    out = s2cs_reward.compute_score(
        data_source="s2cs_qa",
        solution_str="<answer>blue</answer>",
        ground_truth="blue",
        extra_info={"answer_type": "qa", "question": "color?"},
    )
    assert out["score"] == 1.0
    assert out["method"] == "judge"

    miss = s2cs_reward.compute_score(
        data_source="s2cs_qa",
        solution_str="<answer>red</answer>",
        ground_truth="blue",
        extra_info={"answer_type": "qa", "question": "color?"},
    )
    assert miss["score"] == 0.0


def test_paper_set_routes_to_pfb():
    out = s2cs_reward.compute_score(
        data_source="s2cs_paper_set",
        solution_str='<answer>{"paper_ids":[1,2,3]}</answer>',
        ground_truth="",
        extra_info={"answer_type": "paper_set", "relevance": {"1": 3, "2": 3, "3": 1}, "est_total_relevant": 2},
    )
    assert 0.0 <= out["score"] <= 1.0
    assert out["score"] > 0.0
    assert out["method"] == "paper_set"
    assert out["parse"] == "json"
    assert "recall_at_est" in out and "rank" in out and "n_pred" in out


def test_paper_set_no_answer_scores_zero_with_parse_flag():
    out = s2cs_reward.compute_score(
        data_source="s2cs_paper_set",
        solution_str="I could not find any papers.",
        ground_truth="",
        extra_info={"answer_type": "paper_set", "relevance": {"1": 3}, "est_total_relevant": 1},
    )
    assert out["score"] == 0.0
    assert out["parse"] == "no_answer"


def test_paper_set_relevance_accepts_json_string():
    out = s2cs_reward.compute_score(
        data_source="s2cs_paper_set",
        solution_str='<answer>{"paper_ids":[1]}</answer>',
        ground_truth="",
        extra_info={"answer_type": "paper_set", "relevance": '{"1": 3}', "est_total_relevant": 1},
    )
    assert out["score"] > 0.0


def test_compute_score_batch_mixed(monkeypatch):
    async def fake_judge(question, gold, prediction, *, client, model):
        return Verdict(verdict="Correct" if prediction == gold else "Incorrect", reasoning="")

    monkeypatch.setattr(s2cs_reward, "judge", fake_judge)
    monkeypatch.setattr(s2cs_reward, "_judge_client", lambda: ("client", "model"))

    results = s2cs_reward.compute_score_batch(
        data_sources=["s2cs_qa", "s2cs_paper_set"],
        solution_strs=["<answer>blue</answer>", '<answer>{"paper_ids":[1,2]}</answer>'],
        ground_truths=["blue", ""],
        extra_infos=[
            {"answer_type": "qa", "question": "color?"},
            {"answer_type": "paper_set", "relevance": {"1": 3, "2": 3}, "est_total_relevant": 2},
        ],
    )
    assert len(results) == 2
    assert results[0]["score"] == 1.0 and results[0]["method"] == "judge"
    assert results[1]["score"] > 0.0 and results[1]["method"] == "paper_set"
    # verl's batch reward manager arrays every result key to batch length; a mixed
    # qa+paper_set batch MUST share one key set or DataProto.chunk() asserts
    # ("key parse length 16 is not equal to batch size 32"). Regression guard.
    assert set(results[0]) == set(results[1]) == {"score", "method", "parse", "recall_at_est", "rank", "n_pred"}
