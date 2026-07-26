from s2cs.agent.trajectory import Trajectory
from s2cs.eval import paper_search_qa as psqa


def _traj(answer):
    return Trajectory(
        query="q", tool_set=[], turns=[], answer=answer,
        terminated_reason="submit_answer", prompt_tokens=0, completion_tokens=0,
    )


def test_normalize_answer_matches_squad_rules():
    assert psqa.normalize_answer("The  Quick, Brown!") == "quick brown"
    assert psqa.normalize_answer("A Gene") == "gene"


def test_em_check_matches_any_golden_after_normalization():
    assert psqa.em_check("p53", ["TP53", "p53"]) == 1
    assert psqa.em_check("the BRCA1 gene", ["BRCA1 gene"]) == 1  # article + case normalized
    assert psqa.em_check("wrong", ["right"]) == 0


def test_extract_answer_prefers_last_answer_tag_else_raw():
    assert psqa.extract_answer("<answer>first</answer> ... <answer>p53</answer>") == "p53"
    assert psqa.extract_answer("just the answer") == "just the answer"
    assert psqa.extract_answer(None) is None


def test_score_one_and_aggregate():
    preds = [
        (_traj("p53"), ["TP53", "p53"]),     # EM hit
        (_traj("nonsense"), ["insulin"]),    # miss
        (_traj(None), ["x"]),                # no answer → 0
    ]
    assert psqa.score_one(preds[0][0], preds[0][1]) == 1
    assert psqa.score(preds)["em_pass@1"] == 1 / 3


def test_score_empty_is_empty():
    assert psqa.score([]) == {}
