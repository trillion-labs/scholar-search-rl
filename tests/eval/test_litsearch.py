from s2cs.agent.trajectory import Trajectory, Turn
from s2cs.env.tools.search_papers import PaperHit
from s2cs.eval import litsearch


def _hit(cid: int) -> PaperHit:
    return PaperHit(corpus_id=cid, year=None, venue=None, citation_count=0, score=1.0)


def test_recall_and_ndcg_match_litsearch_definitions():
    assert litsearch.calculate_recall([1, 2, 9], [1, 2, 3, 4]) == 0.5
    assert litsearch.calculate_recall([9], []) == 0.0
    # ndcg: hit at rank 0 → dcg=1; idcg for 1 relevant = 1 → ndcg=1
    assert litsearch.calculate_ndcg([7], [7]) == 1.0
    # hit at rank 1 (idx 1) → dcg=1/2, idcg(1 rel)=1 → 0.5
    assert litsearch.calculate_ndcg([9, 7], [7]) == 0.5


def test_score_macro_averages_over_queries():
    preds = [
        ([1, 2, 3], [1, 2]),   # recall@5 = 1.0
        ([9, 8, 7], [1, 2]),   # recall@5 = 0.0
    ]
    m = litsearch.score(preds, k_values=(5,))
    assert m["recall@5"] == 0.5
    assert "ndcg@5" in m


def test_score_empty_predictions_is_empty():
    assert litsearch.score([], k_values=(5, 20)) == {}


def _traj(answer, turns):
    return Trajectory(
        query="q", tool_set=["search_papers", "submit_papers"], turns=turns,
        answer=answer, terminated_reason="submit_answer", prompt_tokens=0, completion_tokens=0,
    )


def test_extract_prefers_submitted_ids_then_backfills_from_search():
    turns = [
        Turn(thought="", action={"name": "search_papers", "arguments": {}},
             observation=[_hit(10), _hit(20), _hit(30)]),
    ]
    # submitted ranks 30 first; search-surfaced 10,20 backfill (30 deduped)
    ranked = litsearch.extract_ranked_ids(_traj("[30, 40]", turns))
    assert ranked == [30, 40, 10, 20]


def test_extract_falls_back_to_search_order_without_submission():
    turns = [
        Turn(thought="", action={"name": "search_papers", "arguments": {}},
             observation=[_hit(5), _hit(6)]),
        Turn(thought="", action={"name": "search_papers", "arguments": {}},
             observation=[_hit(6), _hit(7)]),
    ]
    assert litsearch.extract_ranked_ids(_traj(None, turns)) == [5, 6, 7]
