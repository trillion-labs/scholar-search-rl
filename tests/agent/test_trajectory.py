import dataclasses
import json

from s2cs.agent.trajectory import Trajectory, Turn


@dataclasses.dataclass
class FakeObservation:
    corpus_id: int
    title: str


def test_to_jsonl_roundtrip_plain_dict():
    traj = Trajectory(
        query="q",
        tool_set=["paper_search", "submit_answer"],
        turns=[
            Turn(thought="t1", action={"name": "paper_search", "arguments": {"query": "x"}}, observation={"hits": 3}),
            Turn(thought="t2", action={"name": "submit_answer", "arguments": {"answer": "42"}}, observation=None),
        ],
        answer="42",
        terminated_reason="submit_answer",
        prompt_tokens=100,
        completion_tokens=50,
    )
    line = traj.to_jsonl()
    data = json.loads(line)
    assert data["query"] == "q"
    assert data["answer"] == "42"
    assert data["terminated_reason"] == "submit_answer"
    assert len(data["turns"]) == 2
    assert data["turns"][0]["action"]["name"] == "paper_search"


def test_to_jsonl_serializes_dataclass_observation():
    traj = Trajectory(
        query="q", tool_set=["paper_search"],
        turns=[Turn(thought="t", action={"name": "paper_search", "arguments": {}}, observation=[FakeObservation(corpus_id=1, title="x")])],
        answer=None, terminated_reason="max_turns", prompt_tokens=0, completion_tokens=0,
    )
    line = traj.to_jsonl()
    data = json.loads(line)
    obs = data["turns"][0]["observation"]
    assert obs == [{"corpus_id": 1, "title": "x"}]
