import asyncio
import dataclasses

import pytest

from s2cs.trainer.tools import s2cs_tool


@dataclasses.dataclass
class FakeHit:
    corpus_id: int
    score: float


def _fake_build_tools(**kwargs):
    def search_papers(query, limit=10, mask_paper_ids=None):
        hits = [FakeHit(1, 0.9), FakeHit(2, 0.8)]
        if mask_paper_ids:
            hits = [h for h in hits if h.corpus_id not in mask_paper_ids]
        return hits

    return {"search_papers": search_papers}


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setattr(s2cs_tool, "build_tools", _fake_build_tools)
    s2cs_tool._TOOLS_CACHE.clear()


def _schema():
    return {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_execute_dispatches_named_tool():
    tool = s2cs_tool.S2CSTool({"tool_name": "search_papers"}, _schema())
    iid, _ = asyncio.run(tool.create())
    resp, reward, metrics = asyncio.run(tool.execute(iid, {"query": "graphs"}))
    assert reward == 0.0
    assert '"corpus_id": 1' in resp.text
    assert metrics["tool_name"] == "search_papers"
    assert metrics["status"] == "ok"


def test_seed_paper_ids_become_mask():
    tool = s2cs_tool.S2CSTool({"tool_name": "search_papers"}, _schema())
    iid, _ = asyncio.run(tool.create(seed_paper_ids=[1]))
    resp, _, _ = asyncio.run(tool.execute(iid, {"query": "graphs"}))
    assert '"corpus_id": 1' not in resp.text
    assert '"corpus_id": 2' in resp.text


def test_tool_error_is_caught():
    tool = s2cs_tool.S2CSTool({"tool_name": "search_papers"}, _schema())
    iid, _ = asyncio.run(tool.create())
    resp, reward, metrics = asyncio.run(tool.execute(iid, {"bad_arg": 1}))
    assert metrics["status"] == "error"
    assert reward == 0.0
