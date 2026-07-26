import pytest

pytest.importorskip("astabench", reason="eval group not installed (uv sync --group eval)")


def test_use_openrouter_grader_reassigns_both_paper_finder_models(monkeypatch):
    import astabench.evals.paper_finder.paper_finder_utils as pf_utils
    import astabench.evals.paper_finder.relevance as relevance

    from s2cs.eval.astabench import grader

    sentinel = object()
    monkeypatch.setattr(grader, "get_model", lambda m: sentinel)
    grader.use_openrouter_grader("openrouter/openai/gpt-5.4-mini")
    assert relevance.grader_model is sentinel
    assert pf_utils.parse_result_model is sentinel


def test_default_model_is_openrouter_placeholder(monkeypatch):
    from s2cs.eval.astabench import grader

    captured = []
    monkeypatch.setattr(grader, "get_model", lambda m: captured.append(m))
    grader.use_openrouter_grader()
    assert captured == ["openrouter/openai/gpt-5.4-mini"]
