import astabench.evals.paper_finder.paper_finder_utils as _pf_utils
import astabench.evals.paper_finder.relevance as _relevance
from inspect_ai.model import get_model

DEFAULT_GRADER_MODEL = "openrouter/openai/gpt-5.4-mini"


def use_openrouter_grader(model: str = DEFAULT_GRADER_MODEL) -> None:
    """Route PaperFindingBench's LLM-backed scoring through OpenRouter.

    astabench hardcodes two `get_model("openai/gpt-4o-2024-11-20")` module
    globals at import in the paper_finder eval:
      - `relevance.grader_model` — relevance judge for `semantic` queries.
      - `paper_finder_utils.parse_result_model` — parses the agent's output
        into ExpectedAgentOutput (every query type).
    Reassigning both sends scoring to OpenRouter (OPENROUTER_API_KEY) instead
    of OpenAI. The default is a placeholder — pass the finalized model once
    chosen.
    """
    judge = get_model(model)
    _relevance.grader_model = judge
    _pf_utils.parse_result_model = judge
