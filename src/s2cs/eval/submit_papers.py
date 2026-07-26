import json
from typing import Callable

from s2cs.env.tools.submit_answer import AnswerSubmission


def make_submit_papers() -> Callable[[list[int]], AnswerSubmission]:
    """Paper-set submission for retrieval-scored benchmarks (LitSearch).

    Reuses the existing `AnswerSubmission` terminal so `react.rollout` ends
    naturally — the ranked id list rides in `answer` as a JSON array, which the
    bench scorer parses back out. Keeping this in `eval/` avoids touching the
    agent/env modules just to add an eval-only submission shape.
    """

    def submit_papers(paper_ids: list[int]) -> AnswerSubmission:
        """Submit your final ranked list of relevant paper ids, most relevant
        first. Calling this terminates the session — submit only once you have
        gathered enough candidates; you cannot search further afterward."""
        return AnswerSubmission(answer=json.dumps(list(paper_ids)))

    return submit_papers


def parse_submitted_ids(answer: str | None) -> list[int]:
    """Recover the ranked id list from a `submit_papers` answer. Tolerates the
    model answering in prose by salvaging the first JSON array if present."""
    if not answer:
        return []
    try:
        ids = json.loads(answer)
        if isinstance(ids, list):
            return [int(x) for x in ids]
    except (ValueError, TypeError):
        pass
    start, end = answer.find("["), answer.rfind("]")
    if 0 <= start < end:
        try:
            ids = json.loads(answer[start : end + 1])
            return [int(x) for x in ids if isinstance(x, (int, str))]
        except (ValueError, TypeError):
            return []
    return []
