import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class AnswerSubmission:
    answer: str


def make_submit_answer() -> Callable[[str], AnswerSubmission]:
    def submit_answer(answer: str) -> AnswerSubmission:
        """Submit your final answer. Calling this terminates the session.

        Submit only after you have enough evidence; you cannot search further
        after submitting.
        """
        return AnswerSubmission(answer=answer)
    return submit_answer
