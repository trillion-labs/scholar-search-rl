import dataclasses
import logging
from typing import Literal

import openai

from s2cs.agent.llm import chat_json

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Verdict:
    verdict: Literal["Correct", "Incorrect"]
    reasoning: str


JUDGE_PROMPT = """You are an evaluation assistant. Determine if the predicted answer is semantically equivalent to the labeled answer.

Question: {question}
Labeled Answer: {gold}
Predicted Answer: {prediction}

Return a JSON object: {{"reasoning": "...", "judgment": "Correct"}} or {{"reasoning": "...", "judgment": "Incorrect"}}.
Output only the JSON object, no prose, no markdown.
"""


async def judge(
    question: str,
    gold: str,
    prediction: str,
    *,
    client: openai.AsyncOpenAI,
    model: str,
) -> Verdict:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    try:
        payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=0.0)
    except openai.APIError as exc:
        # Transient HTTP errors (429/5xx/timeout) are retried by the client's
        # max_retries; if they still exhaust, don't crash the caller's rollout —
        # fall back to Incorrect, consistent with the JSON-failure path below.
        log.warning("judge request failed after retries: %s; defaulting to Incorrect", exc)
        return Verdict(verdict="Incorrect", reasoning=f"judge request error: {exc}")
    if payload is None:
        log.warning("judge failed to produce JSON; defaulting to Incorrect")
        return Verdict(verdict="Incorrect", reasoning="judge JSON parse failure")
    raw_v = payload.get("judgment", "")
    v: Literal["Correct", "Incorrect"] = "Correct" if str(raw_v).strip().lower() == "correct" else "Incorrect"
    return Verdict(verdict=v, reasoning=str(payload.get("reasoning", "")))
