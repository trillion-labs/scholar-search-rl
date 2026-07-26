"""PaperSearchQA adapter (EACL 2026, arXiv:2601.18207; jmhb0/PaperSearchQA).

Biomedical single-entity factoid QA over a fixed ~16.2M PubMed corpus. The
closest prior to this project (a sim-RL scholar-search agent), so it is our
head-to-head comparison anchor. We run our agent over the corpus served by the
benchmark's own Search-R1 `/retrieve` HTTP server (wrapped as a `search_pubmed`
tool) and score exact-match Pass@1.

Faithfulness notes:
- `normalize_answer` / `em_check` are ported verbatim from the repo's
  `search-r1/verl/utils/reward_score/qa_em.py` (SQuAD-style normalization,
  match against the `golden_answers` list).
- Their `extract_solution` requires the answer inside `<answer>…</answer>` with
  ≥2 matches — that count is an artifact of Search-R1's generation protocol (the
  prompt template itself embeds one `<answer>` tag). Our agent answers via
  `submit_answer`, so we extract the last `<answer>` block if present, else take
  the submitted text directly. The EM contract is unchanged; only the extraction
  is adapted to our submission shape.
- The corpus is biomedical, disjoint from our CS training substrate — the
  resulting number is a domain-transfer measurement, not in-domain (see
  `docs/evaluation.md` §6).
"""

import dataclasses
import logging
import re
import string
from typing import Any, Callable

from s2cs.agent.trajectory import Trajectory
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult
from s2cs.env.tools.submit_answer import make_submit_answer

log = logging.getLogger(__name__)

PAPER_SEARCH_QA_HF_ID = "jmhb/PaperSearchQA"

TASK_TEMPLATE = """Answer the following biomedical question with a single short factual answer (an entity name, term, or short phrase — not a sentence).

Question: {question}

Use the `search_pubmed` tool to retrieve relevant PubMed abstracts (you may search several times, refining the query). Once you have found the answer, call `submit_answer` with just the answer — as concise as possible, no explanation."""


@dataclasses.dataclass(frozen=True)
class Question:
    sample_id: str
    text: str
    golden_answers: list[str]


def normalize_answer(s: str) -> str:
    """Ported verbatim from PaperSearchQA search-r1 .../reward_score/qa_em.py."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: list[str] | str) -> int:
    """Ported verbatim from PaperSearchQA search-r1 .../reward_score/qa_em.py."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str | None) -> str | None:
    """Recover the answer from our agent's `submit_answer` payload.

    Faithful to the EM core but adapted to our submission: the last `<answer>`
    block if the model emitted tags, otherwise the submitted text verbatim.
    """
    if text is None:
        return None
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def load_questions(split: str = "test", limit: int | None = None) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset(PAPER_SEARCH_QA_HF_ID, split=split)
    out: list[Question] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        golden = row.get("golden_answers") or []
        if not golden and row.get("answer"):
            golden = [row["answer"]]
        out.append(Question(
            sample_id=f"{split}_{i}",
            text=row["question"],
            golden_answers=[str(g) for g in golden],
        ))
    log.info("loaded %d PaperSearchQA questions (split=%s)", len(out), split)
    return out


def score_one(traj: Trajectory, golden_answers: list[str]) -> int:
    answer = extract_answer(traj.answer)
    if answer is None:
        return 0
    return em_check(answer, golden_answers)


def score(predictions: list[tuple[Trajectory, list[int]]]) -> dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {}
    correct = sum(score_one(t, g) for t, g in predictions)
    return {"em_pass@1": correct / n}


def make_search_pubmed(retrieve_url: str, default_topk: int = 10) -> Callable:
    """Wrap the benchmark's Search-R1 `/retrieve` server as a search tool.

    POSTs {queries, topk, return_scores} and returns the retrieved passage
    contents so the agent can read them and answer the factoid.
    """
    import requests

    def search_pubmed(query: str, topk: int = default_topk) -> list[dict[str, Any]]:
        """Search the PubMed corpus and return the top passages' text.

        Returns up to `topk` passages, each {rank, text, score}, ranked by
        relevance. Read the passage text to find the answer to the question.
        """
        resp = requests.post(
            retrieve_url,
            json={"queries": [query], "topk": topk, "return_scores": True},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()["result"][0]
        out: list[dict[str, Any]] = []
        for rank, item in enumerate(result):
            doc = item.get("document", item) if isinstance(item, dict) else {}
            out.append({
                "rank": rank,
                "text": doc.get("contents", "") if isinstance(doc, dict) else str(item),
                "score": item.get("score") if isinstance(item, dict) else None,
            })
        return out

    return search_pubmed


def build_tools(*, retrieve_url: str, default_topk: int = 10) -> dict[str, Callable]:
    return {
        "search_pubmed": make_search_pubmed(retrieve_url, default_topk),
        "submit_answer": make_submit_answer(),
    }


def run(
    *,
    base_url: str,
    model: str,
    retrieve_url: str,
    api_key: str = "EMPTY",
    split: str = "test",
    retrieve_topk: int = 10,
    limit: int | None = None,
    max_turns: int = 40,
    temperature: float = 0.7,
    concurrency: int = 16,
    trajectory_dir: str | None = None,
    chat_format: str = "qwen",
) -> BenchResult:
    questions = load_questions(split=split, limit=limit)
    tools = build_tools(retrieve_url=retrieve_url, default_topk=retrieve_topk)

    prompts = [TASK_TEMPLATE.format(question=q.text) for q in questions]
    sample_ids = [q.sample_id for q in questions]

    import asyncio

    trajs = asyncio.run(run_rollouts(
        prompts,
        base_url=base_url,
        model=model,
        tools=tools,
        api_key=api_key,
        max_turns=max_turns,
        temperature=temperature,
        concurrency=concurrency,
        trajectory_dir=trajectory_dir,
        sample_ids=sample_ids,
        chat_format=chat_format,
    ))

    predictions = [(t, q.golden_answers) for t, q in zip(trajs, questions)]
    metrics = score(predictions)
    return BenchResult(metrics=metrics, n=len(questions))
