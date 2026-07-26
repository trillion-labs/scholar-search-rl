"""SimpleQA transfer eval (OpenAI, arXiv:2411.04368).

Short-factoid QA whose answers live on the general web, NOT in a scholarly
corpus. We run our scholar-trained agent over a **native web-search surface**
(Serper Google API + a URL fetcher) and grade with SimpleQA's own LLM judge.
This measures whether the search behaviour the policy learned on s2cs scholar
tools transfers to an unseen, general-web tool surface — see `docs/evaluation.md`
(tier-L/R transfer) and the LitQA2 multi-surface methodology.

Design choices (transfer-faithfulness):
- **Native web tools only** (`web_search`, `fetch_url`): the agent never trained
  on these names/shapes, so this is the honest "does it generalise to unseen
  tools" signal (the analogue of the AstaBench `asta` native surface). No s2cs
  tool-name adapter here.
- **Web-framed system prompt** (`SIMPLEQA_SYSTEM_PROMPT`): the trained
  `agent.policy.SYSTEM_PROMPT` frames the domain as "a scientific paper corpus",
  which actively mismatches SimpleQA's general-web questions and would confound
  the measurement. We keep the same tool-calling-loop *structure/behaviour*
  instructions but reframe the domain to the web, isolating the search
  *capability* from prompt-domain mismatch. Override via `run(system_prompt=...)`
  to measure under the as-trained scholar prompt instead.
- **Grader** ported from OpenAI simple-evals: classify each answer as CORRECT /
  INCORRECT / NOT_ATTEMPTED. Metrics are SimpleQA's own (correct,
  correct_given_attempted, F-score), so the number is comparable to that table.
"""

import asyncio
import csv
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

import openai

from s2cs.agent.trajectory import Trajectory
from s2cs.env.tools.submit_answer import make_submit_answer
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult

log = logging.getLogger(__name__)

# OpenAI's published SimpleQA test set (4,326 questions; columns metadata/problem/answer).
SIMPLEQA_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
_DEFAULT_CACHE = Path("data/eval/simpleqa_test.csv")

SIMPLEQA_SYSTEM_PROMPT = """You are a research agent answering questions using web search.

You operate in a tool-calling loop: on every turn you MUST respond by calling
exactly one of the available tools — never answer in plain text. The only way to
give your final answer is to call submit_answer.

Workflow:
- Use the web_search and fetch_url tools to gather evidence from the web.
- Think briefly about each step before acting.
- When confident, call submit_answer with your final answer.
- Submitting ends the session. Do not submit until you have enough evidence.
"""

TASK_TEMPLATE = """Answer the following question with a single short factual answer (a name, date, number, or short phrase — not a sentence).

Question: {question}

Use the `web_search` tool to find relevant pages (search several times, refining the query, if needed) and `fetch_url` to read a page in full when a snippet is not enough. Once you have found the answer, call `submit_answer` with just the answer — as concise as possible, no explanation."""


# ── dataset ───────────────────────────────────────────────────────────────


def _ensure_csv(source: str) -> Path:
    """Resolve the SimpleQA CSV to a local path, downloading the URL once."""
    if not source.startswith(("http://", "https://")):
        return Path(source)
    cache = _DEFAULT_CACHE
    if not cache.exists():
        import requests

        cache.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading SimpleQA test set -> %s", cache)
        resp = requests.get(source, timeout=120)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return cache


def load_questions(
    csv_source: str = SIMPLEQA_CSV_URL,
    *,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return [(sample_id, problem, answer)]. `sample` takes a seeded random
    subset (for cost); `limit` is a hard first-N cap (for smokes)."""
    path = _ensure_csv(csv_source)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("problem") and r.get("answer")]
    indexed = list(enumerate(rows))
    if sample is not None and sample < len(indexed):
        indexed = random.Random(seed).sample(indexed, sample)
    if limit is not None:
        indexed = indexed[:limit]
    out = [(f"simpleqa_{i}", r["problem"], r["answer"]) for i, r in indexed]
    log.info("loaded %d SimpleQA questions (sample=%s limit=%s)", len(out), sample, limit)
    return out


# ── native web tools (Serper) ──────────────────────────────────────────────


def make_web_search(api_key: str, default_num: int = 10) -> Callable:
    import requests

    def web_search(query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Search the web (Google) and return the top results.

        Returns a ranked list of results, each {rank, title, url, snippet}. A
        Google "answer box" (a direct extracted answer), when present, is
        included as the first entry. Read snippets to find the answer; use
        fetch_url on a result's url when a snippet is not enough.
        """
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        out: list[dict[str, Any]] = []
        box = data.get("answerBox")
        if box:
            answer = box.get("answer") or box.get("snippet") or box.get("title")
            if answer:
                out.append({"rank": 0, "title": "Google answer box",
                            "url": box.get("link", ""), "snippet": str(answer)})
        for item in data.get("organic", [])[:num_results]:
            out.append({
                "rank": item.get("position", len(out)),
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return out

    return web_search


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_ANYTAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def make_fetch_url(max_chars: int = 6000) -> Callable:
    import html as _html

    import requests

    def fetch_url(url: str) -> dict[str, Any]:
        """Fetch a web page and return its readable text.

        Returns {url, text} with the page's visible text (HTML stripped),
        truncated. Use after web_search when a result snippet does not contain
        enough to answer.
        """
        resp = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0 (s2cs-eval)"}, timeout=30
        )
        resp.raise_for_status()
        text = _TAG_RE.sub(" ", resp.text)
        text = _ANYTAG_RE.sub(" ", text)
        text = _WS_RE.sub(" ", _html.unescape(text)).strip()
        return {"url": url, "text": text[:max_chars]}

    return fetch_url


def build_tools(*, serper_key: str, web_num_results: int = 10) -> dict[str, Callable]:
    return {
        "web_search": make_web_search(serper_key, web_num_results),
        "fetch_url": make_fetch_url(),
        "submit_answer": make_submit_answer(),
    }


# ── grader (OpenAI simple-evals) ────────────────────────────────────────────

GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

_GRADE_LETTER = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def _grader_client(grader_model: str) -> tuple[openai.AsyncOpenAI, str]:
    """Resolve the inspect-style 'openrouter/<model>' string to an
    OpenAI-compatible client + model id. OPENROUTER_API_KEY for the openrouter
    prefix, else OPENAI_API_KEY."""
    if grader_model.startswith("openrouter/"):
        return (
            openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
                max_retries=4,
            ),
            grader_model[len("openrouter/"):],
        )
    return openai.AsyncOpenAI(max_retries=4), grader_model


def _parse_grade(text: str | None) -> str:
    """First A/B/C in the grader output -> label; default NOT_ATTEMPTED (matches
    OpenAI simple-evals' fallback)."""
    m = re.search(r"(A|B|C)", text or "")
    return _GRADE_LETTER.get(m.group(0) if m else "C", "not_attempted")


async def _grade_one(client: openai.AsyncOpenAI, model: str, question: str,
                     target: str, predicted: str) -> str:
    prompt = GRADER_TEMPLATE.format(question=question, target=target,
                                    predicted_answer=predicted or "")
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
    )
    return _parse_grade(resp.choices[0].message.content)


async def grade_all(predictions: list[tuple[str, str, str]], *, grader_model: str,
                    concurrency: int = 16) -> list[str]:
    """Grade [(question, target, predicted)] -> per-item label."""
    client, model = _grader_client(grader_model)
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str, t: str, p: str) -> str:
        async with sem:
            try:
                return await _grade_one(client, model, q, t, p)
            except Exception as exc:
                log.warning("grader call failed (treating as not_attempted): %s", exc)
                return "not_attempted"

    return await asyncio.gather(*(one(q, t, p) for q, t, p in predictions))


def score(labels: list[str]) -> dict[str, float]:
    """SimpleQA's own metrics from per-item CORRECT/INCORRECT/NOT_ATTEMPTED."""
    n = len(labels)
    if n == 0:
        return {}
    correct = labels.count("correct") / n
    incorrect = labels.count("incorrect") / n
    not_attempted = labels.count("not_attempted") / n
    attempted = correct + incorrect
    cga = correct / attempted if attempted else 0.0
    f_score = (2 * correct * cga / (correct + cga)) if (correct + cga) else 0.0
    return {
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "attempted": attempted,
        "correct_given_attempted": cga,
        "f_score": f_score,
    }


# ── driver ──────────────────────────────────────────────────────────────────


def run(
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    csv_source: str = SIMPLEQA_CSV_URL,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
    grader_model: str,
    web_num_results: int = 10,
    max_turns: int = 20,
    temperature: float = 0.7,
    concurrency: int = 16,
    trajectory_dir: str | None = None,
    system_prompt: str = SIMPLEQA_SYSTEM_PROMPT,
    chat_format: str = "qwen",
) -> BenchResult:
    serper_key = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
    if not serper_key:
        raise RuntimeError("set SERPER_API_KEY (or SERPER_KEY_ID) for the simpleqa web surface")

    questions = load_questions(csv_source, sample=sample, seed=seed, limit=limit)
    tools = build_tools(serper_key=serper_key, web_num_results=web_num_results)

    prompts = [TASK_TEMPLATE.format(question=q) for _, q, _ in questions]
    sample_ids = [sid for sid, _, _ in questions]

    trajs: list[Trajectory] = asyncio.run(run_rollouts(
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
        system_prompt=system_prompt,
        chat_format=chat_format,
    ))

    grade_inputs = [(q, a, (t.answer or "")) for (_, q, a), t in zip(questions, trajs)]
    labels = asyncio.run(grade_all(grade_inputs, grader_model=grader_model, concurrency=concurrency))
    return BenchResult(metrics=score(labels), n=len(questions))
