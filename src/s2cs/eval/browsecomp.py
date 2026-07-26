"""BrowseComp transfer eval (OpenAI, arXiv:2504.12516).

Hard multi-hop web-browsing QA. Unlike SimpleQA — where the answer is a single
fact that Google's answer box usually surfaces directly, so a web-augmented
agent saturates near a ceiling — BrowseComp questions are constructed so the
answer is *hard to find*: it takes persistent, many-step searching and
cross-checking. Same native web surface as `simpleqa.py` (Serper `web_search` +
`fetch_url`), so this is the harder-headroom companion lens for the same
web-search-capability transfer question (see `docs/evaluation.md`).

The published test set ships XOR-encrypted (per-row `canary` password); we
decrypt locally with the OpenAI simple-evals scheme. The grader is BrowseComp's
own extract-then-judge prompt; the metric is accuracy (correct / n), plus the
fraction that submitted any answer (`answered`).
"""

import asyncio
import base64
import csv
import hashlib
import logging
import os
import random
import re
from pathlib import Path

import openai

from s2cs.agent.trajectory import Trajectory
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult
from s2cs.eval.simpleqa import _grader_client, build_tools

log = logging.getLogger(__name__)

# OpenAI's published BrowseComp test set (1,266 questions; columns problem/answer/canary,
# problem+answer base64-XOR-encrypted under the row's canary).
BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
_DEFAULT_CACHE = Path("data/eval/browsecomp_test.csv")

# Same tool-loop/behaviour framing as simpleqa, but the prompt pushes *persistent*
# multi-step browsing — BrowseComp answers are not on the first page.
BROWSECOMP_SYSTEM_PROMPT = """You are a research agent answering hard questions using web search.

You operate in a tool-calling loop: on every turn you MUST respond by calling
exactly one of the available tools — never answer in plain text. The only way to
give your final answer is to call submit_answer.

These questions are deliberately hard: the answer is rarely on the first page of
results. Search persistently — issue many different queries, follow leads across
pages, and use fetch_url to read promising pages in full. Cross-check before you
commit.

Workflow:
- Use web_search and fetch_url to gather and verify evidence.
- Reformulate your query when results are unhelpful; try several angles.
- When you are confident, call submit_answer with the exact short answer.
- Submitting ends the session. Do not submit until the evidence supports it.
"""

TASK_TEMPLATE = """Answer the following question with a single short factual answer (a name, date, number, or short phrase — not a sentence).

Question: {question}

This is a hard question: the answer is not immediately searchable. Use `web_search` repeatedly with different queries and `fetch_url` to read pages in full, cross-checking sources, until you find and verify the answer. Once confident, call `submit_answer` with just the answer."""


# ── dataset (encrypted; decrypt with the per-row canary) ────────────────────


def _derive_key(password: str, length: int) -> bytes:
    """Repeat the SHA-256 of the password to `length` bytes (OpenAI scheme)."""
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """XOR-decrypt a base64 ciphertext with the canary-derived keystream."""
    enc = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(enc))
    return bytes(a ^ b for a, b in zip(enc, key)).decode()


def _ensure_csv(source: str) -> Path:
    """Resolve the BrowseComp CSV to a local path, downloading the URL once."""
    if not source.startswith(("http://", "https://")):
        return Path(source)
    cache = _DEFAULT_CACHE
    if not cache.exists():
        import requests

        cache.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading BrowseComp test set -> %s", cache)
        resp = requests.get(source, timeout=120)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return cache


def load_questions(
    csv_source: str = BROWSECOMP_CSV_URL,
    *,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return [(sample_id, problem, answer)], decrypting each row with its canary.
    `sample` takes a seeded random subset (for cost); `limit` is a hard first-N
    cap (for smokes)."""
    path = _ensure_csv(csv_source)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("problem") and r.get("answer") and r.get("canary")]
    indexed = list(enumerate(rows))
    if sample is not None and sample < len(indexed):
        indexed = random.Random(seed).sample(indexed, sample)
    if limit is not None:
        indexed = indexed[:limit]
    out = [
        (f"browsecomp_{i}", _decrypt(r["problem"], r["canary"]), _decrypt(r["answer"], r["canary"]))
        for i, r in indexed
    ]
    log.info("loaded %d BrowseComp questions (sample=%s limit=%s)", len(out), sample, limit)
    return out


# ── grader (OpenAI simple-evals BrowseComp: extract-then-judge) ─────────────

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.""".strip()

_CORRECT_RE = re.compile(r"correct\s*:\s*(yes|no)", re.IGNORECASE)


async def _grade_one(client: openai.AsyncOpenAI, model: str, question: str,
                     correct_answer: str, response: str) -> bool:
    prompt = GRADER_TEMPLATE.format(question=question, correct_answer=correct_answer,
                                    response=response or "")
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
    )
    m = _CORRECT_RE.search(resp.choices[0].message.content or "")
    return bool(m and m.group(1).lower() == "yes")


async def grade_all(predictions: list[tuple[str, str, str]], *, grader_model: str,
                    concurrency: int = 16) -> list[bool]:
    """Grade [(question, correct_answer, response)] -> per-item correct bool."""
    client, model = _grader_client(grader_model)
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str, a: str, p: str) -> bool:
        async with sem:
            try:
                return await _grade_one(client, model, q, a, p)
            except Exception as exc:
                log.warning("grader call failed (treating as incorrect): %s", exc)
                return False

    return await asyncio.gather(*(one(q, a, p) for q, a, p in predictions))


def score(labels: list[bool]) -> dict[str, float]:
    """BrowseComp metric: accuracy = correct / n."""
    n = len(labels)
    if n == 0:
        return {}
    return {"accuracy": sum(1 for x in labels if x) / n}


# ── driver ──────────────────────────────────────────────────────────────────


def run(
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    csv_source: str = BROWSECOMP_CSV_URL,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
    grader_model: str,
    web_num_results: int = 10,
    max_turns: int = 40,
    temperature: float = 0.7,
    concurrency: int = 16,
    trajectory_dir: str | None = None,
    system_prompt: str = BROWSECOMP_SYSTEM_PROMPT,
    chat_format: str = "qwen",
) -> BenchResult:
    serper_key = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
    if not serper_key:
        raise RuntimeError("set SERPER_API_KEY (or SERPER_KEY_ID) for the browsecomp web surface")

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
    metrics = score(labels)
    metrics["answered"] = sum(1 for t in trajs if (t.answer or "").strip()) / len(trajs) if trajs else 0.0
    return BenchResult(metrics=metrics, n=len(questions))
