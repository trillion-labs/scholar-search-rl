"""GAIA transfer eval (Mialon et al., arXiv:2311.12983).

GAIA is a general-assistant benchmark whose questions need multi-step reasoning
and tool use. The official set is gated on HF and many questions attach a file
(image/spreadsheet/audio) requiring multimodal + file-reading tools we don't
have. We therefore evaluate over the **file-less, web-answerable subset** on the
same native web surface as simpleqa/browsecomp (Serper `web_search` +
`fetch_url`) — an honest "does web-search capability transfer" lens, not a full
GAIA run. Default dataset is the local curated subset (no HF gating).

Scoring is GAIA's own **deterministic exact/quasi-exact match** (`question_scorer`,
ported from the GAIA repo) — no LLM judge. Metric: accuracy + answered rate.
"""

import ast
import asyncio
import json
import logging
import os
import random
import re
import string
from pathlib import Path

from s2cs.agent.trajectory import Trajectory
from s2cs.eval.local_runner import run_rollouts
from s2cs.eval.result import BenchResult
from s2cs.eval.simpleqa import build_tools

log = logging.getLogger(__name__)

# Local curated file-less GAIA subset (no HF gating, no attachments). Override via
# --gaia-path or the GAIA_PATH env.
_DEFAULT_GAIA_PATH = "data/gaia.jsonl"

# GAIA's answer-format instruction, adapted to our submit_answer tool: the exact
# final answer goes in submit_answer (no "FINAL ANSWER:" prefix needed).
GAIA_SYSTEM_PROMPT = """You are a general AI assistant answering hard questions using web search.

You operate in a tool-calling loop: on every turn you MUST respond by calling
exactly one of the available tools — never answer in plain text. The only way to
give your final answer is to call submit_answer.

These questions need multi-step research: the answer is rarely on the first page.
Search persistently, follow leads across pages, and use fetch_url to read pages
in full. Cross-check before you commit.

Your submitted answer must be EXACT and minimal: a number (no commas/units unless
asked), OR as few words as possible, OR a comma-separated list — matching what the
question asks for. Do not add explanation or a sentence; submit only the answer.
"""

TASK_TEMPLATE = """Answer the following question with the exact, minimal final answer (a number, as few words as possible, or a comma-separated list).

Question: {question}

Use `web_search` repeatedly with different queries and `fetch_url` to read pages in full, cross-checking sources, until you find and verify the answer. Then call `submit_answer` with just the exact answer (no units unless required, no explanation)."""


# ── dataset (local curated subset; answers stored as a stringified list) ─────


def _canon_gold(row: dict) -> str:
    """The curated set stores the gold as a stringified list, e.g. "['34689']".
    Recover GAIA's canonical answer string: a lone element verbatim, or a
    comma-joined list (matching question_scorer's list handling)."""
    raw = row.get("correct_answer") or row.get("answer") or ""
    try:
        val = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return str(raw).strip()
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x).strip() for x in val)
    return str(val).strip()


def load_questions(
    path: str = _DEFAULT_GAIA_PATH,
    *,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return [(sample_id, question, gold_answer)]. `sample` takes a seeded random
    subset; `limit` is a hard first-N cap (for smokes)."""
    with open(Path(path), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if (r.get("question") or r.get("query")) and (r.get("correct_answer") or r.get("answer"))]
    indexed = list(enumerate(rows))
    if sample is not None and sample < len(indexed):
        indexed = random.Random(seed).sample(indexed, sample)
    if limit is not None:
        indexed = indexed[:limit]
    out = [(f"gaia_{i}", (r.get("question") or r.get("query")), _canon_gold(r)) for i, r in indexed]
    log.info("loaded %d GAIA questions (sample=%s limit=%s)", len(out), sample, limit)
    return out


# ── scorer (GAIA's own deterministic exact/quasi-exact match) ───────────────


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_number_str(number_str: str) -> float:
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _normalize_str(s: str, *, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Ported from the GAIA repo: numeric golds compared as floats, comma/semicolon
    lists element-wise, everything else as punctuation/space-stripped strings.

    The agent may submit a non-string answer (the GLM tool-arg parser JSON-coerces
    "34689" -> int 34689), so coerce both sides to str before any string op."""
    model_answer = "" if model_answer is None else str(model_answer)
    ground_truth = "" if ground_truth is None else str(ground_truth)
    if _is_float(ground_truth):
        return _normalize_number_str(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in [",", ";"]):
        gt_elems = re.split(r"[,;]", ground_truth)
        ma_elems = re.split(r"[,;]", model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        ok = []
        for ma, gt in zip(ma_elems, gt_elems):
            if _is_float(gt.strip()):
                ok.append(_normalize_number_str(ma) == float(gt.strip()))
            else:
                ok.append(_normalize_str(ma, remove_punct=False) == _normalize_str(gt, remove_punct=False))
        return all(ok)
    return _normalize_str(model_answer) == _normalize_str(ground_truth)


def score(labels: list[bool]) -> dict[str, float]:
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
    path: str = _DEFAULT_GAIA_PATH,
    sample: int | None = None,
    seed: int = 0,
    limit: int | None = None,
    web_num_results: int = 10,
    max_turns: int = 40,
    temperature: float = 0.7,
    concurrency: int = 16,
    trajectory_dir: str | None = None,
    system_prompt: str = GAIA_SYSTEM_PROMPT,
    chat_format: str = "qwen",
) -> BenchResult:
    serper_key = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
    if not serper_key:
        raise RuntimeError("set SERPER_API_KEY (or SERPER_KEY_ID) for the gaia web surface")

    path = os.environ.get("GAIA_PATH", path)
    questions = load_questions(path, sample=sample, seed=seed, limit=limit)
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

    preds = ["" if t.answer is None else str(t.answer) for t in trajs]
    labels = [question_scorer(p, gold) for (_, _, gold), p in zip(questions, preds)]
    metrics = score(labels)
    metrics["answered"] = sum(1 for p in preds if p.strip()) / len(preds) if preds else 0.0
    return BenchResult(metrics=metrics, n=len(questions))
