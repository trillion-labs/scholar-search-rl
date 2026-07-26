"""verl `custom_reward_function` for the s2cs scholar agent.

Routes on `extra_info["answer_type"]`: QA rows go to the binary LM judge
(`s2cs.agent.judge.judge`), `paper_set` rows to the PaperFindingBench
`adjusted_f1` scorer (`s2cs.synthesis.paper_set.score_paper_set`). The judge
endpoint defaults to an OpenAI-compatible local server.

Two entrypoints:
- `compute_score` — per-sample (sync), used in unit tests / naive manager.
- `compute_score_batch` — the verl `batch` reward manager contract; runs all
  judge calls concurrently in one event loop under a shared client + semaphore.
"""

import asyncio
import collections
import json
import logging
import os
import re

import openai

from s2cs.agent.judge import judge
from s2cs.synthesis.paper_set import parse_paper_set_submission, score_paper_set

log = logging.getLogger(__name__)

_DUMP_STEP = 0  # module-level batch counter (one reward worker process per run)

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# GLM-style tool calls aren't hermes JSON: <tool_call>{name}\n<arg_key>k</arg_key>
# <arg_value>v</arg_value>...</tool_call>. Match the name + the arg_key/arg_value body.
_GLM_TOOL_CALL = re.compile(r"<tool_call>\s*([A-Za-z_]\w*)\s*\n(.*?)</tool_call>", re.DOTALL)
_GLM_ARG = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)
_CLIENT: dict[str, tuple] = {}


def _extract_answer(text: str) -> str | None:
    """The agent's final answer = the `answer` arg of the LAST `submit_answer`
    tool call. Parse it from whichever tool-call dialect the policy emits:

    1. hermes JSON (`<tool_call>{"name": "submit_answer", ...}</tool_call>`) — Qwen.
    2. GLM-style XML (`<tool_call>submit_answer\n<arg_key>answer</arg_key>
       <arg_value>...</arg_value></tool_call>`) — the JSON regex never matches this,
       so a GLM model's submit_answer was silently scored no_answer before.
    3. `<answer>...</answer>` tag (legacy).

    Returns None when the agent never submitted (scored 0 without judging) — we do
    NOT fall back to the whole transcript / final prose (the value4k Bug-1 lesson:
    that fed the judge the entire think+tool blob and broke the no_answer signal)."""
    text = text or ""
    answer = None
    for m in _TOOL_CALL.finditer(text):
        try:
            call = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if call.get("name") != "submit_answer":
            continue
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if isinstance(args, dict) and "answer" in args:
            answer = str(args["answer"]).strip()
    if answer is not None:
        return answer
    for m in _GLM_TOOL_CALL.finditer(text):
        if m.group(1).strip() != "submit_answer":
            continue
        for key, value in _GLM_ARG.findall(m.group(2)):
            if key.strip() == "answer":
                answer = value.strip()
    if answer is not None:
        return answer
    m = _ANSWER.search(text)
    return m.group(1).strip() if m else None


def _judge_client():
    if "default" not in _CLIENT:
        base = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:30000/v1")
        api = os.environ.get("JUDGE_API_KEY", "EMPTY")
        model = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
        _CLIENT["default"] = (
            openai.AsyncOpenAI(base_url=base, api_key=api, max_retries=6, timeout=120.0),
            model,
        )
    return _CLIENT["default"]


def _normalize_relevance(relevance):
    if isinstance(relevance, str):
        relevance = json.loads(relevance)
    if isinstance(relevance, dict):
        return {int(k): int(v) for k, v in relevance.items()}
    return relevance


def _result(score, method, *, parse=None, recall_at_est=None, rank=None, n_pred=None) -> dict:
    """A per-sample reward dict with a FIXED key set across qa and paper_set rows.

    verl's `batch` reward manager appends every key of each sample's dict into a
    per-key list, then stuffs each list into `non_tensor_batch` as an array;
    `DataProto.chunk()` asserts every such array's length == batch size. So a key
    present on only SOME rows (the paper_set-only `parse`/`recall_at_est`/`rank`/
    `n_pred`) crashes a MIXED qa+paper_set batch — "key parse length 16 is not
    equal to batch size 32". Every row must therefore carry the same keys; the
    fields that don't apply to qa are None. Safe: `compute_data_metrics` only
    aggregates correct/method/num_turns, never these (stored + dumped only)."""
    return {
        "score": float(score),
        "method": method,
        "parse": parse,
        "recall_at_est": recall_at_est,
        "rank": rank,
        "n_pred": n_pred,
    }


async def _score_one(solution_str, ground_truth, extra_info, *, client, model) -> dict:
    extra_info = extra_info or {}
    pred_text = _extract_answer(solution_str)

    if extra_info.get("answer_type") == "paper_set":
        relevance = _normalize_relevance(extra_info["relevance"])
        # No LLM fallback here (no client/model): the policy MUST emit clean bench
        # JSON in submit_answer or it scores 0. Keep `parse` so we can tell a genuine
        # miss ("json" parsed, wrong papers) from a formatting failure ("empty"/
        # "no_answer", reward 0 for the wrong reason) — the value4k lesson.
        parse = "no_answer"
        ids = []
        if pred_text:
            ids, parse = await parse_paper_set_submission(pred_text)
        scored = score_paper_set(ids, relevance, est_total_relevant=int(extra_info["est_total_relevant"]))
        return _result(
            scored["reward"],
            "paper_set",
            parse=parse,
            recall_at_est=scored["recall_at_est"],
            rank=scored["rank"],
            n_pred=scored["n_pred"],
        )

    if not pred_text:
        return _result(0.0, "no_answer")
    verdict = await judge(extra_info.get("question", ""), ground_truth, pred_text, client=client, model=model)
    return _result(1.0 if verdict.verdict == "Correct" else 0.0, "judge")


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    client, model = _judge_client()
    return asyncio.run(_score_one(solution_str, ground_truth, extra_info, client=client, model=model))


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs) -> list:
    client, model = _judge_client()
    sem = asyncio.Semaphore(int(os.environ.get("JUDGE_MAX_CONCURRENCY", "16")))

    async def bounded(sol, gt, ei):
        async with sem:
            return await _score_one(sol, gt, ei, client=client, model=model)

    async def gather():
        return await asyncio.gather(*(bounded(s, g, e) for s, g, e in zip(solution_strs, ground_truths, extra_infos)))

    results = asyncio.run(gather())
    methods = collections.Counter(r["method"] for r in results)
    mean = sum(r["score"] for r in results) / max(1, len(results))
    sample = next((_extract_answer(s) for s in solution_strs), None)
    # parse failures (paper_set only): empty/no_answer => reward 0 for a formatting
    # reason, not a retrieval one. A high rate means the policy isn't emitting clean
    # bench JSON — diagnose that before concluding the agent can't find papers.
    n_parse_fail = sum(1 for r in results if r.get("parse") in ("empty", "no_answer"))
    log.warning(
        "[s2cs_reward] n=%d methods=%s mean=%.3f parse_fail=%d | sample_answer=%r",
        len(results),
        dict(methods),
        mean,
        n_parse_fail,
        (sample or "")[:160],
    )
    _maybe_dump(solution_strs, ground_truths, extra_infos, results)
    return results


def _maybe_dump(solution_strs, ground_truths, extra_infos, results) -> None:
    """Persist EVERY rollout of each training step as JSONL — one record per
    sample, full transcript + score breakdown + gold. The dump loops over the
    whole batch rather than a subsample,
    so the FULL per-query reward distribution and every trajectory are
    recoverable for analysis. A worst/best subsample silently censors the middle
    of the distribution — exactly the population we most need to study.

    S2CS_REWARD_DUMP is a DIRECTORY; one file per training step
    (`step_{n:04d}.jsonl`). Always on (the s2cs convention: never discard
    rollouts). Each line carries the per-sample `result` (score, method, parse,
    recall_at_est, rank, n_pred), the question, gold, extracted answer and the
    full `solution_str`; paper_set rows also carry est_total_relevant/relevance."""
    global _DUMP_STEP
    dump_dir = os.environ.get("S2CS_REWARD_DUMP")
    if not dump_dir:
        return
    _DUMP_STEP += 1
    try:
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"step_{_DUMP_STEP:04d}.jsonl")
        with open(path, "w") as f:
            for i, r in enumerate(results):
                ei = extra_infos[i] or {}
                rec = {
                    "step": _DUMP_STEP,
                    "idx": i,
                    "question": ei.get("question", ""),
                    "answer_type": ei.get("answer_type"),
                    "result": r,
                    "extracted_answer": _extract_answer(solution_strs[i]),
                    "gold": ground_truths[i],
                    "solution_str": solution_strs[i],
                }
                if ei.get("answer_type") == "paper_set":
                    rec["est_total_relevant"] = ei.get("est_total_relevant")
                    rec["relevance"] = ei.get("relevance")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("[s2cs_reward] dump failed: %s", e)
