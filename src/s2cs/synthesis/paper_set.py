"""Set-valued paper-finding QA: the gold answer is a SET of papers, not a fact.

A synthetic analog of AstaBench PaperFindingBench. The query describes a shared
criterion ("find all papers that ...") that identifies a small family of papers;
the agent must surface the set. See docs/query_design.md §4 (answer rung 5) and
docs/superpowers/specs/2026-06-11-paper-set-answer-type-design.md.

Two stages, both here:
- **generation** (`make_paper_set_synth`) — policy-agnostic, from one seed paper:
  emit a loose-conjunction question + machine-checkable relevance `criteria`; the
  seed is one member. No gold set yet.
- **gold-set freeze** (`build_gold_set`) — needs the live env: gather candidates by
  searching the corpus for the question, GRADE each 0-3 against the criteria, and
  FREEZE the graded relevant set. Run once at synthesis; training then scores
  (`score_paper_set`, adjusted_f1 = recall@est + nDCG) with no judge in the loop.
"""

import asyncio
import dataclasses
import json
import logging
import math
import re
from typing import Awaitable, Callable

import openai

from s2cs.agent.llm import chat_json
from s2cs.synthesis.retrievability import make_search_queries

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PaperSetQA:
    qa_id: str
    question: str
    criteria: tuple[str, ...]          # relevance conditions the freeze grader scores each candidate on
    seed_paper_id: int                 # the anchor member — graded 3 by construction
    anchor: str                        # content_conjunction | context_conjunction
    answer_type: str = "paper_set"
    # FROZEN graded relevance: (corpus_id, grade 1-3), grade-descending; empty until build_gold_set.
    # grade-0 (irrelevant) candidates are NOT stored — an unknown paper scores 0 at reward time.
    # Grades enable PaperFindingBench-semantic adjusted_f1 (recall@est + nDCG ranking).
    relevance: tuple[tuple[int, int], ...] = ()
    est_total_relevant: int = 0        # recall denominator = # grade-3 (PERFECT); grade 1-2 are ranking-only

    @property
    def gold_paper_ids(self) -> tuple[int, ...]:
        """corpus_ids judged relevant (grade >= 1)."""
        return tuple(cid for cid, _ in self.relevance)

    def to_record(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "source": "paper_set",
            "question": self.question,
            "criteria": list(self.criteria),
            "seed_paper_ids": [self.seed_paper_id],
            "anchor": self.anchor,
            "answer_type": self.answer_type,
            "relevance": [[cid, g] for cid, g in self.relevance],   # graded gold (0-3)
            "gold_paper_ids": list(self.gold_paper_ids),            # derived: grade >= 1
            "est_total_relevant": self.est_total_relevant,
            "rubric_pass": None,
            "stage": None,
            "pass_at_8": None,
        }


# Loose-conjunction anchors: tuned to pin a FAMILY of papers, not exactly one (the
# distinction from single_hop's same-named anchors). See docs/query_design.md §3/§5.1.
PAPER_SET_ANCHOR_RULES = {
    "content_conjunction": (
        "Point at a FAMILY of papers by the INTERSECTION of (a) a research area or approach AND "
        "(b) one shared, concrete property — a task, dataset, evaluation setup, or finding-type — "
        "that several papers share. Deliberately keep it broad enough that a SMALL SET of papers "
        "(roughly 5-15) satisfies both cues, NOT so specific it pins exactly one. Do NOT use any "
        "single paper's coined name, title, or authors."
    ),
    "context_conjunction": (
        "Point at a FAMILY of papers by a CONTENT cue (research area plus one shared property such as "
        "a task or dataset) AND a METADATA constraint they share — a publication YEAR range and/or "
        "VENUE. Keep it broad enough that a SMALL SET (roughly 5-15) satisfies all cues, not exactly "
        "one. Use only year/venue (NOT author, NOT citation count). Do NOT use any coined name, title, "
        "or authors."
    ),
}

VALID_ANCHORS = frozenset(PAPER_SET_ANCHOR_RULES)


GEN_PROMPT = """You write ONE set-valued literature-search task for a paper-finding agent, seeded from the paper below together with works it cites.

[SEED PAPER]
{public}

[BODY] (the seed's full text)
{body}
{meta_block}
[RELATED] (in-corpus papers the seed cites — the seed and SEVERAL of these form a topical family)
{related_block}

Write a question that asks an agent to FIND ALL PAPERS IN A LARGE CORPUS matching a shared criterion, where the seed AND several of the [RELATED] papers are members. Anchor the criterion on a theme the seed genuinely shares with a HANDFUL of the related papers above:
- {anchor_rule}

Then decompose the criterion into 2-3 independent, machine-checkable conditions ("criteria"): each a yes/no property checkable from a paper's title+abstract (e.g. "studies sarcasm detection", "evaluates on a social-media dataset"). The seed AND several related papers must satisfy ALL of them.

Breadth is critical: pick a theme broad enough that a FAMILY of papers (~5-15) qualifies. Do NOT encode properties unique to the seed alone — no specific event names, single proprietary datasets, or exact place+year combinations. If a condition would match only the seed, drop it or widen it.

Requirements:
- The question must clearly ask for a SET ("Find all papers that ...", "Which papers ...").
- Do NOT name, quote, or paraphrase the seed paper's title or coined system/method name.
- Do NOT ask for a fact or value; the answer is the set of matching papers itself.

Return ONLY a JSON object (no prose, no markdown):
{{"question": "...", "criteria": ["...", "..."]}}
If the seed yields no good set-valued task, return {{}}."""


def _format_public(title, abstract, summary) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if summary:
        parts.append(f"Summary: {summary}")
    return "\n\n".join(parts)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def make_paper_set_synth(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    anchor: str = "content_conjunction",
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> Callable[[dict], Awaitable[PaperSetQA | None]]:
    """Build a generation-only paper_set synthesizer for one anchor.

    The returned coroutine takes one paper row (`corpus_id, title, abstract,
    summary, body`, plus `year`/`venue` for context_conjunction) and its in-corpus
    `cited` papers (list of `{corpus_id, title, ...}`, e.g. from `_attach_cited`),
    and returns a `PaperSetQA` with an EMPTY gold set — call `build_gold_set` to
    freeze it. The cited papers ground the criterion in a real topical family so it
    spans more than the seed alone; a seed without `cited` yields None.
    """
    if anchor not in VALID_ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {sorted(VALID_ANCHORS)}")

    async def paper_set_synth(paper: dict) -> PaperSetQA | None:
        cid = int(paper["corpus_id"])
        title = paper.get("title")
        public = _format_public(title, paper.get("abstract"), paper.get("summary"))
        body = paper.get("body") or ""
        if max_body_chars is not None:
            body = body[:max_body_chars]
        if not public or not body:
            return None
        cited = paper.get("cited") or []
        related_block = "\n".join(f"  - {c['title']}" for c in cited if c.get("title"))
        if not related_block:
            return None
        meta_block = ""
        if anchor == "context_conjunction":
            if not paper.get("venue"):
                return None
            bits = []
            if paper.get("year"):
                bits.append(f"year={int(paper['year'])}")
            bits.append(f"venue={str(paper['venue'])!r}")
            meta_block = "\n[METADATA] (use these EXACT values for the year/venue cue)\n" + ", ".join(bits) + "\n"

        prompt = GEN_PROMPT.format(
            public=public, body=body, meta_block=meta_block, related_block=related_block,
            anchor_rule=PAPER_SET_ANCHOR_RULES[anchor],
        )
        payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=temperature)
        if not isinstance(payload, dict):
            return None
        question = str(payload.get("question", "")).strip()
        criteria = tuple(str(c).strip() for c in (payload.get("criteria") or []) if str(c).strip())
        if not question or len(criteria) < 2:
            return None
        if title and _norm(title) in _norm(question):  # no title leak (mirrors single_hop)
            return None
        return PaperSetQA(
            qa_id=f"paper_set_{cid}_{anchor}",
            question=question,
            criteria=criteria,
            seed_paper_id=cid,
            anchor=anchor,
        )

    return paper_set_synth


GRADE_PROMPT = """For EACH criterion below, decide whether the CANDIDATE paper satisfies it, judging only from its title and abstract. This mirrors how PaperFindingBench grades relevance — per criterion, then aggregated.

[CRITERIA]
{criteria}

[CANDIDATE]
Title: {title}
Abstract: {abstract}

Return ONLY a JSON object, no prose: {{"satisfied": [<true|false for criterion 1, 2, ... in order>]}}"""


async def _grade_candidate(cid: int, criteria: tuple[str, ...], paper_info: Callable,
                           *, client: openai.AsyncOpenAI, model: str) -> int:
    """Relevance grade 0-3 = number of criteria the candidate satisfies, capped at 3
    (criteria-coverage, judged per-criterion like PaperFindingBench). Fails closed to 0.
    grade 3 = satisfies all (PERFECT); 1-2 = partial (ranking signal, not a recall member)."""
    meta = await asyncio.to_thread(paper_info, cid)
    if meta is None or not getattr(meta, "abstract", None):
        return 0
    crit_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    prompt = GRADE_PROMPT.format(criteria=crit_block, title=meta.title or "", abstract=meta.abstract)
    payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=0.0)
    if not isinstance(payload, dict) or not isinstance(payload.get("satisfied"), list):
        return 0
    return min(3, sum(1 for x in payload["satisfied"] if x is True))


async def _gather_candidates(question: str, search_papers: Callable, search_snippets: Callable,
                             *, client: openai.AsyncOpenAI, query_model: str,
                             n_queries: int, k: int, pool_cap: int) -> list[int]:
    """Search the corpus for the question via several attempts; return a deduped,
    capped candidate corpus_id pool (insertion order = first-surfaced)."""
    queries = await make_search_queries(question, client=client, model=query_model, n=n_queries)
    out: list[int] = []
    seen: set[int] = set()
    for q in queries:
        try:
            if q["tool"] == "search_snippets":
                hits = await asyncio.to_thread(search_snippets, q["query"], limit=k)
                cids = [int(h.paper_corpus_id) for h in hits]
            else:
                hits = await asyncio.to_thread(
                    search_papers, q["query"], limit=k,
                    year_min=q.get("year_min"), year_max=q.get("year_max"), venue=q.get("venue"),
                )
                cids = [int(h.corpus_id) for h in hits]
        except Exception as exc:
            log.warning("candidate gather query failed (%r): %s", q.get("query"), exc)
            continue
        for cid in cids:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out[:pool_cap]


async def build_gold_set(
    qa: PaperSetQA,
    *,
    search_papers: Callable,
    search_snippets: Callable,
    paper_info: Callable,
    client: openai.AsyncOpenAI,
    model: str,
    query_model: str | None = None,
    seed_candidate_ids: list[int] | None = None,
    n_queries: int = 5,
    k: int = 15,
    pool_cap: int = 60,
    concurrency: int = 8,
    min_size: int = 3,
    max_size: int = 20,
) -> PaperSetQA | None:
    """Freeze the gold set for one generated `PaperSetQA` against the live env.

    Gathers candidates by searching the corpus for the question, unions in
    `seed_candidate_ids` (e.g. the seed's in-corpus citations — real family members
    the search might miss), GRADES each 0-3 against `qa.criteria`, and returns the QA
    with `relevance` ({id: grade} for grade >= 1; seed graded 3) and
    `est_total_relevant`. Seeded candidates are still graded, not trusted blindly.
    Returns None if the relevant count is outside [min_size, max_size] — too small
    collapses to single-paper `identity`, too large means the criterion is too loose.
    """
    query_model = query_model or model
    searched = await _gather_candidates(
        qa.question, search_papers, search_snippets,
        client=client, query_model=query_model, n_queries=n_queries, k=k, pool_cap=pool_cap,
    )
    # Citation members first (high-value, search may miss them), then search hits.
    extra = [int(c) for c in (seed_candidate_ids or [])]
    candidates = [c for c in dict.fromkeys(extra + searched) if c != qa.seed_paper_id][:pool_cap]

    sem = asyncio.Semaphore(concurrency)

    async def grade(cid: int) -> tuple[int, int]:
        async with sem:
            return cid, await _grade_candidate(cid, qa.criteria, paper_info, client=client, model=model)

    graded = await asyncio.gather(*[grade(c) for c in candidates])
    relevance = {qa.seed_paper_id: 3}  # seed satisfies all criteria by construction
    for cid, g in graded:
        if g >= 1:
            relevance[cid] = g

    # PaperFindingBench counts only PERFECT (grade-3) as relevant for recall; grade 1-2
    # are ranking signal (nDCG) only. Gate on the grade-3 count — that's the recall set.
    n_perfect = sum(1 for g in relevance.values() if g == 3)
    if not (min_size <= n_perfect <= max_size):
        log.info("paper_set %s dropped on size-gate: |grade-3|=%d (allowed %d-%d)",
                 qa.qa_id, n_perfect, min_size, max_size)
        return None

    ordered = tuple(sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0])))
    return dataclasses.replace(qa, relevance=ordered, est_total_relevant=n_perfect)


def _dcg(grades: list[int]) -> float:
    # PaperFindingBench find_dcg: position 1 -> /log(2), position 2 -> /log(3), ...
    return sum(g / math.log(i + 1) for i, g in enumerate(grades, start=1))


def _ndcg_rank(grades: list[int]) -> float:
    """Normalized DCG of the submission order vs ideal/worst (PFB lower_bound_corrected_ndcg),
    but degenerate-safe: a uniform all-relevant order is 'perfectly ordered' (1.0), not 0
    (PFB returns 0 there, which would zero out a correct answer — bad as a training reward)."""
    if not grades:
        return 0.0
    hi, lo = _dcg(sorted(grades, reverse=True)), _dcg(sorted(grades))
    if hi == lo:
        return 1.0 if hi > 0 else 0.0
    return (_dcg(grades) - lo) / (hi - lo)


def _harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a > 0 and b > 0) else 0.0


def score_paper_set(
    predicted_ids: list[int],
    relevance,
    *,
    est_total_relevant: int | None = None,
    rel_threshold: int = 3,
) -> dict:
    """Score an ORDERED predicted set against graded gold — the PaperFindingBench
    *semantic* metric `adjusted_f1 = harmonic(recall@est, nDCG-rank)`.

    `relevance` maps corpus_id -> grade 1-3 (a dict or (id, grade) pairs); an
    unsubmitted/unknown paper scores 0. `reward` is the harmonic mean of:
      - **recall@est**: among the top-`est` of the agent's ORDERED submission, the
        fraction "relevant" over total relevant. Per PFB, relevant = **grade-3
        (PERFECT) only** (`rel_threshold=3`); grade 1-2 do NOT count toward recall.
        Truncation to `est` bounds padding — there is NO precision term (matches the
        broad eval metric; precision is only the `specific`-query metric).
      - **rank**: nDCG over ALL submitted grades (0-3) in order — grade 1-2 are the
        ranking signal that rewards most-relevant-first.
    `est_total_relevant` should be the grade-3 count. `predicted_ids` MUST be the
    agent's ranked list (order matters).
    """
    rel = {int(c): int(g) for c, g in (relevance.items() if isinstance(relevance, dict) else relevance)}
    preds = list(dict.fromkeys(int(p) for p in predicted_ids))  # dedup, preserve order
    total = est_total_relevant if est_total_relevant is not None else sum(1 for g in rel.values() if g >= rel_threshold)

    relevant_at_est = sum(1 for p in preds[:total] if rel.get(p, 0) >= rel_threshold)
    recall = relevant_at_est / total if total else 0.0
    rank = _ndcg_rank([rel.get(p, 0) for p in preds])

    return {
        "reward": _harmonic(recall, rank),
        "recall_at_est": recall,
        "rank": rank,
        "relevant_at_est": relevant_at_est,
        "n_pred": len(preds),
        "n_gold": total,
    }


# The agent submits its set via the existing `submit_answer(answer: str)` as the same
# JSON AstaBench PaperFindingBench expects, so the training submission matches the eval
# interface (no separate tool). PaperFindingBench parses with json.loads first, falling
# back to an LLM (`parse_result_model`, gpt-4o) — we mirror that, defaulting the fallback
# to gpt-oss-120b. See docs/superpowers/specs/2026-06-11-paper-set-answer-type-design.md.
PARSE_PROMPT = """The text below is an agent's answer listing the papers it judged relevant to a query. Extract the corpus_ids of those papers.
Return ONLY JSON (no markdown): {{"paper_ids": [<int>, ...]}}. If there are none, return {{"paper_ids": []}}.

[ANSWER]
{answer}"""


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _ids_from_json(text: str) -> list[int]:
    """Extract corpus_ids from the bench-shaped JSON, raising on any unrecognized shape.
    Accepts {"output": {"results": [...]}}, {"results": [...]}, or {"paper_ids": [...]}."""
    obj = json.loads(_strip_fence(text))
    if isinstance(obj, dict) and isinstance(obj.get("output"), dict):
        obj = obj["output"]
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return [int(r["paper_id"]) for r in obj["results"] if isinstance(r, dict) and "paper_id" in r]
    if isinstance(obj, dict) and isinstance(obj.get("paper_ids"), list):
        return [int(x) for x in obj["paper_ids"]]
    raise ValueError("unrecognized submission JSON shape")


async def parse_paper_set_submission(
    answer_text: str,
    *,
    client: openai.AsyncOpenAI | None = None,
    model: str | None = None,
) -> tuple[list[int], str]:
    """Parse the agent's `submit_answer` string into corpus_ids for scoring.

    Mirrors PaperFindingBench: try `json.loads` (the bench shapes) first; on failure,
    fall back to an LLM extraction (`client`/`model`, e.g. gpt-oss-120b) if configured.
    Returns `(paper_ids, how)` where `how` is "json" | "llm" | "empty" — log `how` to
    track the LLM-fallback rate (high rate ⇒ the policy isn't emitting clean JSON).
    """
    text = answer_text or ""
    try:
        return list(dict.fromkeys(_ids_from_json(text))), "json"
    except Exception:
        pass
    if client is None or model is None:
        return [], "empty"
    payload = await chat_json(client, model, [{"role": "user", "content": PARSE_PROMPT.format(answer=text[:8000])}],
                              temperature=0.0)
    if isinstance(payload, dict) and isinstance(payload.get("paper_ids"), list):
        try:
            return list(dict.fromkeys(int(re.sub(r"\D", "", str(x)) or 0) for x in payload["paper_ids"] if str(x).strip())), "llm"
        except (ValueError, TypeError):
            return [], "empty"
    return [], "empty"


async def score_submission(
    answer_text: str,
    qa: PaperSetQA,
    *,
    client: openai.AsyncOpenAI | None = None,
    model: str | None = None,
) -> dict:
    """End-to-end training reward: parse the agent's ORDERED submission, then score it
    against the QA's graded gold (adjusted_f1). Adds `parse` ("json"|"llm"|"empty")."""
    ids, how = await parse_paper_set_submission(answer_text, client=client, model=model)
    out = score_paper_set(ids, qa.relevance, est_total_relevant=qa.est_total_relevant)
    out["parse"] = how
    return out
