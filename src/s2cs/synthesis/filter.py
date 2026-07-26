import asyncio
import dataclasses
import logging
from typing import Callable

import openai

from s2cs.agent.judge import judge
from s2cs.agent.llm import chat, chat_json
from s2cs.synthesis.single_hop import SingleHopQA

log = logging.getLogger(__name__)

CLOSED_BOOK_PROMPT = """Answer the question as concisely as possible — just the answer, no explanation. If you do not know, say "I don't know".

Question: {question}
"""


async def requires_retrieval(
    qa: SingleHopQA,
    *,
    client: openai.AsyncOpenAI,
    model: str,
) -> bool:
    """True if the question genuinely needs retrieval to answer.

    Asks the model closed-book (no paper, no tools) and grades it with the
    agent's semantic-equivalence judge. A question the model already answers
    from parametric memory trains no search skill, so it is dropped (returns
    False); only questions the closed-book model gets wrong are kept.
    """
    closed_book = await chat(
        client,
        model,
        [{"role": "user", "content": CLOSED_BOOK_PROMPT.format(question=qa.question)}],
        temperature=0.0,
    )
    verdict = await judge(qa.question, qa.answer, closed_book.text, client=client, model=model)
    return verdict.verdict == "Incorrect"


RUBRIC_CRITERIA = ("grounded", "closed_form", "unique", "own_work", "findable")


@dataclasses.dataclass(frozen=True)
class RubricVerdict:
    grounded: bool
    closed_form: bool
    unique: bool
    own_work: bool
    findable: bool
    reasoning: str

    @property
    def passed(self) -> bool:
        return all(getattr(self, c) for c in RUBRIC_CRITERIA)


RUBRIC_PROMPT = """You are auditing a single-hop literature QA item for a paper-search training set. Judge it on five independent yes/no criteria, using the paper body provided.

[QUESTION]
{question}

[GOLD ANSWER]
{answer}

[EVIDENCE] (the quote the question was drawn from)
{evidence}

[PAPER BODY]
{body}

Judge each criterion true or false:
- grounded: the gold answer is correct and directly supported by the paper body.
- closed_form: the answer is a short, specific value (number, name, dataset, metric, entity), NOT a long descriptive sentence or open-ended explanation.
- unique: the question pins down exactly ONE correct answer; no other passage in the body gives a different but equally valid answer.
- own_work: the answer is the paper's OWN contribution or finding, NOT bibliographic metadata (date, venue, authors) and NOT a fact the paper attributes to or cites from other work.
- findable: from the QUESTION ALONE a searcher could identify which paper to retrieve (it names a specific system / method / dataset), not vague like "the study" or "the proposed method".

Return ONLY a JSON object, no prose, no markdown:
{{"grounded": true, "closed_form": true, "unique": true, "own_work": true, "findable": true, "reasoning": "<one short sentence>"}}"""


async def rubric_verify(
    qa: SingleHopQA,
    body: str,
    *,
    client: openai.AsyncOpenAI,
    model: str,
) -> RubricVerdict:
    """Score a single-hop QA on five quality criteria with an LLM judge.

    Reads the question, gold answer, evidence quote, and paper body, and returns
    per-criterion booleans. `passed` is the AND of all five — the quality gate.
    Unlike a rollout, this judges QA quality intrinsically, so it does not drop
    good-but-hard items just because the current agent or retrieval can't solve
    them. On JSON failure it fails closed (all False).
    """
    prompt = RUBRIC_PROMPT.format(question=qa.question, answer=qa.answer, evidence=qa.evidence, body=body)
    payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=0.0)
    if payload is None:
        log.warning("rubric_verify JSON failure for %s; failing closed", qa.qa_id)
        return RubricVerdict(False, False, False, False, False, "judge JSON parse failure")
    return RubricVerdict(
        grounded=bool(payload.get("grounded")),
        closed_form=bool(payload.get("closed_form")),
        unique=bool(payload.get("unique")),
        own_work=bool(payload.get("own_work")),
        findable=bool(payload.get("findable")),
        reasoning=str(payload.get("reasoning", "")),
    )


QUERY_GEN_PROMPT = """You are searching for the specific paper this question is about. Write the best concise search query (3 to 10 words, key named entities only) — the system / method / dataset / phenomenon names that uniquely identify the paper. Do NOT include question framing like "what is", "in the", "the proposed". Do NOT include the answer.

Question: {question}

Return ONLY the search query as plain text — no quotes, no markdown, no explanation."""


async def make_search_query(
    question: str,
    *,
    client: openai.AsyncOpenAI,
    model: str,
) -> str:
    """Generate a concise search query for a single-hop QA's question.

    Mirrors what a retrieval-time agent would do: distill the question into
    named entities for the search engine. Sees the question only — no body, no
    gold answer — to keep the empirical retrieval test honest (no body-token leak).
    """
    res = await chat(
        client, model,
        [{"role": "user", "content": QUERY_GEN_PROMPT.format(question=question)}],
        temperature=0.0,
    )
    q = res.text.strip()
    if len(q) >= 2 and q[0] in "\"'" and q[-1] == q[0]:
        q = q[1:-1]
    return q


@dataclasses.dataclass(frozen=True)
class RetrievalCheck:
    query: str
    found: bool
    paper_rank: int | None       # 1-indexed; None if not in top-k of paper_search
    snippet_rank: int | None     # 1-indexed dedup-by-paper rank in snippet_search; None if absent


async def retrievable(
    qa: SingleHopQA,
    paper_search: Callable,
    search_snippets: Callable,
    *,
    client: openai.AsyncOpenAI,
    query_model: str,
    k: int = 10,
) -> RetrievalCheck:
    """Empirical retrievability: can our env surface the seed paper for this QA?

    Generates a search query from the question alone (model-derived, no body /
    no gold), then runs paper_search + search_snippets over the live env. The
    paper is `found` if its `seed_paper_id` appears in the top-k of either —
    measuring "is there a question-derivable query our search can resolve?".
    Distinct from a rollout: no agent loop, no policy reasoning, no judge.
    """
    query = await make_search_query(qa.question, client=client, model=query_model)
    p_hits, s_hits = await asyncio.gather(
        asyncio.to_thread(paper_search, query, limit=k),
        asyncio.to_thread(search_snippets, query, limit=k),
    )
    paper_rank = next((i + 1 for i, h in enumerate(p_hits) if h.corpus_id == qa.seed_paper_id), None)
    seen: list[int] = []
    snippet_rank: int | None = None
    for h in s_hits:
        if h.paper_corpus_id not in seen:
            seen.append(h.paper_corpus_id)
            if h.paper_corpus_id == qa.seed_paper_id:
                snippet_rank = len(seen)
                break
    return RetrievalCheck(
        query=query,
        found=(paper_rank is not None) or (snippet_rank is not None),
        paper_rank=paper_rank,
        snippet_rank=snippet_rank,
    )
