"""Multi-hop citation-bridge QA: the gold answer concerns a paper B reached only
through an intermediate paper A's citing context.

Tier-3 Citation-bridge (forward), pure-bridge: the question identifies A by one of
its OWN reported details, refers to B ONLY as "the work A connects to that detail",
and asks for a fact about B. B's own properties never appear in the question, so the
agent must read A to resolve B. See docs/design/2026-06-16-multihop-citation-bridge-design.md
and docs/query_design.md §3 (Tier-3) / §4 (answer ladder).

Two stages mirror paper_set's gen->freeze:
- SELECT (`make_multi_hop_select`) — policy-agnostic, from A's body + its in-corpus
  cited papers (title+abstract): pick a verbatim citing passage in A that discusses
  ONE cited work, name which one (B), draft the A-context bridge cue.
- GROUND (`ground_multi_hop_answer`) — load B's text, write the final question + a
  B-grounded answer (value/abstract/identity), verbatim-checked against B.
"""

import asyncio
import dataclasses
import logging
import re
from typing import Awaitable, Callable

import openai

from s2cs.agent.llm import chat_json
from s2cs.synthesis.edge_store import Edge
from s2cs.synthesis.single_hop import ANCHOR_RULES

log = logging.getLogger(__name__)

ANSWER_TYPES = frozenset({"value", "abstract", "identity"})

# A-anchor tiers (how the question identifies the intermediate paper A) — reuse single_hop's
# anchor ladder. pure-bridge applies to B only, so A may be named. (Citation anchors
# relational/full_span don't apply to identifying A.)
A_ANCHORS = frozenset({"named", "detail_cue", "paraphrastic", "content_conjunction"})


# The A-check and B-check generate queries with DIFFERENT intents (different targets), so
# they use different prompts. A-check: find the framing paper A (identified by its own
# reported detail). B-check: adversarially try to reach the cited answer-paper B DIRECTLY,
# shortcutting past A — using only what the question reveals about B.
A_QUERY_PROMPT = """You are locating ONE specific paper in a large scientific corpus with a search engine.
The question below is FRAMED AROUND a paper ("the framing paper") identified by its OWN reported detail (a value, result, dataset, or setup that paper reports), and then goes on to ask about a SECOND paper that the framing paper cites. Your target is the FRAMING paper ONLY.

Build EVERY query from how the question describes the FRAMING paper. IGNORE the second/cited paper the question ultimately asks about — do NOT search for ITS content (its values, methods, findings, or name): those describe a different paper and will not surface the framing paper. Available tools:
- search_papers: hybrid search over titles/abstracts; can filter by year_min, year_max, venue.
- search_snippets: search over paper BODY text — use it for the FRAMING paper's specific reported value, finding, or setup.
Vary the attempts (keyword vs natural-language, papers vs snippets). Use year/venue ONLY if the question states them. Do NOT put the answer in a query.

Question: {q}

Return ONLY a JSON list of up to {n} objects, no prose:
[{{"tool": "search_papers" | "search_snippets", "query": "...", "year_min": <int or null>, "year_max": <int or null>, "venue": <str or null>}}]"""

B_QUERY_PROMPT = """You are trying to SHORTCUT a multi-hop literature question. The question below is framed around one paper but ultimately asks about a SECOND paper that the first one cites/discusses. Using ONLY what the question reveals about that SECOND (cited) paper, propose up to {n} search attempts that try to find THAT second paper DIRECTLY — as if skipping the first paper entirely. Available tools:
- search_papers: hybrid search over titles/abstracts; can filter by year_min, year_max, venue.
- search_snippets: search over paper BODY text — use it for a specific reported value, finding, or setup.
If the question reveals little about the second paper on its own, your queries will necessarily be weak — that is fine, write the best you can. Do NOT put the answer value in a query.

Question: {q}

Return ONLY a JSON list of up to {n} objects, no prose:
[{{"tool": "search_papers" | "search_snippets", "query": "...", "year_min": <int or null>, "year_max": <int or null>, "venue": <str or null>}}]"""


def _parse_queries(payload: object, n: int) -> list[dict]:
    items = payload if isinstance(payload, list) else []
    out: list[dict] = []
    for it in items[:n]:
        if isinstance(it, dict) and str(it.get("query", "")).strip():
            tool = it.get("tool") if it.get("tool") in ("search_papers", "search_snippets") else "search_papers"
            out.append({"tool": tool, "query": str(it["query"]).strip(),
                        "year_min": it.get("year_min"), "year_max": it.get("year_max"),
                        "venue": (it.get("venue") or None)})
    return out


async def _surfaced_ranks(
    question: str, prompt_tmpl: str, *,
    search_papers: Callable, search_snippets: Callable,
    client: openai.AsyncOpenAI, model: str, n: int, k: int,
) -> dict[int, int]:
    """Like `_surfaced_ids` but returns {corpus_id: best (min) 1-indexed rank across all
    attempts}. Lets callers distinguish "B is the TOP hit" (a real shortcut) from "B is
    merely present among topical neighbors" (not identifiable without A)."""
    payload = await chat_json(client, model, [{"role": "user", "content": prompt_tmpl.format(q=question, n=n)}],
                              temperature=0.7)
    best: dict[int, int] = {}
    for q in _parse_queries(payload, n):
        try:
            if q["tool"] == "search_snippets":
                hits = await asyncio.to_thread(search_snippets, q["query"], limit=k)
                ids = [int(h.paper_corpus_id) for h in hits]
            else:
                hits = await asyncio.to_thread(search_papers, q["query"], limit=k,
                                               year_min=q.get("year_min"), year_max=q.get("year_max"),
                                               venue=q.get("venue"))
                ids = [int(h.corpus_id) for h in hits]
        except Exception as exc:
            log.warning("retrievability query failed (%r): %s", q.get("query"), exc)
            continue
        for rank, cid in enumerate(ids, start=1):
            if cid not in best or rank < best[cid]:
                best[cid] = rank
    return best


async def probe_chain(
    question: str,
    start_id: int,
    terminal_id: int,
    *,
    search_papers: Callable,
    search_snippets: Callable,
    client: openai.AsyncOpenAI,
    query_model: str,
    n_queries: int = 6,
    k: int = 15,
    b_rank: int = 3,
) -> tuple[bool, bool]:
    """Two-part chain validity, mirroring single_hop's retrievability filter applied
    SYMMETRICALLY with target-specific prompts (two separate query-gen calls). Returns
    `(start_found, terminal_found)`:
    - `start_found` — the START node surfaces (any rank within top-`k`) from `A_QUERY_PROMPT`;
    - `terminal_found` — the TERMINAL surfaces as a TOP hit (best rank <= `b_rank`) from `B_QUERY_PROMPT`.

    Keep the QA iff `start_found and not terminal_found`. terminal_found is RANK-AWARE on purpose:
    the terminal is topically close to the chain and almost always appears *somewhere* in a search
    (measured: ~80% in top-15). Mere presence is not a shortcut — the agent cannot tell which of the
    topical neighbors is the target without walking the chain. A real shortcut is the terminal
    landing at the TOP (`b_rank`, default 3): then the agent can pin it directly, skipping the chain.
    """
    a_ranks = await _surfaced_ranks(question, A_QUERY_PROMPT, search_papers=search_papers,
                                    search_snippets=search_snippets, client=client, model=query_model,
                                    n=n_queries, k=k)
    b_ranks = await _surfaced_ranks(question, B_QUERY_PROMPT, search_papers=search_papers,
                                    search_snippets=search_snippets, client=client, model=query_model,
                                    n=n_queries, k=k)
    start_found = int(start_id) in a_ranks
    br = b_ranks.get(int(terminal_id))
    terminal_found = br is not None and br <= b_rank
    return start_found, terminal_found


# Back-compat alias: the 2-hop path and its tests call check_hop(question, A, B, ...).
check_hop = probe_chain


@dataclasses.dataclass(frozen=True)
class MultiHopDraft:
    intermediate_paper_id: int   # A
    gold_paper_id: int           # B (a paper A cites, chosen in SELECT)
    evidence_a: str              # verbatim passage in A's body that discusses B
    b_label: str                 # how A refers to B in that passage (the bridge cue, A's words)
    a_cue: str = ""              # referring phrase that identifies A (per a_anchor tier)
    a_anchor: str = "detail_cue" # which A-anchor tier produced a_cue


@dataclasses.dataclass(frozen=True)
class MultiHopQA:
    qa_id: str
    question: str
    answer: str
    answer_type: str             # value | abstract | identity
    path: tuple                  # ordered node ids: path[0]=start, path[-1]=terminal (gold)
    edges: tuple                 # Edge per hop; len == len(path) - 1
    evidence_b: str              # verbatim in the terminal's text (supports the answer)
    anchor: str = "detail_cue"
    source: str = "multi_hop"

    @property
    def hops(self) -> int:       # number of NODES (K); = len(edges) + 1
        return len(self.path)

    @property
    def intermediate_paper_id(self) -> int:  # start node (= A at K=2)
        return self.path[0]

    @property
    def gold_paper_id(self) -> int:          # terminal node (= B at K=2)
        return self.path[-1]

    def to_record(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "anchor": self.anchor,
            "hops": self.hops,
            "path": list(self.path),
            "edges": [e.to_record() for e in self.edges],
            "seed_paper_ids": list(self.path[:-1]),  # all non-terminal nodes; K=2 -> [A]
            "intermediate_paper_id": self.intermediate_paper_id,
            "gold_paper_id": self.gold_paper_id,
            "evidence_a": self.edges[0].citing_evidence,  # back-compat: first hop's citing passage
            "evidence_b": self.evidence_b,
            "rubric_pass": None,
            "stage": None,
            "pass_at_8": None,
        }


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


# SELECT is B-first (two calls) to ground the A->B link by construction. CHOOSE picks the
# target B; GROUND-passage then names that SPECIFIC B and asks for A's discussion of it (or
# {}). Naming B before grounding removes the "guess which cited work this passage is" error
# that produced broken bridges (a wrong B with a real-looking answer).
# SELECT is B-first: CHOOSE a target B from A's KNOWN in-corpus citations (so B is always a
# real, in-corpus A->B edge — never a non-corpus work the model merely names), then PASSAGE
# grounds the citing sentence in that specific B. B-first guarantees the citation relationship;
# the VERIFY step (VERIFY_PROMPT) then catches the failure B-first introduces — PASSAGE
# fabricating a sentence when A doesn't actually discuss B — by dropping any evidence_a that
# doesn't really cite B.
CHOOSE_PROMPT = """You set up a MULTI-HOP literature question from the paper below ("A") and the in-corpus works it cites.

[PAPER A]
{public}

[BODY A] (A's full text)
{body}

[CITED] (in-corpus papers A cites)
{cited_block}

Identify ONE cited work from [CITED] that A discusses in [BODY A] DISTINCTIVELY — A characterizes that work (its method, role, finding, or how A uses or compares it) specifically enough that the discussion points to that one cited work and no other. A bare "[N]"-style mention with no characterization does NOT count. That work will be the hidden target "B"; pick the one A discusses most distinctively.

Return ONLY a JSON object (no prose, no markdown):
{{"cited_label": "<the C-number of that work, e.g. C2>"}}
If A does not discuss any single listed work distinctively enough to single it out, return {{}}."""

PASSAGE_PROMPT = """Below is the full text of paper A, and ONE specific work that A cites ("B").

[BODY A]
{body}

[B] (the specific cited work to locate in A)
Title: {b_title}
Abstract: {b_abstract}

Find the place in [BODY A] where A discusses THIS specific work B DISTINCTIVELY — characterizing its method, role, finding, or how A uses or compares it — specifically enough to single out B (not a bare "[N]" mention with no characterization). The passage must genuinely be about B (the work matching the title/abstract above), not some other reference.

Return ONLY a JSON object (no prose, no markdown):
{{"evidence_a": "<a VERBATIM sentence/clause copied from [BODY A] that distinctively discusses B>",
  "b_context": "<a back-reference to B THROUGH A's relationship to it — the ROLE B plays for A in this passage (e.g. 'the baseline A compares its results against', 'the dataset A reuses', 'the method A adapts its pipeline from'). Do NOT restate B's own task, method, domain, or findings, and do NOT use B's title or coined name. It must read as a pointer that only resolves once you have read A — NOT a standalone description by which B itself could be searched.>"}}
If A does not actually discuss this specific work B distinctively, return {{}}."""

VERIFY_PROMPT = """A sentence from paper A cites another work. Decide whether the sentence is actually citing/discussing the CANDIDATE paper below (matching its title/abstract), or some DIFFERENT work.

[SENTENCE FROM A]
{evidence_a}

[CANDIDATE PAPER]
Title: {b_title}
Abstract: {b_abstract}

Be STRICT: answer false if the sentence's topic/contribution does not clearly match the candidate (different method, different task, different domain), even if loosely related.

Return ONLY a JSON object (no prose, no markdown):
{{"match": true | false, "reason": "<one short clause>"}}"""


A_ANCHOR_PROMPT = """Write a short REFERRING PHRASE that points at the paper below ("A"). It will be slotted into a larger question to identify A, so a reader must be able to pick A out from it. Do NOT mention any OTHER work A cites.

[PAPER A]
{public}

[BODY A]
{body}

How to point at A:
- {anchor_rule}

Return ONLY a JSON object (no prose, no markdown):
{{"a_cue": "<a noun phrase referring to A, e.g. 'the paper reporting 75.3 accuracy on WikiText-103' or 'the FlashAttention paper'>"}}
If you cannot, return {{}}."""


# named (T0) for multi-hop A: A is the intermediate paper (pure-bridge applies to B only),
# so we may name A. But the raw corpus title is often dirty (citation strings, "Special Issue"
# editorial notes, author handles, DOIs), so we let the model produce a CLEAN named handle and
# drop junk A's — rather than splicing the title verbatim. This is NOT single_hop's named rule
# (which forbids the title and demands a coined name → over-drops papers that coin nothing).
NAMED_A_RULE = (
    "Give a clean NAMED handle a reader can pick A out by. Prefer a specific name A coins for "
    "its OWN contribution (a system, method, model, or dataset name) — e.g. 'the FlashAttention "
    "paper'. If A coins no such name, use a cleaned form of its title: strip citation/editorial "
    "boilerplate ('Citation:', 'Special Issue on', 'Editorial', 'Erratum'), author names, and DOIs, "
    "phrased as 'the paper titled \"...\"'. Return {} if A is an editorial/erratum/front-matter or "
    "has no identifiable contribution and no usable title."
)


def _format_public(title, abstract, summary) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if summary:
        parts.append(f"Summary: {summary}")
    return "\n\n".join(parts)


def make_edge_discoverer(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> Callable[[dict], Awaitable[Edge | None]]:
    """Build the edge discoverer: from a paper (`corpus_id, title, abstract, summary,
    body`) plus `cited` (its in-corpus cited papers `{corpus_id, title, abstract}`),
    run CHOOSE -> PASSAGE -> VERIFY and return a structural `Edge` (from -> to with the
    verbatim citing passage and the relationship pointer), or None. No anchor, no answer.
    """

    async def discover_edge(paper: dict) -> Edge | None:
        cid = int(paper["corpus_id"])
        public = _format_public(paper.get("title"), paper.get("abstract"), paper.get("summary"))
        body = paper.get("body") or ""
        cited = paper.get("cited") or []
        if not public or not body or not cited:
            return None
        if max_body_chars is not None:
            body = body[:max_body_chars]

        cited_block = "\n".join(
            f"C{i + 1}. {c['title']}\n    {(c.get('abstract') or '')[:400]}"
            for i, c in enumerate(cited)
        )
        # CHOOSE: pick a target B from the paper's KNOWN in-corpus citations (B is a real edge).
        choose = await chat_json(
            client, model,
            [{"role": "user", "content": CHOOSE_PROMPT.format(public=public, body=body, cited_block=cited_block)}],
            temperature=temperature,
        )
        if not isinstance(choose, dict) or choose.get("cited_label") is None:
            return None
        m = re.search(r"\d+", str(choose["cited_label"]))
        if m is None:
            return None
        pos = int(m.group()) - 1  # C1 -> index 0
        if pos < 0 or pos >= len(cited):
            return None
        b = cited[pos]

        # PASSAGE: ground the citing sentence in that SPECIFIC named B (or {} -> miss).
        passage = await chat_json(
            client, model,
            [{"role": "user", "content": PASSAGE_PROMPT.format(
                body=body, b_title=b.get("title") or "", b_abstract=(b.get("abstract") or "")[:600])}],
            temperature=temperature,
        )
        if not isinstance(passage, dict):
            return None
        evidence_a = str(passage.get("evidence_a", "")).strip()
        b_context = str(passage.get("b_context", "")).strip()
        if not evidence_a or not b_context:
            return None
        if _norm(evidence_a) not in _norm(body):  # must be VERBATIM in the citing paper's body
            return None

        # VERIFY: does evidence_a actually cite B? Catches PASSAGE fabricating a sentence when the
        # paper doesn't really discuss B (the failure B-first introduces). Reject on no-match.
        verdict = await chat_json(
            client, model,
            [{"role": "user", "content": VERIFY_PROMPT.format(
                evidence_a=evidence_a, b_title=b.get("title") or "", b_abstract=(b.get("abstract") or "")[:600])}],
            temperature=0.0,
        )
        if not isinstance(verdict, dict) or verdict.get("match") is not True:
            return None

        return Edge(cid, int(b["corpus_id"]), evidence_a, b_context)

    return discover_edge


def make_anchor(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    a_anchor: str = "detail_cue",
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> Callable[[dict], Awaitable[str | None]]:
    """Build the anchor stage: a referring phrase (`a_cue`) for a chain's START node,
    per the chosen tier (the next node is never mentioned). The LLM call also screens the
    paper: it cleans dirty titles and returns {} for junk (editorials, citation-string
    titles). `named` uses NAMED_A_RULE; other tiers reuse single_hop ANCHOR_RULES.
    """
    if a_anchor not in A_ANCHORS:
        raise ValueError(f"unknown a_anchor {a_anchor!r}; expected one of {sorted(A_ANCHORS)}")

    async def anchor(paper: dict) -> str | None:
        public = _format_public(paper.get("title"), paper.get("abstract"), paper.get("summary"))
        body = paper.get("body") or ""
        if max_body_chars is not None:
            body = body[:max_body_chars]
        anchor_rule = NAMED_A_RULE if a_anchor == "named" else ANCHOR_RULES[a_anchor]
        anchored = await chat_json(
            client, model,
            [{"role": "user", "content": A_ANCHOR_PROMPT.format(
                public=public, body=body, anchor_rule=anchor_rule)}],
            temperature=temperature,
        )
        if not isinstance(anchored, dict):
            return None
        return str(anchored.get("a_cue", "")).strip() or None

    return anchor


def make_multi_hop_select(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    a_anchor: str = "detail_cue",
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> Callable[[dict], Awaitable[MultiHopDraft | None]]:
    """The 2-hop SELECT synthesizer: an edge (discover_edge) plus a START anchor, packed
    into a `MultiHopDraft`. Kept for the single-bridge path; N-hop chains use the two
    pieces directly. `a_anchor` (in A_ANCHORS) picks how the question refers to A.
    """
    discover = make_edge_discoverer(client, model, max_body_chars=max_body_chars, temperature=temperature)
    anchor = make_anchor(client, model, a_anchor=a_anchor, max_body_chars=max_body_chars, temperature=temperature)

    async def select(paper: dict) -> MultiHopDraft | None:
        edge = await discover(paper)
        if edge is None:
            return None
        a_cue = await anchor(paper)
        if not a_cue:
            return None
        return MultiHopDraft(
            intermediate_paper_id=edge.from_id,
            gold_paper_id=edge.to_id,
            evidence_a=edge.citing_evidence,
            b_label=edge.pointer_label,
            a_cue=a_cue,
            a_anchor=a_anchor,
        )

    return select


ANSWER_RULES = {
    "value": (
        "Example KINDS of fact that fit: a COUNT B reports about its own work "
        "(how many models / benchmarks / datasets / baselines it evaluated); a SUPERLATIVE "
        "(its best-performing variant and that variant's reported score); a CATEGORICAL choice "
        "(the primary metric, dataset, or task it adopts); or a SPECIFIC reported VALUE (a "
        "result, quantity, or setup). 'answer' is copied in the exact form B states it; "
        "'evidence' is the verbatim sentence from [BODY B] stating it."
    ),
    "abstract": (
        "a short closed-form fact stated in B's ABSTRACT (B's task, a headline result, a "
        "dataset, or a key claim) — not bibliographic metadata. 'evidence' is the verbatim "
        "abstract sentence supporting it."
    ),
    "identity": (
        "B's identity — the name B coins for its own contribution (a method/system/dataset "
        "name) or, lacking one, its title — copied exactly. 'evidence' is the verbatim "
        "sentence naming it."
    ),
}

GROUND_CHAIN_PROMPT = """You finish a MULTI-HOP literature question. You are given how to refer to the START paper, an ordered chain of pointers leading hop-by-hop to a TERMINAL paper, and the TERMINAL paper's text.

[REFER TO START AS]
{a_cue}

[POINTER CHAIN] (each step points at the next work ONLY through the previous work's relationship — never by a name)
{chain_block}

[TERMINAL PAPER]
Title: {b_title}
Abstract: {b_abstract}

[BODY] (terminal paper's full text)
{b_body}

Write ONE natural, self-contained question for a research agent, plus its answer and a verbatim evidence span from the TERMINAL paper.

The QUESTION must:
- identify the START using [REFER TO START AS], then walk the [POINTER CHAIN] in order ("the work that one in turn refers to as ..."), ending at the terminal;
- NEVER name, quote, or paraphrase the title or coined name of ANY intermediate or the terminal paper (each must be reachable only by following the chain);
- NEVER describe the terminal's OWN task, method, domain, dataset, or findings in searchable terms — it must be unidentifiable from the question alone;
- read as ONE clean sentence — do NOT copy long verbatim passages, and do NOT cite author names or markers like "[12]" / "Smith et al.".

The ANSWER must satisfy ALL of this RUBRIC (the kinds listed below are only examples, not a closed list):
1. GROUNDED — explicitly stated in the terminal's text; 'evidence' is a verbatim span from it.
2. ANCHORED IN THE TERMINAL — a concrete detail that only the terminal's text supplies: a result, quantity, or setup it reports, or a specific choice made (which dataset, tool, method, component, or value was used; how many things were evaluated).
3. READ-REQUIRED (the real gate) — you must read the terminal to answer; it cannot be derived from the question's own words or from general knowledge. NEVER ask for the expansion, definition, or textbook meaning of an acronym, term, or standard that the question names.
4. CLOSED-FORM & UNIQUE — one short, unambiguous answer (a number, name, or short phrase) that the terminal determines uniquely.
5. DISCRIMINATIVE — the same question asked about a DIFFERENT paper would plausibly have a different answer.

{answer_rule}

The agent cannot see any of this text; the question must stand alone.

Return ONLY a JSON object (no prose, no markdown):
{{"question": "...", "answer": "...", "evidence": "..."}}
If no fact in the terminal satisfies the rubric, return {{}}."""


def _haystack(answer_type: str, b_title, b_abstract, body: str) -> str:
    if answer_type == "value":
        return body
    if answer_type == "abstract":
        return b_abstract or ""
    return f"{b_title or ''}\n{b_abstract or ''}\n{body}"  # identity


async def ground_chain(
    a_cue: str,
    edges: list,
    *,
    b_title: str | None,
    b_abstract: str | None,
    b_body: str,
    answer_type: str,
    client: openai.AsyncOpenAI,
    model: str,
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> MultiHopQA | None:
    """GROUND stage for an N-hop chain: compose the START anchor + ordered edge pointers
    into one question whose answer is grounded in the TERMINAL (= edges[-1].to_id). Verbatim-
    checks answer/evidence against the right slice of the terminal per `answer_type`, and
    enforces pure-bridge (terminal title absent from the question) + no answer leak. The
    single-edge case reproduces the 2-hop behavior. Returns None on any failure."""
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"unknown answer_type {answer_type!r}; expected one of {sorted(ANSWER_TYPES)}")
    if not edges:
        return None
    body = b_body or ""
    if max_body_chars is not None:
        body = body[:max_body_chars]
    haystack = _haystack(answer_type, b_title, b_abstract, body)
    if not haystack.strip():
        return None

    chain_block = "\n".join(
        f"{i + 1}. {'the start work' if i == 0 else 'that work'} refers to a next work as: {e.pointer_label}"
        for i, e in enumerate(edges)
    )
    prompt = GROUND_CHAIN_PROMPT.format(
        a_cue=a_cue, chain_block=chain_block,
        b_title=b_title or "", b_abstract=b_abstract or "", b_body=body,
        answer_rule=ANSWER_RULES[answer_type],
    )
    payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=temperature)
    if not isinstance(payload, dict):
        return None
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    evidence = str(payload.get("evidence", "")).strip()
    if not question or not answer or not evidence:
        return None

    q = _norm(question)
    if b_title and _norm(b_title) in q:          # pure-bridge: never name the terminal
        return None
    if _norm(answer) in q:                        # answer must not leak into the question
        return None
    if _norm(evidence) not in _norm(haystack):    # evidence verbatim in the terminal (right slice)
        return None
    if _norm(answer) not in _norm(haystack):      # answer grounded in the terminal
        return None

    path = tuple([edges[0].from_id] + [e.to_id for e in edges])
    return MultiHopQA(
        qa_id=f"multi_hop_{'_'.join(str(n) for n in path)}_{answer_type}",
        question=question,
        answer=answer,
        answer_type=answer_type,
        path=path,
        edges=tuple(edges),
        evidence_b=evidence,
    )


async def ground_multi_hop_answer(
    draft: MultiHopDraft,
    *,
    b_title: str | None,
    b_abstract: str | None,
    b_body: str,
    answer_type: str,
    client: openai.AsyncOpenAI,
    model: str,
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> MultiHopQA | None:
    """2-hop shim: wrap the draft's single edge and ground it via `ground_chain`."""
    edge = Edge(draft.intermediate_paper_id, draft.gold_paper_id, draft.evidence_a, draft.b_label)
    qa = await ground_chain(
        draft.a_cue, [edge], b_title=b_title, b_abstract=b_abstract, b_body=b_body,
        answer_type=answer_type, client=client, model=model,
        max_body_chars=max_body_chars, temperature=temperature,
    )
    return None if qa is None else dataclasses.replace(qa, anchor=draft.a_anchor)
