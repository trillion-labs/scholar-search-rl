import dataclasses
import logging
import re
from typing import Awaitable, Callable

import openai

from s2cs.agent.llm import chat_json

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SingleHopQA:
    qa_id: str
    question: str
    answer: str
    seed_paper_id: int
    evidence: str
    anchor: str       # how the question points at the paper (see ANCHOR_RULES)
    answer_type: str  # what it asks for: "value" | "abstract" | "identity"
    cited_paper_ids: tuple[int, ...] = ()  # T2c/T2d: cited papers used as the citation cue; empty otherwise

    def to_record(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "source": "single_hop",
            "question": self.question,
            "answer": self.answer,
            "seed_paper_ids": [self.seed_paper_id],
            "evidence": self.evidence,
            "anchor": self.anchor,
            "answer_type": self.answer_type,
            "cited_paper_ids": list(self.cited_paper_ids),
            "rubric_pass": None,
            "stage": None,
            "pass_at_8": None,
        }


# How the question refers to the paper (the "anchor" — L2 given→withheld). See docs/query_design.md §3.
ANCHOR_RULES = {
    "named": (  # T0
        "Name a specific entity the paper INTRODUCES (its coined system, method, model, or dataset name) "
        "so the exact paper is identifiable from the question alone. Do not quote the title verbatim or "
        "cite authors/year."
    ),
    "paraphrastic": (  # T1
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by PARAPHRASING its "
        "core contribution in plain words (recombining what it does or proposes) so the paraphrase points "
        "to this one paper."
    ),
    "detail_cue": (  # T1
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by ONE distinctive, "
        "specific REPORTED DETAIL of its own work — a precise value, quantity, dataset, finding, or "
        "experimental setup — that is near-unique to this paper, so that detail alone points to it. Anchor "
        "on the concrete reported fact, NOT on describing the method."
    ),
    # An L1-broad × L3-sharp "pointer game" rewrite of the two conjunction anchors was trialed
    # (2026-06): frame each cue as deliberately too weak alone, so only the intersection is unique.
    # A 40-QA blind review scored it ~equivalent to the form below (CLEAN 78% vs 73%; true
    # title-leak 2–5% either way), so we keep the simpler original. The variant, for reference:
    #   content_conjunction: "Pin the paper by the INTERSECTION of two clues, neither of which finds
    #     it alone: a BROAD clue — its research area/task at the level of a field that hundreds of
    #     papers share, naming the field, not what this paper did; and a SHARP clue — one concrete
    #     reported value or setting from [BODY]. Each clue alone leaves many candidates; only their
    #     overlap is unique. If the broad clue alone identifies the paper, it is describing the
    #     contribution — widen it. Not the coined name, title, or authors."
    #   context_conjunction: "Same intersection game, but one clue is metadata: a BROAD content clue
    #     (research area/task at the level of a field) combined with the paper's YEAR and/or VENUE
    #     from [METADATA], plus a SHARP reported detail from [BODY] if those still leave many
    #     candidates. Each clue alone leaves many candidates — a year or venue matches thousands of
    #     papers, a field a subfield; only their overlap pins one. If the content clue alone
    #     identifies the paper, widen it. Use only year/venue (not authors, citation count). Not the
    #     coined name, title, or authors."
    "content_conjunction": (  # T2a — research area (L1) × a specific reported detail (L3), kept disjoint from the title
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by the INTERSECTION of "
        "(a) its research area or approach (which matches many papers) AND (b) at least one specific REPORTED "
        "value, quantity, dataset, finding, or experimental setup "
        "from the body that is NOT a phrase taken from the title. Each cue alone matches many papers; "
        "together they pin exactly this one. The question MUST contain both the research-area cue and the "
        "specific reported detail."
    ),
    "context_conjunction": (  # T2b
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by a CONJUNCTION of a "
        "content cue (its research area plus a distinctive reported detail) AND a metadata constraint it "
        "satisfies — its publication YEAR and/or VENUE. The content cue plus the year/venue together pin "
        "this one paper. Use only year/venue (NOT author, NOT citation count)."
    ),
    "relational_conjunction": (  # T2c — content (L1×L3) × citation cue
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by the INTERSECTION of "
        "(a) a CONTENT cue — its research area plus one specific REPORTED value, quantity, dataset, finding, "
        "or experimental setup from the body that is NOT a phrase taken from the title — AND (b) a CITATION "
        "cue: the fact that it cites one specific paper from [CITED], referred to by that cited paper's topic "
        "or title, NEVER by its C-number (the agent cannot see the numbers). Each cue alone matches many "
        "papers; the content cue together with 'cites <that work>' pins exactly this one. The question MUST "
        "contain both the content cue and the citation cue, and you MUST report which cited paper you used in "
        "'cited_label' (its C-number)."
    ),
    "full_span": (  # T2d — content × metadata (year/venue) × citation
        "Do NOT use the paper's coined name, title, or authors. Point at the paper by the INTERSECTION of "
        "THREE cues, none unique alone: (a) a CONTENT cue — its research area plus one specific REPORTED "
        "detail from the body; (b) a METADATA cue — its publication YEAR and/or VENUE from [METADATA]; "
        "(c) a CITATION cue — that it cites one specific paper from [CITED], referred to by that work's topic "
        "or title (NEVER its C-number). A year matches thousands of papers, a venue thousands, a citation "
        "many — only all three together pin this one. The question MUST contain all three cues. You MUST "
        "report which cited paper you used in 'cited_label' (its C-number). Use only year/venue (NOT author, NOT "
        "citation count)."
    ),
}

# What the question asks for (the "answer" — reading-depth ladder). See docs/query_design.md §4.
ANSWER_RULES = {
    "value": (  # L3 — body
        "The answer is a short, closed-form fact that is part of THIS paper's own work, stated in a "
        "COMPLETE RUNNING-PROSE SENTENCE in [BODY]. Do not take it from a table, a figure, or a bare list "
        "of numbers (a caption, or a value stated in a prose sentence near an equation, is fine). It must "
        "appear in [BODY] and not be inferable from [PUBLIC] or general knowledge. 'evidence' is that full "
        "sentence, verbatim."
    ),
    "abstract": (  # mixed — abstract
        "The answer is a single short, closed-form fact stated in the ABSTRACT — the paper's task, a "
        "headline quantitative result, a dataset, or a key claim — answerable by finding the paper and "
        "reading its abstract, without opening the body. It must NOT be bibliographic metadata and NOT be "
        "inferable from general knowledge. 'evidence' is the verbatim abstract sentence supporting it."
    ),
    "identity": (  # L2 — the paper itself
        "The answer is the paper's identity — the NAME it coins for its own contribution (a method/system/"
        "dataset name) or, lacking one, its TITLE — copied exactly. The question must ASK WHICH PAPER this "
        "is (its title or coined name) — do NOT phrase it as asking for a concept, entity, or value inside "
        "the paper. The question must NEVER state the answer; it points at the paper only through the "
        "anchor above, so the answer is recoverable only by finding the paper. 'evidence' is the verbatim "
        "sentence that names the contribution (for a coined name) or the sentence carrying the anchor cue."
    ),
}

# Coherent (anchor, answer_type) cells we generate. See docs/query_design.md §5.1.
# Excluded: named×identity (degenerate — the name is the answer), paraphrastic×identity
# (leaks — paraphrasing the contribution reconstructs the name; use detail_cue instead).
VALID_CELLS = {
    ("named", "value"), ("named", "abstract"),
    ("paraphrastic", "value"), ("paraphrastic", "abstract"),
    ("detail_cue", "value"), ("detail_cue", "abstract"), ("detail_cue", "identity"),
    ("content_conjunction", "value"), ("content_conjunction", "abstract"), ("content_conjunction", "identity"),
    ("context_conjunction", "value"), ("context_conjunction", "abstract"), ("context_conjunction", "identity"),
    ("relational_conjunction", "value"), ("relational_conjunction", "abstract"), ("relational_conjunction", "identity"),
    ("full_span", "value"), ("full_span", "abstract"), ("full_span", "identity"),
}

CITATION_ANCHORS = frozenset({"relational_conjunction", "full_span"})


EXTRACT_PROMPT = """You write up to 3 exam questions for a literature-search agent, drawn from the specific paper below.

[PUBLIC] (title, abstract, summary — assume the agent can already see this)
{public}
{body_block}{meta_block}{cited_block}
How each question must refer to the paper:
- {anchor_rule}

What each question must ask for:
- {answer_rule}

General requirements:
- Each question has exactly one correct answer; no ambiguity.
- The questions must be DISTINCT — each on a different fact or aspect of the paper. If the paper yields fewer than 3 good ones, return fewer.
- The ANSWER must NOT be bibliographic metadata (publication date, venue, author names / emails / affiliations), nor a fact the paper merely cites or attributes to other / prior work.
- "evidence" is a verbatim quote from {evidence_src} supporting the answer.

Return ONLY a JSON array (no prose, no markdown) of up to 3 objects:
[{{"question": "...", "answer": "...", "evidence": "..."{cited_field}}}, ...]
If the paper yields no good question of this kind, return [].
"""


def _format_public(title: str | None, abstract: str | None, summary: str | None) -> str:
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


def _is_prose(text: str) -> bool:
    """A running-prose sentence, not a table row / figure value / bare number list:
    at least 6 whitespace tokens, most of them containing a letter."""
    toks = text.split()
    if len(toks) < 6:
        return False
    alpha = sum(1 for t in toks if any(c.isalpha() for c in t))
    return alpha / len(toks) >= 0.6


def _validate(cid: int, item: object, *, idx: int, body: str, public: str, title: str | None,
              anchor: str, answer_type: str, cited: list[dict] | None = None) -> SingleHopQA | None:
    if not isinstance(item, dict):
        return None
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    evidence = str(item.get("evidence", "")).strip()
    if not question or not answer or not evidence:
        return None

    q = _norm(question)
    # non-named anchors must not name the paper (title leak)
    if anchor != "named" and title and _norm(title) in q:
        return None

    cited_paper_ids: tuple[int, ...] = ()
    if anchor in CITATION_ANCHORS:
        cited_list = cited or []
        raw = item.get("cited_label")
        m = re.search(r"\d+", str(raw)) if raw is not None else None
        if m is None:
            return None
        pos = int(m.group()) - 1  # C1 -> index 0
        if pos < 0 or pos >= len(cited_list):
            return None
        chosen = cited_list[pos]
        # The citation cue must actually name the cited work, not dangle as a bare label.
        # Cheap proxy: at least one substantive (≥5-char) word of the cited title reappears
        # as a question token. Deliberately strict (exact-token, not topical) — it can reject
        # a heavy paraphrase; the prototype's per-cell yield tells us if that bites.
        cited_title_words = {w for w in _norm(chosen.get("title") or "").split() if len(w) >= 5}
        if cited_title_words and not (cited_title_words & set(q.split())):
            return None
        cited_paper_ids = (int(chosen["corpus_id"]),)

    if answer_type == "value":
        # body-only L3 fact from a prose sentence (no table/figure/equation/number-row golds)
        if _norm(evidence) not in _norm(body):
            return None
        if not _is_prose(evidence):
            return None
        if len(answer.split()) > 15:
            return None
    elif answer_type == "abstract":
        # closed-form fact stated in the abstract (PUBLIC)
        if _norm(evidence) not in _norm(public):
            return None
        if _norm(answer) not in _norm(public):
            return None
        if len(answer.split()) > 15:
            return None
    elif answer_type == "identity":
        # gold = coined name or title, present in the text, never leaked into the question.
        # Safe because identity only pairs with L3-pinning anchors (the cue is disjoint
        # from the name/title) — see VALID_CELLS / docs §5.1.
        haystack = f"{public}\n{body}"
        if _norm(evidence) not in _norm(haystack):
            return None
        if _norm(answer) not in _norm(haystack):
            return None
        if _norm(answer) in q:
            return None
    else:
        return None

    return SingleHopQA(
        qa_id=f"single_hop_{cid}_{anchor}_{answer_type}_{idx}",
        question=question,
        answer=answer,
        seed_paper_id=cid,
        evidence=evidence,
        anchor=anchor,
        answer_type=answer_type,
        cited_paper_ids=cited_paper_ids,
    )


def make_single_hop_synth(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    anchor: str = "named",
    answer_type: str = "value",
    max_body_chars: int | None = None,
    temperature: float = 0.7,
) -> Callable[[dict], Awaitable[list[SingleHopQA]]]:
    """Build a single-hop QA generator for one (anchor, answer_type) format cell.

    `anchor` ∈ ANCHOR_RULES (named / paraphrastic / detail_cue / content_conjunction /
    context_conjunction / relational_conjunction / full_span); `answer_type` ∈ ANSWER_RULES
    (value / abstract / identity). The two citation anchors require `paper["cited"]`.
    Only the coherent cells in VALID_CELLS are allowed — `named×identity` and
    `paraphrastic×identity` are rejected (see docs/query_design.md §5.1).

    The returned coroutine takes one paper row (`corpus_id, title, abstract, summary,
    body`) and returns up to 3 distinct validated QA (fewer, or [], if the paper yields
    fewer of this kind). Corpus-level conjunction uniqueness is left to the filter stage.
    """
    if anchor not in ANCHOR_RULES:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {sorted(ANCHOR_RULES)}")
    if answer_type not in ANSWER_RULES:
        raise ValueError(f"unknown answer_type {answer_type!r}; expected one of {sorted(ANSWER_RULES)}")
    if (anchor, answer_type) not in VALID_CELLS:
        raise ValueError(
            f"({anchor}, {answer_type}) is not a coherent cell — "
            "named×identity is degenerate, paraphrastic×identity leaks; see docs/query_design.md §5.1"
        )

    evidence_src = {"value": "[BODY]", "abstract": "[PUBLIC]"}.get(answer_type, "[PUBLIC] or [BODY]")

    async def single_hop_synth(paper: dict) -> list[SingleHopQA]:
        cid = int(paper["corpus_id"])
        title = paper.get("title")
        public = _format_public(title, paper.get("abstract"), paper.get("summary"))
        if not public:
            return []
        # Always send the body: value/abstract anchors and the detail_cue/conjunction
        # cues all draw on body-level detail, and it lets the question describe the paper
        # without echoing the (often descriptive) title.
        body = paper.get("body") or ""
        if max_body_chars is not None:
            body = body[:max_body_chars]
        if not body:
            return []
        if anchor in CITATION_ANCHORS:
            cited = paper.get("cited") or []
            if not cited:
                return []
        else:
            cited = []
        # context_conjunction (T2b) and full_span (T2d) need a venue for the metadata cue
        # (year alone is universal -> no narrowing). Skip venue-less papers.
        if anchor in ("context_conjunction", "full_span") and not paper.get("venue"):
            return []
        meta_block = ""
        if anchor in ("context_conjunction", "full_span"):
            bits = []
            if paper.get("year"):
                bits.append(f"year={int(paper['year'])}")
            bits.append(f"venue={str(paper['venue'])!r}")
            meta_block = "\n[METADATA] (use these EXACT values for the year/venue cue)\n" + ", ".join(bits) + "\n"
        cited_block = ""
        if cited:
            lines = "\n".join(
                f"  C{i+1}. ({int(c['year']) if c.get('year') else 'n.d.'}) {c['title']}"
                for i, c in enumerate(cited)
            )
            cited_block = (
                "\n[CITED] (papers THIS paper cites - refer to one by its topic/title, never by a C-number)\n"
                + lines + "\n"
            )
        prompt = EXTRACT_PROMPT.format(
            public=public,
            body_block=f"\n[BODY] (the paper's full text)\n{body}\n",
            meta_block=meta_block,
            cited_block=cited_block,
            cited_field=', "cited_label": "<the C-label of the [CITED] paper you used, e.g. C2>"' if cited else "",
            evidence_src=evidence_src,
            anchor_rule=ANCHOR_RULES[anchor],
            answer_rule=ANSWER_RULES[answer_type],
        )
        payload = await chat_json(client, model, [{"role": "user", "content": prompt}], temperature=temperature)
        items = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []

        out: list[SingleHopQA] = []
        seen: set[str] = set()
        for item in items[:3]:
            qa = _validate(cid, item, idx=len(out), body=body, public=public, title=title,
                           anchor=anchor, answer_type=answer_type, cited=cited)
            if qa is not None and _norm(qa.question) not in seen:
                seen.add(_norm(qa.question))
                out.append(qa)
        return out

    return single_hop_synth
