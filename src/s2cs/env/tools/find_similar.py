from typing import Callable, Literal

from pymilvus import Collection

from s2cs.env.graph import S2Graph
from s2cs.env.tools.search_papers import EF, PaperHit


SimilarKind = Literal["semantic", "co_citation", "bibliographic_coupling"]


def _co_citation_neighbors(graph: S2Graph, paper_id: int, limit: int) -> list[tuple[int, int]]:
    citers = set(graph.cited_by(paper_id))
    counts: dict[int, int] = {}
    for c in citers:
        for cited in graph.references(c):
            if cited == paper_id:
                continue
            counts[cited] = counts.get(cited, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def _bibliographic_coupling_neighbors(graph: S2Graph, paper_id: int, limit: int) -> list[tuple[int, int]]:
    refs = set(graph.references(paper_id))
    counts: dict[int, int] = {}
    for r in refs:
        for citer in graph.cited_by(r):
            if citer == paper_id:
                continue
            counts[citer] = counts.get(citer, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def _semantic_neighbors(collection: Collection, paper_id: int, limit: int) -> list[PaperHit]:
    rows = collection.query(expr=f"corpus_id == {int(paper_id)}", output_fields=["embedding"])
    if not rows:
        return []
    vec = rows[0]["embedding"]
    res = collection.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": EF}},
        limit=min(limit + 1, EF),
        output_fields=["corpus_id", "year", "venue", "citationcount"],
    )
    out: list[PaperHit] = []
    for hit in res[0]:
        cid = int(hit.entity.get("corpus_id"))
        if cid == int(paper_id):
            continue
        out.append(PaperHit(
            corpus_id=cid,
            year=hit.entity.get("year"), venue=hit.entity.get("venue") or None,
            citation_count=hit.entity.get("citationcount") or 0,
            score=float(hit.distance),
        ))
        if len(out) >= limit:
            break
    return out


def make_find_similar(collection: Collection, graph: S2Graph) -> Callable[..., list[PaperHit]]:
    def find_similar(
        paper_id: int, kind: SimilarKind = "semantic", limit: int = 10,
    ) -> list[PaperHit]:
        """Find papers similar to `paper_id`. `kind` chooses the relation:

        - "semantic": nearest neighbours in the paper-level embedding space —
          topical similarity that does not require any citation link.
        - "co_citation": papers most often cited together with this paper —
          they appear together in other papers' reference lists.
        - "bibliographic_coupling": papers that share many references with
          this paper — they read from the same prior literature.

        Use semantic for lateral exploration. Use the graph-based variants
        when you want similarity grounded in actual citation behaviour.
        """
        if kind == "semantic":
            return _semantic_neighbors(collection, paper_id, limit)
        ranked = (
            _co_citation_neighbors(graph, paper_id, limit) if kind == "co_citation"
            else _bibliographic_coupling_neighbors(graph, paper_id, limit)
        )
        return [
            PaperHit(corpus_id=cid, year=None, venue=None,
                     citation_count=0, score=float(score))
            for cid, score in ranked
        ]
    return find_similar
