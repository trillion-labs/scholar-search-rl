import dataclasses
from typing import Callable

from pymilvus import AnnSearchRequest, Collection, RRFRanker

DEFAULT_HYBRID_POOL = 50
# HNSW search depth. Milvus requires the per-search `limit` <= `ef`, so the
# retrieval pool is capped at EF: asking for more just returns the top EF
# (fast, fixed beam) rather than widening ef to match (accurate but slow — a
# k=250 search at ef=250+ blew past the rollout timeout). For paper_set the
# gold set is small and reward is recall@est, so a 128-result cap per call is
# ample; the agent accumulates coverage across multiple searches.
EF = 128


@dataclasses.dataclass(frozen=True)
class PaperHit:
    corpus_id: int
    year: int | None
    venue: str | None
    citation_count: int
    score: float


def make_search_papers(
    collection: Collection,
    encode_hybrid: Callable[[str], tuple[list[float], dict[int, float]]],
) -> Callable[..., list[PaperHit]]:
    def search_papers(
        query: str,
        limit: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
        venue: str | None = None,
        mask_paper_ids: set[int] | None = None,
    ) -> list[PaperHit]:
        """Search papers by natural-language query.

        Returns up to `limit` papers ranked by hybrid (BGE-M3 dense + learned
        sparse) similarity. Pass year_min / year_max / venue to narrow the
        candidate pool; omit them to search the whole corpus.
        """
        dense_vec, sparse_vec = encode_hybrid(query)
        clauses: list[str] = []
        if year_min is not None:
            clauses.append(f"year >= {year_min}")
        if year_max is not None:
            clauses.append(f"year <= {year_max}")
        if venue:
            clauses.append(f'venue == "{venue}"')
        expr = " and ".join(clauses) if clauses else None

        pool = max(limit * 5, DEFAULT_HYBRID_POOL)
        if mask_paper_ids:
            pool += len(mask_paper_ids)
        pool = min(pool, EF)

        fields = ["corpus_id", "year", "venue", "citationcount"]
        if sparse_vec:
            dense_req = AnnSearchRequest(
                data=[dense_vec],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"ef": EF}},
                limit=pool,
                expr=expr,
            )
            sparse_req = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_embedding",
                param={"metric_type": "IP", "params": {}},
                limit=pool,
                expr=expr,
            )
            res = collection.hybrid_search(
                [dense_req, sparse_req],
                rerank=RRFRanker(60),
                limit=pool,
                output_fields=fields,
            )
        else:
            # No learned-sparse from the encoder (e.g. an OpenAI /embeddings backend):
            # fall back to a plain dense search instead of a degenerate hybrid.
            res = collection.search(
                data=[dense_vec],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"ef": EF}},
                limit=pool,
                expr=expr,
                output_fields=fields,
            )

        out: list[PaperHit] = []
        for hit in res[0]:
            cid = int(hit.entity.get("corpus_id"))
            if mask_paper_ids and cid in mask_paper_ids:
                continue
            out.append(
                PaperHit(
                    corpus_id=cid,
                    year=hit.entity.get("year"),
                    venue=hit.entity.get("venue") or None,
                    citation_count=hit.entity.get("citationcount") or 0,
                    score=float(hit.distance),
                )
            )
            if len(out) >= limit:
                break
        return out

    return search_papers
