import dataclasses
from typing import Callable

from pymilvus import Collection

from s2cs.env.tools.search_papers import EF


@dataclasses.dataclass(frozen=True)
class SnippetHit:
    chunk_id: int
    paper_corpus_id: int
    section_idx: int
    score: float


def make_search_snippets(
    collection: Collection,
    encode_dense: Callable[[str], list[float]],
) -> Callable[..., list[SnippetHit]]:
    def search_snippets(
        query: str,
        limit: int = 10,
        paper_ids: list[int] | None = None,
        mask_paper_ids: set[int] | None = None,
    ) -> list[SnippetHit]:
        """Search text snippets across papers by natural-language query.

        Returns up to `limit` snippets ranked by similarity. Pass `paper_ids`
        to restrict the search to a specific set of papers (useful after
        narrowing down candidate papers with search_papers).
        """
        clauses: list[str] = []
        if paper_ids:
            in_list = ",".join(str(int(p)) for p in paper_ids)
            clauses.append(f"paper_corpus_id in [{in_list}]")
        topk = limit if not mask_paper_ids else limit + 4 * len(mask_paper_ids)
        topk = min(topk, EF)
        res = collection.search(
            data=[encode_dense(query)],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": EF}},
            limit=topk,
            expr=" and ".join(clauses) if clauses else None,
            output_fields=["chunk_id", "paper_corpus_id", "section_idx"],
        )
        out: list[SnippetHit] = []
        for hit in res[0]:
            pid = int(hit.entity.get("paper_corpus_id"))
            if mask_paper_ids and pid in mask_paper_ids:
                continue
            out.append(SnippetHit(
                chunk_id=int(hit.entity.get("chunk_id")),
                paper_corpus_id=pid,
                section_idx=int(hit.entity.get("section_idx")),
                score=float(hit.distance),
            ))
            if len(out) >= limit:
                break
        return out
    return search_snippets
