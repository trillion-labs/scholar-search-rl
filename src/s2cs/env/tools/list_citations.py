from typing import Callable

from s2cs.env.graph import S2Graph
from s2cs.env.tools.list_references import PaperRef


def make_list_citations(graph: S2Graph) -> Callable[..., list[PaperRef]]:
    def list_citations(paper_id: int, limit: int = 20) -> list[PaperRef]:
        """List papers that cite `paper_id` (its incoming citations).

        Returns up to `limit` citing papers.
        """
        return [PaperRef(corpus_id=cid) for cid in graph.cited_by(paper_id, limit=limit)]
    return list_citations
