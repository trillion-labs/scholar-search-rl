import dataclasses
from typing import Callable

from s2cs.env.graph import S2Graph


@dataclasses.dataclass(frozen=True)
class PaperRef:
    corpus_id: int


def make_list_references(graph: S2Graph) -> Callable[..., list[PaperRef]]:
    def list_references(paper_id: int, limit: int | None = None) -> list[PaperRef]:
        """List papers that `paper_id` cites (its outgoing citations).

        Only papers present in the local corpus appear here; citations to
        external papers are silently omitted.
        """
        return [PaperRef(corpus_id=cid) for cid in graph.references(paper_id, limit=limit)]
    return list_references
