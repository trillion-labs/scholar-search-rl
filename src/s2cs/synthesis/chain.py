"""Grow a citation chain from a start node, reusing stored edges and discovering
the tail edge only when missing. See the N-hop edge-store design doc."""

import logging
from typing import Awaitable, Callable

from s2cs.synthesis.edge_store import Edge

log = logging.getLogger(__name__)


async def build_chain(
    start_id: int,
    *,
    hops: int,
    min_hops: int,
    store: dict[int, list[Edge]],
    discover: Callable[[int], Awaitable[Edge | None]],
) -> list[Edge]:
    """Return an ordered edge list of length up to `hops - 1` (>= `min_hops - 1`, else []).
    At each step reuse a stored out-edge from the tail to a fresh node, else `discover` one.
    Guards against cycles (never revisit a node already on the path)."""
    path_nodes = {start_id}
    node = start_id
    edges: list[Edge] = []
    for _ in range(hops - 1):
        nxt = None
        for e in store.get(node, []):  # reuse a stored out-edge to a fresh node
            if e.to_id not in path_nodes:
                nxt = e
                break
        if nxt is None:  # else discover one
            cand = await discover(node)
            if cand is not None and cand.to_id not in path_nodes:
                nxt = cand
        if nxt is None:
            break
        edges.append(nxt)
        path_nodes.add(nxt.to_id)
        node = nxt.to_id
    return edges if len(edges) >= min_hops - 1 else []
