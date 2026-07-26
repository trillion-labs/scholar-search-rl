import asyncio

from s2cs.synthesis.chain import build_chain
from s2cs.synthesis.edge_store import Edge


def test_reuses_store_then_discovers_to_depth():
    store = {1: [Edge(1, 2, "e12", "p12")]}  # A->B already known

    async def discover(node):
        return Edge(2, 3, "e23", "p23") if node == 2 else None  # B->C discovered

    edges = asyncio.run(build_chain(1, hops=3, min_hops=2, store=store, discover=discover))
    assert [(e.from_id, e.to_id) for e in edges] == [(1, 2), (2, 3)]


def test_drops_when_below_min_hops():
    async def discover(node):
        return None

    assert asyncio.run(build_chain(1, hops=3, min_hops=2, store={}, discover=discover)) == []


def test_cycle_guard_stops_chain():
    store = {1: [Edge(1, 2, "e", "p")], 2: [Edge(2, 1, "e", "p")]}  # B points back to A

    async def discover(node):
        return None

    edges = asyncio.run(build_chain(1, hops=4, min_hops=2, store=store, discover=discover))
    assert [(e.from_id, e.to_id) for e in edges] == [(1, 2)]  # 2->1 skipped (1 in path)
