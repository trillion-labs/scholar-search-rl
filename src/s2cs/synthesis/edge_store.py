"""Persistent, deduplicated store of citation edges (the reusable unit of a
multi-hop chain): `from` cites `to`, with the verbatim citing sentence and the
relationship label `from` uses to point at `to`. See
docs/design/2026-06-30-multihop-nhop-edge-store-design.md."""

import dataclasses
import json
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class Edge:
    from_id: int
    to_id: int
    citing_evidence: str  # verbatim sentence in from_id's body discussing to_id
    pointer_label: str  # how from_id refers to to_id (relationship only; never to_id's name)

    def to_record(self) -> dict:
        return {"from": self.from_id, "to": self.to_id,
                "citing_evidence": self.citing_evidence, "pointer_label": self.pointer_label}

    @classmethod
    def from_record(cls, d: dict) -> "Edge":
        return cls(int(d["from"]), int(d["to"]), d["citing_evidence"], d["pointer_label"])


def load_edges(path: Path) -> dict[int, list[Edge]]:
    idx: dict[int, list[Edge]] = {}
    if not path.exists():
        return idx
    seen: set[tuple[int, int]] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = Edge.from_record(json.loads(line))
            if (e.from_id, e.to_id) in seen:
                continue
            seen.add((e.from_id, e.to_id))
            idx.setdefault(e.from_id, []).append(e)
    return idx


def append_edges(path: Path, edges: Iterable[Edge], seen: set[tuple[int, int]]) -> int:
    n = 0
    with path.open("a") as fh:
        for e in edges:
            key = (e.from_id, e.to_id)
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps(e.to_record(), ensure_ascii=False) + "\n")
            n += 1
    return n
