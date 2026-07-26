import logging
from collections import defaultdict
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


class S2Graph:
    def __init__(self, edges_path: Path) -> None:
        self.edges_path = edges_path
        self._out: dict[int, list[int]] = defaultdict(list)
        self._in: dict[int, list[int]] = defaultdict(list)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        log.info("loading citation graph from %s", self.edges_path)
        rows = duckdb.connect().execute(
            f"SELECT src, dst FROM read_parquet('{self.edges_path}')"
        ).fetchall()
        for src, dst in rows:
            self._out[int(src)].append(int(dst))
            self._in[int(dst)].append(int(src))
        self._loaded = True
        log.info("graph loaded: %d edges, %d nodes with out-edges, %d nodes with in-edges",
                 len(rows), len(self._out), len(self._in))

    def references(self, paper_id: int, limit: int | None = None) -> list[int]:
        self._load()
        out = self._out.get(paper_id, [])
        return out[:limit] if limit is not None else out

    def cited_by(self, paper_id: int, limit: int | None = None) -> list[int]:
        self._load()
        out = self._in.get(paper_id, [])
        return out[:limit] if limit is not None else out
