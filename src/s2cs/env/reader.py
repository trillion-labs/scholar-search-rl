import dataclasses
import logging
import re
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SectionRef:
    section_idx: int
    title: str | None
    n_lines: int


@dataclasses.dataclass(frozen=True)
class PaperMeta:
    corpus_id: int
    title: str | None
    abstract: str | None
    summary: str | None
    year: int | None
    venue: str | None
    citation_count: int
    classification: str | None
    reference_count: int
    sections: list[SectionRef]


@dataclasses.dataclass(frozen=True)
class ReadResult:
    paper_id: int
    section_idx: int | None
    offset: int
    limit: int
    total_lines: int
    text: str


@dataclasses.dataclass(frozen=True)
class Match:
    line_number: int
    line: str
    context_before: list[str]
    context_after: list[str]


class PaperReader:
    def __init__(self, duckdb_path: Path) -> None:
        self.con = duckdb.connect(str(duckdb_path), read_only=True)
        self._has_meta = self._table_exists("papers_meta")
        self._has_sections = self._table_exists("papers_sections")

    def _table_exists(self, name: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        return row is not None

    def lexical_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        rows = self.con.execute("""
            SELECT corpus_id, score FROM (
                SELECT corpus_id, fts_main_papers_text.match_bm25(corpus_id, ?) AS score
                FROM papers_text
            ) WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?
        """, [query, limit]).fetchall()
        return [(int(cid), float(score)) for cid, score in rows]

    def meta(self, paper_id: int) -> PaperMeta | None:
        row = self.con.execute(
            "SELECT corpus_id, title, abstract, summary FROM papers_text WHERE corpus_id = ?",
            [paper_id],
        ).fetchone()
        if not row:
            return None

        year = venue = classification = None
        citation_count = reference_count = 0
        if self._has_meta:
            meta_row = self.con.execute("""
                SELECT year, venue, citationcount, classification
                FROM papers_meta WHERE corpus_id = ?
            """, [paper_id]).fetchone()
            if meta_row:
                year, venue, citation_count, classification = meta_row
                citation_count = int(citation_count or 0)

        sections: list[SectionRef] = []
        if self._has_sections:
            sec_rows = self.con.execute("""
                SELECT section_idx, section_title, content
                FROM papers_sections WHERE corpus_id = ?
                ORDER BY section_idx
            """, [paper_id]).fetchall()
            for sidx, stitle, scontent in sec_rows:
                n_lines = len((scontent or "").splitlines())
                sections.append(SectionRef(section_idx=int(sidx), title=stitle, n_lines=n_lines))
            reference_count = sum(1 for s in sections if (s.title or "").lower().startswith("ref"))

        return PaperMeta(
            corpus_id=int(row[0]),
            title=row[1],
            abstract=row[2],
            summary=row[3],
            year=int(year) if year is not None else None,
            venue=venue,
            citation_count=int(citation_count),
            classification=classification,
            reference_count=int(reference_count),
            sections=sections,
        )

    def body_lines(self, paper_id: int) -> list[str]:
        row = self.con.execute(
            "SELECT body FROM papers_text WHERE corpus_id = ?", [paper_id]
        ).fetchone()
        if not row or row[0] is None:
            return []
        return row[0].splitlines()

    def section_lines(self, paper_id: int, section_idx: int) -> list[str]:
        if not self._has_sections:
            return []
        row = self.con.execute("""
            SELECT content FROM papers_sections WHERE corpus_id = ? AND section_idx = ?
        """, [paper_id, section_idx]).fetchone()
        if not row or row[0] is None:
            return []
        return row[0].splitlines()

    def read(
        self, paper_id: int, section_idx: int | None = None,
        offset: int = 0, limit: int = 200,
    ) -> ReadResult:
        lines = self.section_lines(paper_id, section_idx) if section_idx is not None else self.body_lines(paper_id)
        slice_ = lines[offset : offset + limit]
        return ReadResult(
            paper_id=paper_id,
            section_idx=section_idx,
            offset=offset,
            limit=limit,
            total_lines=len(lines),
            text="\n".join(slice_),
        )

    def find_in_paper(
        self, paper_id: int, pattern: str, context: int = 2, max_hits: int = 20,
    ) -> list[Match]:
        lines = self.body_lines(paper_id)
        rx = re.compile(pattern, re.IGNORECASE)
        hits: list[Match] = []
        for i, line in enumerate(lines):
            if rx.search(line):
                hits.append(Match(
                    line_number=i,
                    line=line,
                    context_before=lines[max(0, i - context) : i],
                    context_after=lines[i + 1 : i + 1 + context],
                ))
                if len(hits) >= max_hits:
                    break
        return hits
