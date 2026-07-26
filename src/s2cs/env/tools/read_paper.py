from typing import Callable

from s2cs.env.reader import PaperReader, ReadResult


def make_read_paper(reader: PaperReader) -> Callable[..., ReadResult]:
    def read_paper(
        paper_id: int,
        section_idx: int | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> ReadResult:
        """Read a slice of a paper's text, line-paginated.

        If `section_idx` is provided, reads lines of that specific section
        (use paper_meta to discover sections). Otherwise reads the full body.
        Returns lines [offset, offset+limit) joined by newlines, plus the
        section's total line count so you can decide whether to keep paging.
        """
        return reader.read(paper_id, section_idx=section_idx, offset=offset, limit=limit)
    return read_paper
