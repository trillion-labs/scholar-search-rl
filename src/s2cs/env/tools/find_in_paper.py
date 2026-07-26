from typing import Callable

from s2cs.env.reader import Match, PaperReader


def make_find_in_paper(reader: PaperReader) -> Callable[..., list[Match]]:
    def find_in_paper(
        paper_id: int, pattern: str, context: int = 2, max_hits: int = 20,
    ) -> list[Match]:
        """Search inside a single paper's body for a regex `pattern`.

        Returns up to `max_hits` matches, each with the matched line, its line
        number, and `context` surrounding lines on each side.
        """
        return reader.find_in_paper(paper_id, pattern, context=context, max_hits=max_hits)
    return find_in_paper
