from typing import Callable

from s2cs.env.reader import PaperMeta, PaperReader


def make_paper_info(reader: PaperReader) -> Callable[..., PaperMeta | None]:
    def paper_info(paper_id: int) -> PaperMeta | None:
        """Look up a paper's metadata: title, abstract, summary, year, venue,
        citation count, classification, and the list of section titles with
        line counts for navigation.

        Returns None if the paper is not in the local corpus. This is the cheap
        first step you should take when you have a paper_id and need to decide
        whether to read further.
        """
        return reader.meta(paper_id)
    return paper_info
