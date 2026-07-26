from typing import Callable

from pymilvus import Collection

from s2cs.env.encoder import BatchedEncoder
from s2cs.env.graph import S2Graph
from s2cs.env.reader import PaperReader
from s2cs.env.tools.find_in_paper import make_find_in_paper
from s2cs.env.tools.find_similar import make_find_similar
from s2cs.env.tools.list_citations import make_list_citations
from s2cs.env.tools.list_references import make_list_references
from s2cs.env.tools.paper_info import make_paper_info
from s2cs.env.tools.read_paper import make_read_paper
from s2cs.env.tools.search_papers import make_search_papers
from s2cs.env.tools.search_snippets import make_search_snippets
from s2cs.env.tools.submit_answer import make_submit_answer


def build_registry(
    *,
    papers: Collection,
    chunks: Collection,
    graph: S2Graph,
    reader: PaperReader,
    encoder: BatchedEncoder,
) -> dict[str, Callable]:
    return {
        "search_papers":   make_search_papers(papers, encoder.encode_hybrid),
        "search_snippets": make_search_snippets(chunks, encoder.encode_dense),
        "read_paper":      make_read_paper(reader),
        "find_in_paper":   make_find_in_paper(reader),
        "list_references": make_list_references(graph),
        "list_citations":  make_list_citations(graph),
        "find_similar":    make_find_similar(papers, graph),
        "paper_info":      make_paper_info(reader),
        "submit_answer":   make_submit_answer(),
    }


def compose(registry: dict[str, Callable], names: list[str]) -> dict[str, Callable]:
    missing = [n for n in names if n not in registry]
    if missing:
        raise KeyError(f"unknown tools: {missing}")
    return {n: registry[n] for n in names}
