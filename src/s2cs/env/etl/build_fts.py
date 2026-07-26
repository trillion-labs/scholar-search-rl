import dataclasses
import logging
from pathlib import Path

import duckdb
import tyro

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildFtsArgs:
    raw_dir: Path = Path("data/raw/s2orc")
    out_path: Path = Path("data/processed/papers.duckdb")
    papers_meta_path: Path = Path("data/processed/papers_meta.parquet")
    threads: int = 16


def main(args: BuildFtsArgs) -> None:
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        args.out_path.unlink()

    con = duckdb.connect(str(args.out_path))
    con.execute(f"SET threads = {args.threads}")
    con.execute("INSTALL fts; LOAD fts;")

    log.info("materializing papers_text from %s/*.parquet", args.raw_dir)
    con.execute(f"""
        CREATE TABLE papers_text AS
        SELECT
            corpus_id,
            parsed_title AS title,
            abstract,
            summary,
            COALESCE(
                NULLIF(text, ''),
                array_to_string(list_transform(sections, x -> x.content), '\n\n')
            ) AS body
        FROM read_parquet('{args.raw_dir}/*.parquet', union_by_name=true)
    """)
    n_text = con.execute("SELECT COUNT(*) FROM papers_text").fetchone()[0]
    log.info("papers_text rows: %s", f"{n_text:,}")

    log.info("materializing papers_sections (flattened)")
    con.execute(f"""
        CREATE TABLE papers_sections AS
        SELECT
            corpus_id,
            CAST(idx AS INTEGER) - 1 AS section_idx,
            sec.title AS section_title,
            sec.content AS content
        FROM (
            SELECT corpus_id, UNNEST(sections, recursive := false) AS sec, generate_subscripts(sections, 1) AS idx
            FROM read_parquet('{args.raw_dir}/*.parquet', union_by_name=true)
            WHERE sections IS NOT NULL AND len(sections) > 0
        )
    """)
    n_sec = con.execute("SELECT COUNT(*) FROM papers_sections").fetchone()[0]
    log.info("papers_sections rows: %s", f"{n_sec:,}")

    if args.papers_meta_path.exists():
        log.info("materializing papers_meta from %s", args.papers_meta_path)
        con.execute(f"""
            CREATE TABLE papers_meta AS
            SELECT * FROM read_parquet('{args.papers_meta_path}')
        """)
    else:
        log.warning("papers_meta.parquet not found at %s; skipping papers_meta table",
                    args.papers_meta_path)

    log.info("indexing papers_sections by corpus_id")
    con.execute("CREATE INDEX papers_sections_corpus_id_idx ON papers_sections(corpus_id)")

    log.info("building FTS5 index on (title, abstract, summary)")
    con.execute("PRAGMA create_fts_index('papers_text', 'corpus_id', 'title', 'abstract', 'summary')")
    log.info("done -> %s", args.out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BuildFtsArgs))
