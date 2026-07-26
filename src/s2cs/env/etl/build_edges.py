import dataclasses
import json
import logging
from pathlib import Path

import duckdb
import tyro

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildEdgesArgs:
    raw_dir: Path = Path("data/raw/s2orc")
    out_dir: Path = Path("data/processed")
    threads: int = 32
    memory_limit: str = "120GB"


SETUP_SQL = r"""
CREATE OR REPLACE MACRO norm_doi(s) AS
    NULLIF(
        LOWER(TRIM(REGEXP_REPLACE(
            REGEXP_REPLACE(s, '^\s*(https?://(dx\.)?doi\.org/|doi:\s*)', '', 'i'),
            '\s+', ''
        ))),
        ''
    );
"""


def main(args: BuildEdgesArgs) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    glob = str(args.raw_dir / "*.parquet")

    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(SETUP_SQL)
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{glob}', union_by_name=true)")

    con.execute("""
        CREATE TABLE papers AS
        SELECT
            corpus_id,
            parsed_title,
            abstract,
            summary,
            classification,
            year,
            venue,
            citationcount,
            referencecount,
            COALESCE(
                norm_doi(DOI),
                norm_doi(parsed_external_ids.doi),
                norm_doi(metadata_externalids.DOI)
            ) AS norm_doi,
            LENGTH(text) AS text_chars,
            CAST(LENGTH(text) AS BIGINT) / 4 AS text_tokens_est,
            (sections IS NOT NULL AND len(sections) > 0) AS has_sections,
            (summary IS NOT NULL AND summary != '') AS has_summary,
            (abstract IS NOT NULL AND abstract != '') AS has_abstract,
            (parsed_title IS NOT NULL AND parsed_title != '') AS has_title
        FROM raw
    """)
    n_papers = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    con.execute("""
        CREATE TABLE doi_to_corpus AS
        SELECT norm_doi, MIN(corpus_id) AS corpus_id
        FROM papers
        WHERE norm_doi IS NOT NULL
        GROUP BY norm_doi
    """)
    n_doi = con.execute("SELECT COUNT(*) FROM doi_to_corpus").fetchone()[0]

    con.execute("""
        CREATE TABLE refs AS
        SELECT
            corpus_id AS citing_corpus_id,
            norm_doi(ref.doi) AS ref_norm_doi
        FROM (
            SELECT corpus_id, unnest("references") AS ref FROM raw
        )
    """)
    n_refs_total = con.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    n_refs_with_doi = con.execute("SELECT COUNT(*) FROM refs WHERE ref_norm_doi IS NOT NULL").fetchone()[0]

    con.execute("""
        CREATE TABLE edges AS
        SELECT DISTINCT
            r.citing_corpus_id AS src,
            d.corpus_id        AS dst
        FROM refs r
        JOIN doi_to_corpus d ON r.ref_norm_doi = d.norm_doi
        WHERE r.citing_corpus_id != d.corpus_id
    """)
    n_edges = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_src = con.execute("SELECT COUNT(DISTINCT src) FROM edges").fetchone()[0]
    n_dst = con.execute("SELECT COUNT(DISTINCT dst) FROM edges").fetchone()[0]

    edges_path = args.out_dir / "edges.parquet"
    con.execute(f"COPY (SELECT * FROM edges ORDER BY src, dst) TO '{edges_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    meta_path = args.out_dir / "papers_meta.parquet"
    con.execute(f"""
        COPY (
            SELECT
                corpus_id, norm_doi, year, venue, citationcount, classification,
                has_title, has_abstract, has_summary, has_sections,
                text_tokens_est
            FROM papers
        ) TO '{meta_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    tok = con.execute("""
        SELECT
            AVG(text_tokens_est) AS avg,
            APPROX_QUANTILE(text_tokens_est, 0.5) AS p50,
            APPROX_QUANTILE(text_tokens_est, 0.95) AS p95,
            APPROX_QUANTILE(text_tokens_est, 0.99) AS p99,
            MAX(text_tokens_est) AS max
        FROM papers WHERE text_tokens_est IS NOT NULL
    """).fetchone()

    cov = con.execute("""
        SELECT
            AVG(CASE WHEN has_title    THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN has_abstract THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN has_summary  THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN has_sections THEN 1.0 ELSE 0.0 END)
        FROM papers
    """).fetchone()

    classes = con.execute("""
        SELECT classification, COUNT(*) AS n FROM papers
        WHERE classification IS NOT NULL
        GROUP BY classification ORDER BY n DESC LIMIT 20
    """).fetchall()

    years = con.execute("""
        SELECT year, COUNT(*) AS n FROM papers
        WHERE year IS NOT NULL
        GROUP BY year ORDER BY year
    """).fetchall()

    in_degree = con.execute("""
        SELECT
            AVG(deg) AS avg,
            APPROX_QUANTILE(deg, 0.5) AS p50,
            APPROX_QUANTILE(deg, 0.95) AS p95,
            APPROX_QUANTILE(deg, 0.99) AS p99,
            MAX(deg) AS max
        FROM (SELECT COUNT(*) AS deg FROM edges GROUP BY dst)
    """).fetchone()

    out_degree = con.execute("""
        SELECT
            AVG(deg) AS avg,
            APPROX_QUANTILE(deg, 0.5) AS p50,
            APPROX_QUANTILE(deg, 0.95) AS p95,
            APPROX_QUANTILE(deg, 0.99) AS p99,
            MAX(deg) AS max
        FROM (SELECT COUNT(*) AS deg FROM edges GROUP BY src)
    """).fetchone()

    stats = {
        "n_papers": int(n_papers),
        "n_papers_with_doi": int(n_doi),
        "doi_coverage": round(n_doi / max(n_papers, 1), 4),
        "n_refs_total": int(n_refs_total),
        "n_refs_with_doi": int(n_refs_with_doi),
        "ref_doi_coverage": round(n_refs_with_doi / max(n_refs_total, 1), 4),
        "avg_refs_per_paper": round(n_refs_total / max(n_papers, 1), 2),
        "n_edges": int(n_edges),
        "n_unique_src": int(n_src),
        "n_unique_dst": int(n_dst),
        "avg_resolved_edges_per_paper": round(n_edges / max(n_papers, 1), 2),
        "text_tokens_est": {
            "avg": round(float(tok[0] or 0), 1),
            "p50": int(tok[1] or 0),
            "p95": int(tok[2] or 0),
            "p99": int(tok[3] or 0),
            "max": int(tok[4] or 0),
        },
        "coverage": {
            "title":    round(float(cov[0] or 0), 4),
            "abstract": round(float(cov[1] or 0), 4),
            "summary":  round(float(cov[2] or 0), 4),
            "sections": round(float(cov[3] or 0), 4),
        },
        "in_degree": {
            "avg": round(float(in_degree[0] or 0), 1),
            "p50": int(in_degree[1] or 0),
            "p95": int(in_degree[2] or 0),
            "p99": int(in_degree[3] or 0),
            "max": int(in_degree[4] or 0),
        },
        "out_degree": {
            "avg": round(float(out_degree[0] or 0), 1),
            "p50": int(out_degree[1] or 0),
            "p95": int(out_degree[2] or 0),
            "p99": int(out_degree[3] or 0),
            "max": int(out_degree[4] or 0),
        },
        "classification_top20": [{"label": c, "n": int(n)} for c, n in classes],
        "year_distribution": [{"year": int(y), "n": int(n)} for y, n in years],
    }

    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    log.info("papers=%s with_doi=%s edges=%s", f"{n_papers:,}", f"{n_doi:,}", f"{n_edges:,}")
    log.info("edges        -> %s", edges_path)
    log.info("papers_meta  -> %s", meta_path)
    log.info("stats        -> %s", stats_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(BuildEdgesArgs))
