<p align="center">
  <img src="assets/s3-logo.png" alt="Simulated Scholar Search (S3) logo" width="240">
</p>

<h1 align="center">Simulated Scholar Search (S3)</h1>

<p align="center">
  A local scientific-literature environment for training and evaluating search agents.
</p>

<p align="center">
  <a href="https://github.com/trillion-labs/scholar-search-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4f46e5.svg" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900.svg?logo=nvidia&logoColor=white" alt="CUDA 12.8">
  <img src="https://img.shields.io/badge/Trainer-verl-7C3AED.svg" alt="verl trainer">
</p>

S3 is a local scientific-literature search environment for studying whether
search behavior learned through reinforcement learning can transfer to new
corpora and search tools.

Instead of calling an external search API during every rollout, S3 trains
agents against a controlled corpus of approximately 1.12 million computer
science papers. The environment rewards finding the correct answer while
requiring realistic search behavior: refining queries, comparing similar
papers, reading supporting passages, and following citation links.

This repository contains the environment, data synthesis, evaluation adapters,
and verl-based trainer integration.

## What is released

- Hybrid paper and passage retrieval with Milvus, DuckDB, and BGE-M3
- Citation and reverse-citation graph navigation
- ReAct agent loop with nine scientific-literature tools
- Single-hop, multi-hop, and paper-set question synthesis
- Evaluation adapters for literature and general-search benchmarks
- verl training data, tool, reward, and serving integration
- Unit tests and a fully resolved Python dependency lockfile

The Python package is named `s2cs`; the project and environment are referred to
as **Simulated Scholar Search (S3)**.

## Repository layout

```text
src/s2cs/env/         Retrieval environment, corpus ETL, and paper tools
src/s2cs/agent/       Agent loop, policy, judging, and trajectories
src/s2cs/synthesis/   Question synthesis, filtering, and difficulty grading
src/s2cs/eval/        Literature-search and web-search evaluation adapters
src/s2cs/trainer/     verl data, tools, rewards, and integration patches
configs/trainer/      Trainer tool configuration
tests/                Mock-based unit tests
```

## Setup

The resolved environment targets Python 3.12 on Linux with CUDA 12.8.
[uv](https://docs.astral.sh/uv/) is used for dependency management.

For code inspection and non-trainer tests:

```bash
uv sync --group dev --group eval
uv run pytest tests --ignore=tests/trainer
```

For the trainer dependency set:

```bash
uv sync --group dev --group trainer
```

Copy [`.env.example`](.env.example) to `.env` and adjust the paths and service
endpoints for your environment. No credentials are committed.

## Corpus entry point

The corpus downloader is exposed as a Python module:

```bash
uv run python -m s2cs.env.etl.download --help
uv run python -m s2cs.env.etl.download --revision <dataset-revision>
```

Building the full environment additionally requires paper and passage
embeddings, DuckDB indexes, a citation-edge store, and Milvus ingestion. The
corpus and generated indexes are large research artifacts and are not stored in
Git.

## Research scope

The release is intended for studying:

- simulated-to-real transfer of search behavior;
- curriculum and question design for search agents;
- scientific-literature retrieval with citation navigation;
- outcome-based rewards for multi-turn tool use.

It is not a hosted search product or a drop-in Semantic Scholar replacement.

## Acknowledgements

This codebase builds on open-source research software. The training integration
uses [verl](https://github.com/volcengine/verl); integration patches are
included under `src/s2cs/trainer/patches/`.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [Apache License 2.0](LICENSE).
