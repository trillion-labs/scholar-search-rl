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

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#system-overview">System overview</a> ·
  <a href="#repository-map">Repository map</a> ·
  <a href="#citation">Citation</a>
</p>

S3 is a controlled scientific-literature search environment for reinforcement
learning. Agents search a local corpus, inspect papers, follow citations, and
submit answers through a nine-tool interface—without calling a live search API
during every rollout.

| Corpus | Retrieval | Agent | Training |
| --- | --- | --- | --- |
| ~1.12M computer-science papers | Milvus + DuckDB + BGE-M3 | Multi-turn ReAct, 9 tools | verl integration |

The main research question is whether search behavior learned in a reproducible
local environment transfers to new corpora and search surfaces.

## Start here

| If you want to… | Start with |
| --- | --- |
| Understand the search environment | [`src/s2cs/env`](src/s2cs/env) and the [system overview](#system-overview) |
| Study the agent loop and tool calls | [`src/s2cs/agent`](src/s2cs/agent) |
| Generate research questions | [`src/s2cs/synthesis`](src/s2cs/synthesis) |
| Connect S3 to verl | [`src/s2cs/trainer`](src/s2cs/trainer) and [`configs/trainer`](configs/trainer) |
| Run evaluation adapters | [`src/s2cs/eval`](src/s2cs/eval) |
| Inspect behavior without services | [`tests`](tests) |

## System overview

```text
questions
   │
   ▼
ReAct agent ──────── search / read / navigate / answer
   │                                │
   │                                ▼
   └───────────────────── nine literature tools
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                      Milvus     DuckDB    citation graph
                         │
                         ▼
              trajectory + outcome reward
                         │
                         ▼
                   verl training
```

The environment supports hybrid paper and passage retrieval, full-text reading,
forward and reverse citation traversal, and explicit answer submission. The
synthesis modules create single-hop, multi-hop, and paper-set questions; the
evaluation modules adapt the same agent to literature and general-search
benchmarks.

## Quick start

The resolved environment targets Python 3.12 on Linux with CUDA 12.8.
[uv](https://docs.astral.sh/uv/) is used for dependency management.

```bash
git clone https://github.com/trillion-labs/scholar-search-rl.git
cd scholar-search-rl

cp .env.example .env
uv sync --group dev --group eval
uv run pytest tests --ignore=tests/trainer
```

The tests use mocks for external services, so they are the quickest way to
inspect tool and agent behavior before building the corpus.

### Trainer environment

```bash
uv sync --group dev --group trainer
VERL_SRC=/path/to/compatible/verl src/s2cs/trainer/vendor_setup.sh
```

`vendor_setup.sh` copies the supplied verl tree into a gitignored local
directory, applies the integration patches, and installs it as an editable
package.

### Working with coding agents

Point the agent to the [repository map](#repository-map) and `.env.example`
first. On a non-CUDA machine, use read-only or dependency-light checks instead
of rewriting the Linux lockfile:

```bash
uv lock --check
python -m compileall -q src tests
```

## Build the corpus

The base corpus is
[`AlgorithmicResearchGroup/s2orc-cs-enriched`](https://huggingface.co/datasets/AlgorithmicResearchGroup/s2orc-cs-enriched).
Start with the downloader:

```bash
uv run python -m s2cs.env.etl.download --help
uv run python -m s2cs.env.etl.download --revision <dataset-revision>
```

The complete environment also requires paper and passage embeddings, DuckDB
indexes, a citation-edge store, and Milvus ingestion. Configure local paths and
service endpoints in `.env`.

## Repository map

```text
src/s2cs/env/         Retrieval environment, corpus ETL, and paper tools
src/s2cs/agent/       Agent loop, policy, judging, and trajectories
src/s2cs/synthesis/   Question synthesis, filtering, and difficulty grading
src/s2cs/eval/        Literature-search and web-search evaluation adapters
src/s2cs/trainer/     verl data, tools, rewards, and integration patches
configs/trainer/      Trainer tool configuration
tests/                Mock-based unit tests
```

The Python package is named `s2cs`; the project and environment are referred to
as **Simulated Scholar Search (S3)**.

## Research scope

S3 is intended for studying:

- simulated-to-real transfer of search behavior;
- curriculum and question design for search agents;
- scientific-literature retrieval with citation navigation; and
- outcome-based rewards for multi-turn tool use.

It is not a hosted search product or a drop-in Semantic Scholar replacement.

## Acknowledgements

The training integration uses [verl](https://github.com/volcengine/verl).
Integration patches are included under `src/s2cs/trainer/patches/`.

## Citation

If S3 is useful in your research, cite the software using
[`CITATION.cff`](CITATION.cff).

## License

Released under the [Apache License 2.0](LICENSE).
