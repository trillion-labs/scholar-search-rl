PATHS := src tests

.PHONY: sync sync-trainer lint format check test test-trainer clean-logs

sync:
	uv sync --group dev --group eval

sync-trainer:
	uv sync --group dev --group trainer

lint:
	uv run ruff check $(PATHS)

format:
	uv run ruff format $(PATHS)

check:
	uv run ruff check $(PATHS)
	uv run ruff format --check $(PATHS)

test:
	uv run pytest tests --ignore=tests/trainer

test-trainer:
	uv run pytest tests/trainer

clean-logs:
	find logs -type f -name '*.out' -mtime +14 -delete
