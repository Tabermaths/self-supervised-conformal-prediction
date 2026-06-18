.PHONY: install install-dev test lint format check reference-figures clean

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

check: lint test

reference-figures:
	python scripts/reproduce_reference_figures.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache outputs/reference_figures
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
