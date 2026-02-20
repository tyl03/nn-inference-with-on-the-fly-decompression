# Insures that if a file has the same name as the command that the command doesn't get skipped.
.PHONY: install dev format lint test check

# One-time setup
install:
	python -m pip install -e . -c constraints.txt

dev: install
	python -m pip install -e ".[dev]" -c constraints.txt

# Auto-fix formatting + imports
format:
	python -m black .
	python -m ruff check . --fix

# Verifies that no changes is needed
lint:
	python -m ruff check .
	python -m black --check .

# Run tests
test:
	python -m pytest

# Check to run before committing
check: lint test