# Insures that if a file has the same name as the command that the command doesn't get skipped.
.PHONY: install format lint test check

# One-time setup
install:
	pip install -e ".[dev]"

# Auto-fix formatting + imports
format:
	black .
	ruff check . --fix

# Verifies that no changes is needed
lint:
	ruff check .
	black --check .

# Run tests
test:
	pytest

# Check to run before committing
check: lint test