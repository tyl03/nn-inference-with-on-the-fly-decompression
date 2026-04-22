.PHONY: install dev format lint test check torch-cpu base-deps dev-deps

# Install CPU-only PyTorch
torch-cpu:
	python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# Install non-torch runtime dependencies
base-deps:
	python -m pip install "numpy>=2,<3" "matplotlib>=3.7,<4" "zstandard==0.25.0"

# Install non-torch dev dependencies
dev-deps:
	python -m pip install "pytest==8.4.0" "black>=24.0.0" "ruff>=0.4.0" "setuptools==80.9.0"

# One-time setup
install: torch-cpu base-deps
	python -m pip install -e . --no-deps

dev: install dev-deps
	python -m pip install -e ".[dev]" --no-deps

# Auto-fix formatting + imports
format:
	python -m black .
	python -m ruff check . --fix

# Verifies that no changes are needed
lint:
	python -m ruff check .
	python -m black --check .

# Run tests
test:
	python -m pytest

# Check to run before committing
check: lint test