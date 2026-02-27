# Efficient Neural Network Inference with On-the-Fly Decompression

![CI](https://github.com/tyl03/nn-inference-with-on-the-fly-decompression/actions/workflows/ci.yml/badge.svg)

This project investigates memory-efficient inference of pruned neural networks by storing model weights in compressed form and decompressing one layer at a time during inference.

Instead of loading the full model into memory, only a single layer is decompressed, evaluated, and discarded before moving to the next layer. This significantly reduces peak memory usage at the cost of additional decompression overhead.

The approach is suitable for CPU-only environments and memory-constrained devices.

## Table of Contents

- [Efficient Neural Network Inference with On-the-Fly Decompression](#efficient-neural-network-inference-with-on-the-fly-decompression)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Clone the repository](#clone-the-repository)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install the project](#install-the-project)
      - [With Make (Linux / macOS / WSL)](#with-make-linux--macos--wsl)
      - [Without Make (Works everywhere)](#without-make-works-everywhere)
  - [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Code Quality \& Testing](#code-quality--testing)
  - [Project Structure](#project-structure)
  - [Continuous Integration](#continuous-integration)
  - [Credits](#credits)
  - [License](#license)

## Requirements

- Python 3.10 or newer (tested with Python 3.12)
- Linux / macOS / WSL recommended for full `Makefile` support
- Windows Powershell supported (without `make`)
- CPU-only environment (No GPU needed)

## Installation

### Clone the repository

```bash
git clone https://github.com/tyl03/nn-inference-with-on-the-fly-decompression.git
cd nn-inference-with-on-the-fly-decompression
```

### Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Remember to change your Python Interpreter to the created environment.

### Troubleshooting

#### Resetting the virtual environment

If your environment gets into a bad state, delete and recreate it:

macOS / Linux / WSL:

```bash
deactivate  # if active
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
make dev
```

Powershell:

```bash
deactivate  # if active
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Install the project

#### With Make (Linux / macOS / WSL)

```bash
make install
```

You can then verify that everything works:

```bash
make check
```

If `make check` shows an error, then run:

```bash
make format
make check
```

Available Make commands:

```bash
make install # Install runtime dependencies only
make dev # Install runtime + dev dependencies
make format # Auto-format code (Black + Ruff fix)
make lint # Check formatting without modifying files
make test # Run pytest
make check # Run lint + test
```

#### Without Make (Works everywhere)

Install project with development dependencies:

```bash
pip install -e ".[dev]"
```

Format code:

```bash
black .
ruff check . --fix
```

Check formatting without modifying code:

```bash
ruff check .
black --check .
```

Run tests:

```bash
pytest
```

## Usage

After installation, the package can be imported as:

```python
from nn_compression.pruning import global_magnitude_prune_linear_layers
from nn_compression.export_compressed import export_fcn_to_compressed
from nn_compression.layerwise_inference import layerwise_evaluate_accuracy
```

## Running Experiments

To reproduce the main experiment:

```bash
python scripts/layerwise_inference_exp.py
```

This compares:

- Baseline FP32 inference
- Pruned FP32 inference
- Layerwise Zstd compressed inference

It reports:

- Accuracy
- Storage footprint
- Peak decompressed layer size
- Inference timing

## Code Quality & Testing

This project enforces consistent formatting and linting using:

- **Black** (formatter)
- **Ruff** (linter + import sorting)
- **PyTest** (unit testing)

Before committing changes, run:

```bash
make check
```

Or manually:

```bash
ruff check .
black --check .
pytest
```

All tests must pass before merging changes.

## Project Structure

```text
.
├── src/
│   └── nn_compression/
│       ├── pruning.py
│       ├── training.py
│       ├── export_compressed.py
│       ├── layerwise_inference.py
│       └── ...
├── scripts/
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

The project follows the standard `src/` layout for Python packaging.

## Continuous Integration

GitHub Actions automatically:

- Creates a fresh environment
- Installs the package using `pip install -e ".[dev]"`
- Runs the test suite

This ensures portability and reproducibility.

## Credits

```markdown
Author:
Tülin Cetinkaya
BSc Software Technology, Technical University of Denmark (DTU)

This project was developed as part of a Bachelor's thesis on memory-efficient neural network inference.

The project builds upon:

- PyTorch
- Zstandard
- NumPy
- Matplotlib
- PyTest
```

## License

This project is licensed under the MIT License.
