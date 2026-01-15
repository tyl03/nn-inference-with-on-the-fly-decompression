# Bachelor Project - PyTorch

This project explores quantization techniques in PyTorch, with a focus on memory-efficient inference suitable for CPU-based, memory-constrained devices.

---

## Requirements

- Python 3.10 or newer (tested with Python 3.12)
- No GPU required (CPU-only PyTorch)
- Windows (tested)

---

## Setup (Windows)

Create and activate a virtual environment:

```cmd
py -m venv myenv
myenv\Scripts\activate
```

---

## Install dependencies

```cmd
python -m pip install -r requirements.txt
```

## Run script (example)

```cmd
python scripts\test.py
```

---

## Project Structure

```text
.
├── src/
│   └── quantization.py
├── scripts/
│   └── test.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Continous Integration

This project uses GitHub Actions to automatically verify that the code runs on a clean environment. On each push, the workflow installs dependencies from `requirements.txt` and executes a small test script to ensure that the project is reproducible and free of setup errors.

---

## Running scripts

Run all scripts from the project root using:

```python
python -m scripts.train_mnist
python -m scripts.pipeline
```

---

## Running tests

To run all tests from the project root:

```python
python -m pytest -q
```
