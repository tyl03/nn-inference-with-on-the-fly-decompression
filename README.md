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

src/ # Core project code (quantization, inference logic)
scripts/ # Runnable scripts and sanity checks
