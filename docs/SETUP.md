# ⚙️ Setup

[← Back to README](../README.md)

> 🐍 TensorFlow requires **Python 3.10–3.12** (no wheel for 3.13+ yet). If your default `python`/`py` is newer, create the virtual environment with an explicit 3.12 interpreter.

```bash
py -3.12 -m venv .venv          # Windows; use `python3.12 -m venv .venv` elsewhere
.venv\Scripts\activate           # Windows; `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
pip install -e .                 # optional: makes config/domain/data/... importable from anywhere
```

Next up: [Usage](USAGE.md) →
