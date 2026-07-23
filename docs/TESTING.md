# ✅ Tests

[← Back to README](../README.md)

```bash
pip install -e ".[dev]"
pytest
```

Unit tests target the pure-logic classes — `AdaptiveFrameSampler`, `MotionImageEncoder`, `DatasetSplitter`, `LabelEncoder`, `LabelMapRepository` — that have no OpenCV/TensorFlow dependency.

🎯 That means they run without a GPU, a dataset, or native libraries installed — the fast, no-data feedback loop for changes to that logic.
