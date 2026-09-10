# ✅ Tests

[← Back to README](../README.md)

```bash
pip install -e ".[dev]"
pytest
```

Unit tests target the small, deterministic classes: `AdaptiveFrameSampler`, `MotionImageEncoder`, `DatasetSplitter`, `LabelEncoder`, `LabelMapRepository`, `FrameWindow`, `MotionTrigger`, `DoorApproachDecider` (plain numpy) and `MaskCleaner` (needs OpenCV).

🎯 They run without a GPU, a dataset, a camera or the Keras model. They are the fast, no-data feedback loop for changes to that logic. The camera path (`LiveSession`) and the network are exercised through the CLI and the [demo](DEMO.md) instead.
