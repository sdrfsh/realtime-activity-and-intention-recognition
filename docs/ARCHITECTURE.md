# 🏗️ Architecture

[← Back to README](../README.md)

Everything lives directly under `src/` as plain top-level modules/packages — there's no wrapping package name to import through:

```
src/
  config.py            configuration: paths + hyperparameters, one dataclass per concern
  domain/               plain data objects (LabeledClip, MotionImage, DatasetSplit, PredictionResult, ...)
  data/                 all disk/video I/O + the label-map file (VideoReader, LabelRepository, LabelMapRepository, ImageLoader, ImageWriter)
  preprocessing/        tracking + sampling, one transformation per class
  augmentation/         image augmentation, one strategy per class
  modeling/             network construction, training, and inference
  pipelines/            orchestrators that sequence the classes above
  app.py                Application: the main class — builds and runs the pipelines
  __main__.py           CLI entry point (python src/__main__.py ...)
```

## 📦 Imports

Internally, modules import each other flatly, e.g. `from domain.entities import MotionImage` or `from preprocessing.noise_reducer import NoiseReducer`.

This works because `src/` itself (not a subfolder inside it) is what gets put on `sys.path`, either by running a script directly (Python adds its own directory to `sys.path` automatically) or via the editable install (see [Setup](SETUP.md)).

## 🧩 Who does what

Each class has exactly one job:

| Layer | Classes | Responsibility |
|---|---|---|
| `preprocessing/` | `NoiseReducer`, `BackgroundSubtractor`, `MotionContourDetector` | 🎯 tracking: clean, isolate, and confirm the moving subject |
| `preprocessing/` | `OpticalFlowEstimator`, `AdaptiveFrameSampler`, `MotionImageEncoder` | ⏱️ velocity-adaptive sampling and the recency-weighted motion image |
| `augmentation/` | `MirrorAugmenter`, `ShiftCropAugmenter` | 🔄 training-set augmentation |
| `modeling/` | `AlexNetBuilder`, `DatasetSplitter`, `LabelEncoder`, `ModelTrainer` | 🧮 building, splitting, and training the network |
| `modeling/` | `NetworkInputFormatter`, `ActivityPredictor` | ⚡ preparing and running real-time inference, decoding predictions back to class names |
| `data/` | `LabelMapRepository` | 🏷️ persists which class index means which label (e.g. `0 -> entering`), so inference can report a name, not a number |
| `pipelines/` | `ClipPreprocessingPipeline`, `DatasetPreparationPipeline`, `TrainingPipeline`, `RealtimeInferencePipeline` | 🔗 sequence the classes above — no business logic of their own |
| `app.py` | `Application` | 🚪 the composition root: wires concrete classes together from `Settings` and exposes `prepare_dataset()`, `train()`, `predict()` |

## ⬇️ Dependency direction

Dependencies only point downward:

```
pipelines → preprocessing / modeling / data → domain
```

Nothing in `domain/` or `preprocessing/` imports from `pipelines/` or `app.py`.
