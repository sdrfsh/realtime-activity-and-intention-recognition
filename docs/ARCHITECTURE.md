# 🏗️ Architecture

[← Back to README](../README.md)

Everything lives directly under `src/` as plain top-level modules/packages. There's no wrapping package name to import through:

```
src/
  config.py            configuration: paths, hyperparameters, scene (door side), live options; one dataclass per concern
  domain/               plain data objects (LabeledClip, MotionImage, DatasetSplit, PredictionResult, DoorApproach, LiveDecision)
  data/                 all disk/video/camera I/O: frame sources, readers/writers, label repositories
  preprocessing/        tracking + sampling, one transformation per class
  augmentation/         image augmentation, one strategy per class
  modeling/             deciding: network construction/training/inference and the heading-based door decider
  pipelines/            orchestration: clip/window pipelines, LiveSession (+ its LivePreview), dataset/training pipelines
  app.py                Application: the main class that builds and runs the pipelines
  __main__.py           CLI entry point (python src/__main__.py ...)
```

## 📦 Imports

Internally, modules import each other flatly, e.g. `from domain.entities import MotionImage` or `from preprocessing.noise_reducer import NoiseReducer`.

This works because `src/` itself (not a subfolder inside it) is what gets put on `sys.path`, either by running a script directly (Python adds its own directory to `sys.path` automatically) or via the editable install (see [Setup](SETUP.md)).

## 🧩 Who does what

Each class has exactly one job:

| Layer | Classes | Responsibility |
|---|---|---|
| `preprocessing/` | `NoiseReducer`, `BackgroundSubtractor`, `MaskCleaner`, `MotionContourDetector` | 🎯 tracking: clean, isolate, denoise the mask, and confirm the moving subject |
| `preprocessing/` | `OpticalFlowEstimator`, `AdaptiveFrameSampler`, `MotionImageEncoder`, `MotionTrigger`, `FrameWindow` | velocity-adaptive sampling, motion state, and bounded live windows |
| `augmentation/` | `MirrorAugmenter`, `ShiftCropAugmenter` | 🔄 training-set augmentation |
| `modeling/` | `AlexNetBuilder`, `DatasetSplitter`, `LabelEncoder`, `ModelTrainer` | 🧮 building, splitting, and training the network |
| `modeling/` | `NetworkInputFormatter`, `ActivityPredictor` | ⚡ preparing and running network inference, decoding predictions back to class names |
| `modeling/` | `DoorApproachDecider` | 🚪 follow the subject's centroid across a window and decide entering/passing by from its heading toward the configured door edge |
| `data/` | `FrameSource`, `FileFrameSource`, `CameraFrameSource`, repositories | provide frames and persist labels/images; only `CameraFrameSource` opens camera capture |
| `pipelines/` | `ClipPreprocessingPipeline`, `WindowClipPipeline`, `DatasetPreparationPipeline`, `TrainingPipeline` | the offline path: clips → motion images → augmented dataset → trained model |
| `pipelines/` | `LiveSession`, `LivePreview` | the continuous camera path: trigger, window, one decision per window (by heading via `DoorApproachDecider`, or by the network via `ActivityPredictor`); `LivePreview` only draws |
| `app.py` | `Application` | the composition root: wires concrete classes together from `Settings` and exposes `prepare_dataset()`, `train()`, `predict()`, `live()` |

## ⬇️ Dependency direction

Dependencies only point downward:

```
pipelines → preprocessing / modeling / data → domain
```

Nothing in `domain/` or `preprocessing/` imports from `pipelines/` or `app.py`.

`LiveSession` owns one stateful KNN subtractor for its complete session. Its
capture thread only reads frames into a bounded queue; the caller's thread
performs noise reduction, trigger/window management, motion encoding and the
decision, and hands the current state to `LivePreview` for drawing.

## 🗂️ Around `src/`

```
tests/       fast unit tests for the pure-logic classes (see TESTING.md)
examples/    demo/debug scripts, not part of the package (see DEMO.md)
docs/        this documentation
conftest.py  puts src/ on sys.path for pytest
```
