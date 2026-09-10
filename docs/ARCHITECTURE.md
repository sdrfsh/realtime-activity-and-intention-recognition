# 🏗️ Architecture

[← Back to README](../README.md)

Everything lives directly under `src/` as plain top-level modules/packages — there's no wrapping package name to import through:

```
src/
  config.py            configuration: paths + hyperparameters + scene (door side), one dataclass per concern
  domain/               plain data objects (LabeledClip, MotionImage, DatasetSplit, PredictionResult, ...)
  data/                 all disk/video I/O + frame sources and the label-map file
  preprocessing/        tracking + sampling, one transformation per class
  augmentation/         image augmentation, one strategy per class
  modeling/             network construction, training, and inference
  pipelines/            clip/window orchestration and continuous LiveSession
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
| `preprocessing/` | `DoorApproachDecider` | 🚪 follow the subject's centroid across a window and decide entering/passing by from its heading toward the configured door edge |
| `preprocessing/` | `NoiseReducer`, `BackgroundSubtractor`, `MaskCleaner`, `MotionContourDetector` | 🎯 tracking: clean, isolate, denoise the mask, and confirm the moving subject |
| `preprocessing/` | `OpticalFlowEstimator`, `AdaptiveFrameSampler`, `MotionImageEncoder`, `MotionTrigger`, `FrameWindow` | velocity-adaptive sampling, motion state, and bounded live windows |
| `augmentation/` | `MirrorAugmenter`, `ShiftCropAugmenter` | 🔄 training-set augmentation |
| `modeling/` | `AlexNetBuilder`, `DatasetSplitter`, `LabelEncoder`, `ModelTrainer` | 🧮 building, splitting, and training the network |
| `modeling/` | `NetworkInputFormatter`, `ActivityPredictor` | ⚡ preparing and running real-time inference, decoding predictions back to class names |
| `data/` | `FrameSource`, `FileFrameSource`, `CameraFrameSource`, repositories | provide frames and persist labels/images; only `CameraFrameSource` opens camera capture |
| `pipelines/` | `ClipPreprocessingPipeline`, `WindowClipPipeline`, `LiveSession`, dataset/training pipelines | preserve the training clip path and run the continuous camera path; `LiveSession` decides per window either by heading (`DoorApproachDecider`) or by the network (`ActivityPredictor`) |
| `app.py` | `Application` | the composition root: wires concrete classes together from `Settings` and exposes `prepare_dataset()`, `train()`, `predict()`, `live()` |

## ⬇️ Dependency direction

Dependencies only point downward:

```
pipelines → preprocessing / modeling / data → domain
```

Nothing in `domain/` or `preprocessing/` imports from `pipelines/` or `app.py`.

`LiveSession` owns one stateful KNN subtractor for its complete session. Its
capture thread only reads frames into a bounded queue; the worker performs
noise reduction, trigger/window management, motion encoding, and prediction.
