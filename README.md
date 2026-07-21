`# Real-Time Activity and Intention Recognition`

A layered Python service for recognizing a person's **intention at an
automatic door from video** — specifically, whether someone walking into
view is **entering** (approaching and heading through the door) or just
**passing by** (walking along or around the door without going through it).
The same pipeline generalizes to other single-subject activity recognition
tasks, but the door use case is what it's built and demonstrated for.

It works by compressing a video clip into a single still image that encodes
both the *shape* and *speed* of the motion, then classifying that image with
a shallow convolutional network suited to embedded deployment (e.g. running
next to the door itself, not in the cloud).

## How it works

1. **Tracking.** Each frame is denoised (Gaussian blur + dilation/erosion),
   then the moving subject is separated from the background with KNN-based
   background subtraction — chosen for its robustness to shadows and
   changing lighting. Clips with no detectable moving contour are discarded.
2. **Adaptive sampling.** Rather than sampling frames at a fixed rate,
   the sampling interval adapts to how fast the subject is moving
   (estimated via optical flow): fast motion is sampled more densely so it
   isn't missed, slow motion less densely so it isn't redundant. The first
   and last few samples of each clip are trimmed to reduce noise.
3. **Motion image encoding.** The sampled frames are combined into a single
   image via recency-weighted summation — each new frame is folded in at
   full strength while everything accumulated so far is decayed by 30%, so
   recent motion dominates and older motion fades out. This "dynamic image"
   is what gets classified, instead of a full video or an optical-flow
   volume — far cheaper for embedded hardware. Because it encodes a whole
   trajectory in one image, it naturally captures *where someone was headed*
   (toward the door vs. along it), not just what they looked like in one frame.
4. **Augmentation.** Mirrored and randomly shifted variants of each motion
   image expand the training set with translation/orientation invariance.
5. **Classification.** A shallow AlexNet (5 convolutional + 3 fully
   connected layers) with Tanh activations, Softmax output, and
   normal-Glorot weight initialization classifies the motion image into the
   trained set of classes (e.g. `entering` / `passing_by`). It's trained
   with SGD (tuned learning rate/momentum, favoring generalization over
   convergence speed) on a 60/30/10 train/validation/test split.
6. **Real-time inference.** The same tracking → sampling → encoding
   pipeline runs end-to-end on a single freshly captured clip and feeds the
   trained model, returning the recognized intention by name (not just a
   class index) — e.g. `"entering"` with a confidence score — for real-time/
   embedded use, such as triggering the door.

The class labels themselves aren't hardcoded: whatever label strings appear
in your training data's `labeled_data.csv` become the classes the model
learns and later reports by name (see `LabelEncoder` / `LabelMapRepository`
below). The `entering` / `passing_by` pair is the reference scenario used
throughout this repo's tests and demos.

## Architecture (layers)

Everything lives directly under `src/` as plain top-level modules/packages —
there's no wrapping package name to import through:

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

Internally, modules import each other flatly, e.g. `from domain.entities
import MotionImage` or `from preprocessing.noise_reducer import
NoiseReducer` — this works because `src/` itself (not a subfolder inside it)
is what gets put on `sys.path`, either by running a script directly (Python
adds its own directory to `sys.path` automatically) or via the editable
install below.

Each class has exactly one job:

| Layer | Classes | Responsibility |
|---|---|---|
| `preprocessing/` | `NoiseReducer`, `BackgroundSubtractor`, `MotionContourDetector` | tracking: clean, isolate, and confirm the moving subject |
| `preprocessing/` | `OpticalFlowEstimator`, `AdaptiveFrameSampler`, `MotionImageEncoder` | velocity-adaptive sampling and the recency-weighted motion image |
| `augmentation/` | `MirrorAugmenter`, `ShiftCropAugmenter` | training-set augmentation |
| `modeling/` | `AlexNetBuilder`, `DatasetSplitter`, `LabelEncoder`, `ModelTrainer` | building, splitting, and training the network |
| `modeling/` | `NetworkInputFormatter`, `ActivityPredictor` | preparing and running real-time inference, decoding predictions back to class names |
| `data/` | `LabelMapRepository` | persists which class index means which label (e.g. `0 -> entering`), so inference can report a name, not a number |
| `pipelines/` | `ClipPreprocessingPipeline`, `DatasetPreparationPipeline`, `TrainingPipeline`, `RealtimeInferencePipeline` | sequence the classes above — no business logic of their own |
| `app.py` | `Application` | the composition root: wires concrete classes together from `Settings` and exposes `prepare_dataset()`, `train()`, `predict()` |

Dependencies only point downward (`pipelines` → `preprocessing`/`modeling`/`data`
→ `domain`); nothing in `domain/` or `preprocessing/` imports from `pipelines/`
or `app.py`.

## Setup

TensorFlow requires **Python 3.10–3.12** (it has no wheel for 3.13+ yet). If
your default `python`/`py` is newer, create the virtual environment with an
explicit 3.12 interpreter:

```bash
py -3.12 -m venv .venv          # Windows; use `python3.12 -m venv .venv` elsewhere
.venv\Scripts\activate           # Windows; `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
pip install -e .                 # optional: makes config/domain/data/... importable from anywhere
```

## Usage

Run from the project root — no install required, since running a script
directly puts `src/` on the import path automatically:

```bash
# 1. raw labeled clips -> augmented motion-image dataset
python src/__main__.py prepare-dataset

# 2. prepared dataset -> trained AlexNet model
python src/__main__.py train

# 3. classify a single new clip
python src/__main__.py predict path/to/clip.mp4
# -> predicted intention: entering (confidence 97.31%)
```

Expected input layout (created automatically by `Settings.paths.ensure_exists()`):

```
data/
  raw_videos/            *.mp4 clips of people near the door
  labeled_data.csv        File,Label pairs, e.g. "clip001.mp4,entering" / "clip002.mp4,passing_by"
models/
  alexnet_model.keras      the trained network, written by `train`, read by `predict`
  label_map.json           class-index -> label-name mapping, written by `train`, read by `predict`
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Unit tests target the pure-logic classes (`AdaptiveFrameSampler`,
`MotionImageEncoder`, `DatasetSplitter`, `LabelEncoder`, `LabelMapRepository`)
that have no OpenCV/TensorFlow dependency, so they run without a GPU, a
dataset, or native libraries installed. This is the fast, no-data feedback
loop for changes to that logic.

## Trying it end to end without a real camera

`examples/` has small utilities (demo/debug aids, not part of the package)
for exercising the full OpenCV/TensorFlow pipeline without needing real
door footage — they render a moving square whose trajectory either heads
into a marked door zone (`entering`) or passes alongside it (`passing_by`):

```bash
# one command: generates both synthetic clips, prepares the dataset, trains,
# then predicts on both and reports whether each intention was recognized
python examples/run_door_demo.py

# or step through it manually:
python examples/generate_synthetic_clip.py data/raw_videos/clip.mp4 --intent entering
python examples/visualize_pipeline.py data/raw_videos/clip.mp4 data/contact_sheet.jpg
```

`visualize_pipeline.py` runs the real preprocessing classes on one clip and
saves a labeled contact sheet (raw frame | background-subtracted mask | final
motion image), and prints the estimated velocity, the resulting adaptive
sampling rate, and which frame indices got sampled — useful for
sanity-checking `AdaptiveFrameSampler` and `MotionImageEncoder` against an
actual clip.

**On `run_door_demo.py`'s classification accuracy:** the demo proves the
*pipeline* end to end — tracking, adaptive sampling, motion-image encoding,
augmentation, training, and inference all run without error, and predictions
come back as real label names (`"entering"`, not `"class 0"`). Whether the
prediction is *correct* on this demo is not meaningful to read into: AlexNet
has ~28M parameters and expects real datasets of hundreds of clips per class
(the reference methodology this repo implements was validated on 738 real
clips); training it on 2 synthetic clips is expected to overfit and swing
unpredictably between epochs and runs. Point it at a real, larger labeled
dataset to evaluate actual accuracy.
