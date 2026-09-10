# 🚀 Usage

[← Back to README](../README.md)

Run from the project root. No install is required, since running a script directly puts `src/` on the import path automatically.

```bash
# 1. raw labeled clips -> augmented motion-image dataset
python src/__main__.py prepare-dataset

# 2. prepared dataset -> trained AlexNet model
python src/__main__.py train

# 3. classify a single new clip
python src/__main__.py predict path/to/clip.mp4
# -> predicted intention: entering (confidence 97.31%)

# 4. continuous camera inference
python src/__main__.py live --camera 0 --window 3.0 --motion-threshold 2000

# 5. the door is along the top edge of the camera image (subject heading there = entering)
python src/__main__.py live --camera 0 --door-side top
```

## 🧠 Which model runs

`predict` and `live` default to the published pretrained classifier
[`sdrfsh/alexnet-door-entry-classifier`](https://huggingface.co/sdrfsh/alexnet-door-entry-classifier)
on the Hugging Face Hub. The `.keras` file (~700 MB) is downloaded once and
cached by `huggingface_hub`; no local training is needed. Keras 3 runs on the
JAX backend by default (set `KERAS_BACKEND` to override).

`--model` switches the model:

```bash
python src/__main__.py live --model models/alexnet_model.keras     # the model `train` wrote
python src/__main__.py live --model hf://someone/other-repo/file.keras
```

A local model reads its class order from the `label_map.json` beside it; the
Hub model uses `NetworkSettings.pretrained_classes` (`0 = passing_by`,
`1 = entering`).

`--camera` accepts an OpenCV device index or a capture URL. Use
`--no-preview` on a display-less board and `--no-decide` for motion-only mode.
`LiveSettings` controls the source, window, warm-up, cooldown, motion
threshold, FPS probe size, and queue bound. The live session measures capture
FPS at startup and keeps one KNN background model for the whole session.

`predict` remains a file-backed development/regression command and feeds a
file source through the same live session code path.

## 🚪 Where the door is, and how a decision is made

By default (`--decision heading`) the intention is read straight off the
subject's movement. Tell the pipeline which edge of the camera image the door
lies along with `--door-side right|left|top|bottom` (or `SceneSettings.door_side`).
For every window the tracked subject's foreground centroid is followed; if it
travels toward that edge it is **entering**, otherwise it is **passing by**.
Purely 2-D: only the four frame edges are distinguished and depth is ignored.
The frame is never rotated; the preview shows the normal camera view with
only the door edge highlighted.

`SceneSettings` tunes the rule: `approach_min_travel` (how far toward the door
the centroid must move, as a fraction of the frame; default 0.10),
`approach_max_angle_deg` (how far the heading may deviate from straight at the
door; default 60°), `trajectory_smoothing_frames` and `min_foreground_pixels`
for noise robustness, and `max_foreground_fraction` to ignore frames where the
background model has not settled.

`--decision network` switches back to classifying the motion image with the
AlexNet model (`--model` picks which); the door side is then ignored.

```bash
python src/__main__.py live --camera 0 --door-side top
python src/__main__.py predict clip.mp4 --door-side bottom
python src/__main__.py live --camera 0 --decision network --model models/alexnet_model.keras
```

## 📁 Expected input layout

Created automatically by `Settings.paths.ensure_exists()`:

```
data/
  raw_videos/            *.mp4 clips of people near the door
  labeled_data.csv        File,Label pairs, e.g. "clip001.mp4,entering" / "clip002.mp4,passing_by"
models/
  alexnet_model.keras      the trained network, written by `train`, read by `predict --model models/alexnet_model.keras`
  label_map.json           class-index -> label-name mapping, written by `train`, read alongside the local model
```

## 🔗 Related

- 🎥 No real camera footage yet? See [Demo Without a Camera](DEMO.md)
- ✅ Want to verify your changes? See [Testing](TESTING.md)
