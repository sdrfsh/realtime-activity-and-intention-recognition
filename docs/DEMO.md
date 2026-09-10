# 🎥 Trying It End to End Without a Real Camera

[← Back to README](../README.md)

`examples/` has small utilities (demo/debug aids, not part of the package) for exercising the full OpenCV/Keras pipeline without needing real door footage. They render a moving square whose trajectory either heads into a marked door zone (`entering`) or passes alongside it (`passing_by`).

## ▶️ One-shot demo

```bash
# generates both synthetic clips, prepares the dataset, trains,
# then predicts on both and reports whether each intention was recognized
python examples/run_door_demo.py
```

## 🔍 Or step through it manually

```bash
python examples/generate_synthetic_clip.py data/raw_videos/clip.mp4 --intent entering
python examples/visualize_pipeline.py data/raw_videos/clip.mp4 data/contact_sheet.jpg
```

`visualize_pipeline.py` runs the real preprocessing classes on one clip and saves a labeled contact sheet (raw frame | background-subtracted mask | final motion image), and prints:
- the estimated velocity
- the resulting adaptive sampling rate
- which frame indices got sampled

Useful for sanity-checking `AdaptiveFrameSampler` and `MotionImageEncoder` against an actual clip.

`profile_predict.py` times one network prediction and one optical-flow call on the current machine (downloads the pretrained model on first run):

```bash
python examples/profile_predict.py
```

## ⚠️ A note on `run_door_demo.py`'s accuracy

The demo proves the **pipeline** end to end: tracking, adaptive sampling, motion-image encoding, augmentation, training, and inference all run without error, and predictions come back as real label names (`"entering"`, not `"class 0"`).

Whether the prediction is *correct* on this demo isn't meaningful to read into: AlexNet has ~28M parameters and expects real datasets of hundreds of clips per class (the reference methodology this repo implements was validated on 738 real clips). Training it on 2 synthetic clips is expected to overfit and swing unpredictably between epochs and runs.

👉 Point it at a real, larger labeled dataset to evaluate actual accuracy.

## Camera demo

No training needed: by default the decision comes from the subject's
heading toward the door edge; the pretrained Hub model is only downloaded
when you pass `--decision network`.

```bash
python src/__main__.py live --camera 0 --window 3.0
# or the model you trained with the demo above:
python src/__main__.py live --camera 0 --model models/alexnet_model.keras
```

The session calibrates, shows the camera preview, detects motion, fills a
fixed window, and prints the predicted label with decision latency. Walk
toward the camera and verify that one label appears within one second of
motion onset. Cooldown suppresses repeated decisions while the subject is
still present. For an embedded board, pass its OpenCV camera index or URL;
for headless operation add `--no-preview`. Tell it which edge of the image
the door lies along with `--door-side left|right|top|bottom`: walking toward
that edge is `entering`, anything else `passing_by` (see
[Usage](USAGE.md#-where-the-door-is-and-how-a-decision-is-made)). Add
`--decision network` to classify with the model instead.
