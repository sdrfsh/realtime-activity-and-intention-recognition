# 🚪 Real-Time Activity and Intention Recognition

A layered Python service that watches a video feed near an automatic door and figures out someone's **intention**: are they 🚶‍♂️ **entering** (heading through the door), or just 🔀 **passing by**?

It works by compressing a video clip into a single still image that encodes both the *shape* and *speed* of motion, then classifying that image with a shallow convolutional network light enough to run embedded — right next to the door, not in the cloud. The same pipeline generalizes to other single-subject activity recognition tasks, but the door use case is what it's built and demonstrated for.

## 📖 Documentation

| | |
|---|---|
| 🧠 [How It Works](docs/HOW_IT_WORKS.md) | The 6-stage pipeline: tracking → sampling → motion-image encoding → augmentation → classification → real-time inference |
| 🏗️ [Architecture](docs/ARCHITECTURE.md) | Layer-by-layer breakdown of `src/`, who does what, and the dependency rules |
| ⚙️ [Setup](docs/SETUP.md) | Python version requirements and installation |
| 🚀 [Usage](docs/USAGE.md) | CLI commands for preparing data, training, and predicting |
| ✅ [Testing](docs/TESTING.md) | Running the fast, no-GPU unit test suite |
| 🎥 [Demo Without a Camera](docs/DEMO.md) | Try the full pipeline with synthetic clips — no door footage needed |

## ⚡ Quick start

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python src/__main__.py prepare-dataset
python src/__main__.py train
python src/__main__.py predict path/to/clip.mp4
# -> predicted intention: entering (confidence 97.31%)
```

See [Setup](docs/SETUP.md) and [Usage](docs/USAGE.md) for the full walkthrough.

## 🏷️ Labels are yours to define

Class labels aren't hardcoded — whatever label strings appear in your `labeled_data.csv` become the classes the model learns and reports by name. `entering` / `passing_by` is just the reference scenario used throughout this repo's tests and demos.
