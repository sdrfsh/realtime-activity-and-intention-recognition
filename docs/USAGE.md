# 🚀 Usage

[← Back to README](../README.md)

Run from the project root — no install required, since running a script directly puts `src/` on the import path automatically.

```bash
# 1. raw labeled clips -> augmented motion-image dataset
python src/__main__.py prepare-dataset

# 2. prepared dataset -> trained AlexNet model
python src/__main__.py train

# 3. classify a single new clip
python src/__main__.py predict path/to/clip.mp4
# -> predicted intention: entering (confidence 97.31%)
```

## 📁 Expected input layout

Created automatically by `Settings.paths.ensure_exists()`:

```
data/
  raw_videos/            *.mp4 clips of people near the door
  labeled_data.csv        File,Label pairs, e.g. "clip001.mp4,entering" / "clip002.mp4,passing_by"
models/
  alexnet_model.keras      the trained network, written by `train`, read by `predict`
  label_map.json           class-index -> label-name mapping, written by `train`, read by `predict`
```

## 🔗 Related

- 🎥 No real camera footage yet? See [Demo Without a Camera](DEMO.md)
- ✅ Want to verify your changes? See [Testing](TESTING.md)
