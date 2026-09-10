"""End-to-end demo of the automatic-door use case: generates one "entering"
and one "passing_by" synthetic clip, prepares the dataset, trains a model,
then predicts on both clips and reports whether each was recognized
correctly. A demo/debug aid only — it drives the real `Application`, the
same one the CLI uses.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_synthetic_clip import generate_clip  # noqa: E402

from app import Application  # noqa: E402
from config import load_settings  # noqa: E402
from data.label_repository import LabelRepository  # noqa: E402
from domain.entities import LabeledClip  # noqa: E402


def main(epochs: int) -> None:
    settings = load_settings()
    settings.paths.ensure_exists()
    settings = replace(settings, network=replace(settings.network, epochs=epochs, batch_size=2))

    entering_path = settings.paths.raw_video_dir / "demo_entering.mp4"
    passing_path = settings.paths.raw_video_dir / "demo_passing_by.mp4"
    generate_clip(entering_path, "entering")
    generate_clip(passing_path, "passing_by")

    LabelRepository(settings.paths.labels_csv).write(
        [
            LabeledClip(entering_path, "entering"),
            LabeledClip(passing_path, "passing_by"),
        ]
    )

    app = Application(settings)

    print("preparing dataset (tracking, adaptive sampling, motion image, augmentation)...")
    app.prepare_dataset()

    print(f"training ({epochs} epochs)...")
    app.train()

    print("\npredicting on both clips:")
    for path, expected in ((entering_path, "entering"), (passing_path, "passing_by")):
        # evaluate the model just trained, not the pretrained Hub default
        result = app.predict(path, model_source=settings.paths.trained_model_path)
        if result is None:
            print(f"  {path.name}: no motion detected")
            continue
        confidence = result.class_probabilities[result.label_index]
        status = "correct" if result.label == expected else "MISMATCH"
        print(
            f"  {path.name}: predicted='{result.label}' (confidence {confidence:.2%}), "
            f"expected='{expected}' [{status}]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    main(args.epochs)
