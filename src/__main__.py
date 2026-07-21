"""Command-line entry point.

Usage (run from the project root):
    python src/__main__.py prepare-dataset
    python src/__main__.py train
    python src/__main__.py predict path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import Application


def _prepare_dataset(app: Application, _args: argparse.Namespace) -> None:
    app.prepare_dataset()


def _train(app: Application, _args: argparse.Namespace) -> None:
    app.train()


def _predict(app: Application, args: argparse.Namespace) -> None:
    result = app.predict(Path(args.video_path))
    if result is None:
        print("no motion detected in clip; no prediction made")
        return
    confidence = result.class_probabilities[result.label_index]
    print(f"predicted intention: {result.label} (confidence {confidence:.2%})")
    print(f"class probabilities: {result.class_probabilities}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="activity-recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="run background subtraction, sampling and augmentation over raw clips"
    )
    prepare_parser.set_defaults(handler=_prepare_dataset)

    train_parser = subparsers.add_parser("train", help="train and save the AlexNet model")
    train_parser.set_defaults(handler=_train)

    predict_parser = subparsers.add_parser("predict", help="classify a single video clip")
    predict_parser.add_argument("video_path", help="path to the video clip to classify")
    predict_parser.set_defaults(handler=_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = Application()
    args.handler(app, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
