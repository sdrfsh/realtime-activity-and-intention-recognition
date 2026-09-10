"""Command-line entry point.

Usage (run from the project root):
    python src/__main__.py prepare-dataset
    python src/__main__.py train
    python src/__main__.py predict path/to/clip.mp4
    python src/__main__.py live --camera 0
    python src/__main__.py live --model models/alexnet_model.keras   # locally trained model
    python src/__main__.py live --door-side top                      # door along the top edge of the frame
    python src/__main__.py live --decision network                   # classify with the AlexNet model instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import Application
from config import DECISION_MODES, DOOR_SIDES


def _prepare_dataset(app: Application, _args: argparse.Namespace) -> None:
    app.prepare_dataset()


def _train(app: Application, _args: argparse.Namespace) -> None:
    app.train()


def _predict(app: Application, args: argparse.Namespace) -> None:
    result = app.predict(
        Path(args.video_path), model_source=args.model, door_side=args.door_side, decision_mode=args.decision
    )
    if result is None:
        print("no motion detected in clip; no prediction made")
        return
    confidence = result.class_probabilities[result.label_index]
    print(f"predicted intention: {result.label} (confidence {confidence:.2%})")
    print(f"class probabilities: {result.class_probabilities}")


def _camera_value(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _live(app: Application, args: argparse.Namespace) -> None:
    app.live(
        camera_source=_camera_value(args.camera),
        window_seconds=args.window,
        motion_threshold=args.motion_threshold,
        preview=not args.no_preview,
        decide=not args.no_decide,
        model_source=args.model,
        door_side=args.door_side,
        decision_mode=args.decision,
    )


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "inference model: a local .keras path (e.g. models/alexnet_model.keras, "
            "as written by `train`) or hf://namespace/repo[/file]. Default: the "
            "pretrained Hub classifier from NetworkSettings.model_source"
        ),
    )


def _add_decision_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--door-side",
        choices=DOOR_SIDES,
        default=None,
        help=(
            "which edge of the camera image the door lies along; a subject heading toward "
            "that edge is 'entering', anything else is 'passing_by' "
            "(default: SceneSettings.door_side = right)"
        ),
    )
    parser.add_argument(
        "--decision",
        choices=DECISION_MODES,
        default=None,
        help=(
            "'heading' decides from the tracked subject's direction relative to --door-side; "
            "'network' classifies the motion image with the AlexNet model and ignores the door side "
            "(default: LiveSettings.decision_mode = heading)"
        ),
    )


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
    _add_model_argument(predict_parser)
    _add_decision_arguments(predict_parser)
    predict_parser.set_defaults(handler=_predict)

    live_parser = subparsers.add_parser("live", help="run continuous camera inference")
    live_parser.add_argument("--camera", default="0", help="camera index or capture URL")
    live_parser.add_argument("--window", type=float, default=3.0, help="window length in seconds")
    live_parser.add_argument(
        "--motion-threshold",
        type=int,
        default=None,
        help="foreground pixels needed to start a window (default: LiveSettings.motion_pixel_threshold)",
    )
    live_parser.add_argument("--no-preview", action="store_true")
    live_parser.add_argument("--no-decide", action="store_true")
    _add_model_argument(live_parser)
    _add_decision_arguments(live_parser)
    live_parser.set_defaults(handler=_live)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = Application()
    args.handler(app, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
