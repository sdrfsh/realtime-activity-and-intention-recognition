"""Measure frozen-model and optical-flow inference costs on the target box."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from config import load_settings  # noqa: E402
from modeling.activity_predictor import ActivityPredictor  # noqa: E402
from modeling.network_input_formatter import NetworkInputFormatter  # noqa: E402
from preprocessing.optical_flow_estimator import OpticalFlowEstimator  # noqa: E402


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(values, value) * 1000)


def main() -> None:
    settings = load_settings()
    predictor = ActivityPredictor(
        settings.network,
        settings.network.model_source,
        settings.network.pretrained_classes,
    )
    formatter = NetworkInputFormatter(settings.network)
    flow = OpticalFlowEstimator(settings.sampling)
    image = np.zeros((90, 160), dtype=np.uint8)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    predict_times: list[float] = []
    flow_times: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        predictor.predict(formatter.format(image))
        predict_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        flow.estimate_velocity(frame, frame)
        flow_times.append(time.perf_counter() - start)
    print(f"predict p50={percentile(predict_times, 50):.1f}ms p95={percentile(predict_times, 95):.1f}ms")
    print(f"optical flow p50={percentile(flow_times, 50):.1f}ms p95={percentile(flow_times, 95):.1f}ms")


if __name__ == "__main__":
    main()
