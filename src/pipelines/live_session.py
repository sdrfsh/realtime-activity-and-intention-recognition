"""The continuous camera path: frames stream in from a ``FrameSource``, each
one is tracked into a foreground mask, a motion trigger opens a fixed-length
window, and when the window is full one decision is made for it, either from
the subject's heading toward the door edge (``DoorApproachDecider``) or by
classifying the window's motion image (``ActivityPredictor``).

Owns one stateful KNN background model for the whole session. Capture runs
on its own thread into a bounded queue; all processing happens on the
caller's thread in :meth:`run`.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from config import LiveSettings, NetworkSettings, SamplingSettings, TrackingSettings
from data.frame_source import FrameSource
from domain.entities import LiveDecision
from modeling.activity_predictor import ActivityPredictor
from modeling.door_approach_decider import DoorApproachDecider
from modeling.network_input_formatter import NetworkInputFormatter
from pipelines.live_preview import LivePreview
from pipelines.window_clip_pipeline import WindowClipPipeline
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.frame_window import FrameWindow
from preprocessing.mask_cleaner import MaskCleaner
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.motion_trigger import MotionTrigger
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.optical_flow_estimator import OpticalFlowEstimator

DecisionSink = Callable[[LiveDecision], None]
MotionSink = Callable[[], None]


class LiveSession:
    def __init__(
        self,
        source: FrameSource,
        tracking: TrackingSettings,
        sampling: SamplingSettings,
        network: NetworkSettings,
        live: LiveSettings,
        predictor: ActivityPredictor | None = None,
        door_decider: DoorApproachDecider | None = None,
        decision_sink: DecisionSink | None = None,
        motion_sink: MotionSink | None = None,
        decide: bool = True,
    ) -> None:
        self._heading_mode = live.decision_mode == "heading"
        if self._heading_mode and door_decider is None:
            raise ValueError("heading decision mode needs a DoorApproachDecider")
        if not self._heading_mode and predictor is None and decide:
            raise ValueError("network decision mode needs an ActivityPredictor")

        self._source = source
        self._live = live
        self._predictor = predictor
        self._door_decider = door_decider
        self._decision_sink = decision_sink
        self._motion_sink = motion_sink
        self._decide = decide

        self._noise_reducer = NoiseReducer(tracking)
        self._background_subtractor = BackgroundSubtractor(tracking)
        self._mask_cleaner = MaskCleaner(tracking)
        self._trigger = MotionTrigger(live)
        self._fps = self._measure_fps()
        self._window = FrameWindow(max(2, round(self._fps * live.window_seconds)))
        self._window_pipeline = WindowClipPipeline(
            MotionContourDetector(tracking),
            OpticalFlowEstimator(sampling),
            AdaptiveFrameSampler(sampling),
            MotionImageEncoder(sampling),
        )
        self._formatter = NetworkInputFormatter(network)
        self._stop_event = threading.Event()

        if self._heading_mode:
            self._class_names = list(DoorApproachDecider.CLASSES)
        else:
            self._class_names = list(getattr(predictor, "classes", []) or [])
        self._preview = LivePreview(
            class_names=self._class_names,
            motion_pixel_threshold=live.motion_pixel_threshold,
            decision_source="heading" if self._heading_mode else "network",
            door_side=door_decider.door_side if self._heading_mode else None,
        )
        # Kept for the preview: the latest mask, the last tensor that would
        # be fed to the network, and the last decision.
        self._last_mask: np.ndarray | None = None
        self._last_network_input: np.ndarray | None = None
        self._last_decision: LiveDecision | None = None

    def _measure_fps(self) -> float:
        measured = getattr(self._source, "measure_fps", None)
        if callable(measured):
            return max(float(measured(self._live.fps_probe_frames)), 1.0)
        return max(float(self._source.fps() or 30.0), 1.0)

    # ------------------------------------------------------------------ processing
    def warm_up(self, seconds: float | None = None) -> None:
        """Feed the background model for a while so the scene is learnt
        before any motion can trigger a window."""
        duration = self._live.warmup_seconds if seconds is None else seconds
        deadline = time.monotonic() + duration
        for frame in self._source.frames():
            self._background_subtractor.apply(self._noise_reducer.apply(frame))
            if time.monotonic() >= deadline:
                break

    def step(self, frame: np.ndarray) -> LiveDecision | None:
        """Process one frame; returns a decision only on the frame that
        completes a window."""
        mask = self._mask_cleaner.apply(self._background_subtractor.apply(self._noise_reducer.apply(frame)))
        self._last_mask = mask
        now = time.monotonic()
        started = self._trigger.started_at
        if self._trigger.update(mask, now):
            started = self._trigger.started_at
            self._window.clear()
            if self._motion_sink is not None:
                self._motion_sink()
        if self._trigger.collecting:
            self._window.append(frame, mask)
        if not self._trigger.collecting or not self._window.is_full:
            return None

        raw_frames, masks = self._window.raw(), self._window.masks()
        self._window.clear()
        self._trigger.finish(now)
        motion_image = self._window_pipeline.process(raw_frames, masks)
        if motion_image is None:
            return None
        self._last_network_input = self._formatter.format(motion_image.image)
        if not self._decide:
            return None

        decision = self._decide_window(masks, self._last_network_input, started)
        if decision is not None:
            self._last_decision = decision
            if self._decision_sink is not None:
                self._decision_sink(decision)
        return decision

    def _decide_window(
        self, masks: list[np.ndarray], network_input: np.ndarray, started: float | None
    ) -> LiveDecision | None:
        if self._heading_mode:
            approach = self._door_decider.decide(masks)
            if approach is None:
                return None
            label = approach.label
            probabilities = np.array([1.0 - approach.entering_score, approach.entering_score], dtype=np.float32)
            confidence = float(probabilities[self._class_names.index(label)])
        else:
            prediction = self._predictor.predict(network_input)
            label = prediction.label
            probabilities = prediction.class_probabilities
            confidence = float(probabilities[prediction.label_index])
        decided_at = time.monotonic()
        started_at = started if started is not None else decided_at
        return LiveDecision(
            label=label,
            confidence=confidence,
            class_probabilities=probabilities,
            started_at=started_at,
            decided_at=decided_at,
            latency_s=decided_at - started_at,
        )

    # ------------------------------------------------------------------ running
    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        collecting = (len(self._window), self._window.capacity) if self._trigger.collecting else None
        return self._preview.render(frame, self._last_mask, self._last_network_input, self._last_decision, collecting)

    def run_until_decision(self) -> LiveDecision | None:
        """Single-threaded: consume the source until the first decision (used by ``predict``)."""
        for frame in self._source.frames():
            decision = self.step(frame)
            if decision is not None:
                return decision
        return None

    def run(self) -> None:
        """Capture on a background thread, process (and preview) here until
        the source ends, ``q`` is pressed in the preview, or :meth:`stop`."""
        frames: deque[np.ndarray] = deque(maxlen=self._live.queue_max_frames)
        condition = threading.Condition()

        def capture() -> None:
            try:
                for frame in self._source.frames():
                    with condition:
                        frames.append(frame)
                        condition.notify()
                    if self._stop_event.is_set():
                        break
            finally:
                with condition:
                    self._stop_event.set()
                    condition.notify_all()

        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        try:
            while not self._stop_event.is_set() or frames:
                with condition:
                    condition.wait_for(lambda: bool(frames) or self._stop_event.is_set())
                    if not frames:
                        continue
                    frame = frames.popleft()
                self.step(frame)
                if self._live.preview:
                    cv2.imshow("activity recognition", self.render_preview(frame))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop()
        finally:
            self.stop()
            thread.join(timeout=2.0)
            cv2.destroyAllWindows()

    def stop(self) -> None:
        self._stop_event.set()
        self._source.close()
