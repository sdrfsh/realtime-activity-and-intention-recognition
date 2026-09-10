from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from config import LiveSettings, NetworkSettings, SamplingSettings, TrackingSettings
from data.frame_source import FrameSource
from domain.entities import DoorApproach, LiveDecision, PredictionResult
from modeling.activity_predictor import ActivityPredictor
from modeling.network_input_formatter import NetworkInputFormatter
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.door_approach_decider import DoorApproachDecider
from preprocessing.frame_window import FrameWindow
from preprocessing.mask_cleaner import MaskCleaner
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.motion_trigger import MotionTrigger
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.optical_flow_estimator import OpticalFlowEstimator
from pipelines.window_clip_pipeline import WindowClipPipeline


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
        predictor: ActivityPredictor | None,
        decision_sink: DecisionSink | None = None,
        motion_sink: MotionSink | None = None,
        decide: bool = True,
        door_decider: DoorApproachDecider | None = None,
    ) -> None:
        self._source = source
        self._tracking = tracking
        self._live = live
        self._predictor = predictor
        self._decision_sink = decision_sink
        self._motion_sink = motion_sink
        self._decide = decide
        self._door_decider = door_decider
        self._heading_mode = live.decision_mode == "heading"
        if self._heading_mode and door_decider is None:
            raise ValueError("heading decision mode needs a DoorApproachDecider")
        if not self._heading_mode and predictor is None and decide:
            raise ValueError("network decision mode needs an ActivityPredictor")
        self._noise_reducer = NoiseReducer(tracking)
        self._background_subtractor = BackgroundSubtractor(tracking)
        self._mask_cleaner = MaskCleaner(tracking)
        if self._heading_mode:
            self._class_names: list[str] = list(DoorApproachDecider.CLASSES)
        else:
            self._class_names = list(getattr(predictor, "classes", []) or [])
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
        # Kept for the preview: the exact tensor last fed to the network and
        # the decision it produced, plus the most recent foreground mask.
        self._last_network_input: np.ndarray | None = None
        self._last_decision: LiveDecision | None = None
        self._last_mask: np.ndarray | None = None
        self._last_approach: DoorApproach | None = None

    def _measure_fps(self) -> float:
        measured = getattr(self._source, "measure_fps", None)
        if callable(measured):
            return max(float(measured(self._live.fps_probe_frames)), 1.0)
        return max(float(self._source.fps() or 30.0), 1.0)

    def warm_up(self, seconds: float | None = None) -> None:
        duration = self._live.warmup_seconds if seconds is None else seconds
        deadline = time.monotonic() + duration
        for frame in self._source.frames():
            self._background_subtractor.apply(self._noise_reducer.apply(frame))
            if time.monotonic() >= deadline:
                break

    def step(self, frame: np.ndarray) -> LiveDecision | None:
        clean_frame = self._noise_reducer.apply(frame)
        mask = self._mask_cleaner.apply(self._background_subtractor.apply(clean_frame))
        self._last_mask = mask
        now = time.monotonic()
        started = self._trigger.started_at
        triggered = self._trigger.update(mask, now)
        if triggered:
            started = self._trigger.started_at
            self._window.clear()
            if self._motion_sink is not None:
                self._motion_sink()
        if self._trigger.collecting:
            self._window.append(frame, mask)
        if not self._trigger.collecting or not self._window.is_full:
            return None
        raw_frames, masks = self._window.raw(), self._window.masks()
        motion_image = self._window_pipeline.process(raw_frames, masks)
        self._window.clear()
        self._trigger.finish(now)
        if motion_image is None:
            return None
        network_input = self._formatter.format(motion_image.image)
        self._last_network_input = network_input
        if not self._decide:
            return None
        if self._heading_mode:
            approach = self._door_decider.decide(masks)
            if approach is None:
                return None
            self._last_approach = approach
            label = approach.label
            class_probabilities = np.array([1.0 - approach.entering_score, approach.entering_score], dtype=np.float32)
            confidence = float(class_probabilities[self._class_names.index(label)])
        else:
            prediction: PredictionResult = self._predictor.predict(network_input)
            label = prediction.label
            class_probabilities = prediction.class_probabilities
            confidence = float(prediction.class_probabilities[prediction.label_index])
        decided_at = time.monotonic()
        decision = LiveDecision(
            label=label,
            confidence=confidence,
            class_probabilities=class_probabilities,
            started_at=started or decided_at,
            decided_at=decided_at,
            latency_s=decided_at - (started or decided_at),
        )
        self._last_decision = decision
        if self._decision_sink is not None:
            self._decision_sink(decision)
        return decision

    # ------------------------------------------------------------------ preview
    _ENTERING_COLOR = (0, 220, 0)
    _PASSING_COLOR = (0, 165, 255)

    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        """Left: the camera frame with the trigger state and the network's
        output (label, confidence and per-class probabilities). Right column:
        the live cleaned mask on top and the exact 227x227 tensor last fed to
        the network below."""
        camera = frame.copy()
        height = camera.shape[0]
        if self._heading_mode:
            self._draw_door_edge(camera)
        self._draw_status(camera)
        self._draw_network_output(camera)

        panel_height = height // 2
        mask_panel = self._mask_panel(panel_height)
        input_panel = self._network_input_panel(panel_height)
        column_width = max(mask_panel.shape[1], input_panel.shape[1])
        column = np.vstack(
            (
                self._pad_width(mask_panel, column_width),
                self._pad_width(input_panel, column_width),
            )
        )
        if column.shape[0] != height:  # odd frame heights
            column = self._pad_height(column, height)
        return np.hstack((camera, column))

    def _draw_status(self, camera: np.ndarray) -> None:
        pixels = 0 if self._last_mask is None else int(np.count_nonzero(self._last_mask))
        if self._trigger.collecting:
            status = f"collecting {len(self._window)}/{self._window.capacity}"
            color = (0, 200, 255)
        else:
            status = "idle"
            color = (200, 200, 200)
        self._put_text(camera, f"{status}   fg px {pixels} / thr {self._live.motion_pixel_threshold}", (10, 28), color, 0.6)

    _DOOR_COLOR = (255, 200, 0)

    def _draw_door_edge(self, camera: np.ndarray) -> None:
        """Highlight the frame edge the door lies along."""
        height, width = camera.shape[:2]
        thickness = max(4, min(height, width) // 60)
        side = self._door_decider.door_side
        if side == "top":
            cv2.rectangle(camera, (0, 0), (width - 1, thickness), self._DOOR_COLOR, -1)
            origin = (width // 2 - 30, thickness + 22)
        elif side == "bottom":
            cv2.rectangle(camera, (0, height - 1 - thickness), (width - 1, height - 1), self._DOOR_COLOR, -1)
            origin = (width // 2 - 30, height - thickness - 8)
        elif side == "left":
            cv2.rectangle(camera, (0, 0), (thickness, height - 1), self._DOOR_COLOR, -1)
            origin = (thickness + 8, height // 2)
        else:
            cv2.rectangle(camera, (width - 1 - thickness, 0), (width - 1, height - 1), self._DOOR_COLOR, -1)
            origin = (width - thickness - 70, height // 2)
        self._put_text(camera, "door", origin, self._DOOR_COLOR, 0.6, 2)

    def _draw_network_output(self, camera: np.ndarray) -> None:
        height, width = camera.shape[:2]
        decision = self._last_decision
        if decision is None:
            source = "heading" if self._heading_mode else "network"
            self._put_text(camera, f"{source}: no decision yet", (10, height - 16), (200, 200, 200), 0.6)
            return
        color = self._label_color(decision.label)
        self._put_text(camera, f"{decision.label}  {decision.confidence:.0%}", (10, height - 52), color, 1.0, 2)
        # per-class probability bars along the bottom edge
        probabilities = np.asarray(decision.class_probabilities, dtype=np.float32).ravel()
        bar_left, bar_top, bar_height, gap = 10, height - 36, 10, 4
        bar_width = max(60, min(220, width // 3))
        for index, probability in enumerate(probabilities):
            name = self._class_names[index] if index < len(self._class_names) else f"class {index}"
            y = bar_top + index * (bar_height + gap)
            cv2.rectangle(camera, (bar_left, y), (bar_left + bar_width, y + bar_height), (40, 40, 40), -1)
            fill = int(bar_width * float(np.clip(probability, 0.0, 1.0)))
            cv2.rectangle(camera, (bar_left, y), (bar_left + fill, y + bar_height), self._label_color(name), -1)
            self._put_text(camera, f"{name} {probability:.0%}", (bar_left + bar_width + 8, y + bar_height), (255, 255, 255), 0.45, 1)

    def _mask_panel(self, panel_height: int) -> np.ndarray:
        if self._last_mask is None:
            panel = np.zeros((panel_height, panel_height * 16 // 9, 3), dtype=np.uint8)
        else:
            mask_h, mask_w = self._last_mask.shape[:2]
            panel_width = max(1, round(panel_height * mask_w / mask_h))
            panel = cv2.cvtColor(
                cv2.resize(self._last_mask, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )
        self._put_text(panel, "cleaned mask (live)", (10, 24), (255, 255, 255), 0.55)
        return panel

    def _network_input_panel(self, panel_height: int) -> np.ndarray:
        if self._last_network_input is None:
            panel = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
            title = "network input (none yet)"
        else:
            panel = cv2.resize(
                np.clip(self._last_network_input, 0, 255).astype(np.uint8),
                (panel_height, panel_height),
                interpolation=cv2.INTER_NEAREST,
            )
            title = "network input"
        self._put_text(panel, title, (10, 24), (255, 255, 255), 0.55)
        return panel

    @classmethod
    def _label_color(cls, label: str) -> tuple[int, int, int]:
        return cls._ENTERING_COLOR if label == "entering" else cls._PASSING_COLOR

    @staticmethod
    def _pad_width(image: np.ndarray, width: int) -> np.ndarray:
        extra = width - image.shape[1]
        return image if extra <= 0 else cv2.copyMakeBorder(image, 0, 0, 0, extra, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    @staticmethod
    def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
        extra = height - image.shape[0]
        if extra < 0:
            return image[:height]
        return image if extra == 0 else cv2.copyMakeBorder(image, 0, extra, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.65,
        thickness: int = 1,
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def run_until_decision(self) -> LiveDecision | None:
        for frame in self._source.frames():
            decision = self.step(frame)
            if decision is not None:
                return decision
        return None

    def run(self) -> None:
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
                decision = self.step(frame)
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