"""Central, overridable configuration for the whole pipeline.

Every hyperparameter here traces back to a decision documented in
``legacy/3. Methodology v1.1.docx`` (see the docstring on each field group).
Nothing in the rest of the codebase should hardcode a path or a magic
number that belongs here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathSettings:
    """Filesystem layout. All paths are created on demand, never assumed to exist."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(init=False)
    raw_video_dir: Path = field(init=False)
    background_removed_dir: Path = field(init=False)
    motion_image_dir: Path = field(init=False)
    augmented_image_dir: Path = field(init=False)
    labels_csv: Path = field(init=False)
    filtered_labels_csv: Path = field(init=False)
    training_csv: Path = field(init=False)
    models_dir: Path = field(init=False)
    trained_model_path: Path = field(init=False)
    label_map_path: Path = field(init=False)

    def __post_init__(self) -> None:
        data_dir = self.project_root / "data"
        models_dir = self.project_root / "models"
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "raw_video_dir", data_dir / "raw_videos")
        object.__setattr__(self, "background_removed_dir", data_dir / "background_removed")
        object.__setattr__(self, "motion_image_dir", data_dir / "motion_images")
        object.__setattr__(self, "augmented_image_dir", data_dir / "augmented_images")
        object.__setattr__(self, "labels_csv", data_dir / "labeled_data.csv")
        object.__setattr__(self, "filtered_labels_csv", data_dir / "contour_filtered.csv")
        object.__setattr__(self, "training_csv", data_dir / "trained_data.csv")
        object.__setattr__(self, "models_dir", models_dir)
        object.__setattr__(self, "trained_model_path", models_dir / "alexnet_model.keras")
        object.__setattr__(self, "label_map_path", models_dir / "label_map.json")

    def ensure_exists(self) -> None:
        for directory in (
            self.data_dir,
            self.raw_video_dir,
            self.background_removed_dir,
            self.motion_image_dir,
            self.augmented_image_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrackingSettings:
    """Docx 3.1 "Tracking": noise removal + KNN background subtraction + contour presence check."""

    frame_size: tuple[int, int] = (160, 90)  # (width, height), matches BackgroundSubtraction v1.8
    gaussian_kernel: tuple[int, int] = (15, 15)
    morphology_iterations: int = 2
    detect_shadows: bool = False
    canny_thresholds: tuple[int, int] = (5, 255)
    video_fps: int = 28
    video_fourcc: str = "MP4V"
    # KNN background model. ``knn_dist2_threshold`` is the squared intensity
    # distance a pixel must move before it counts as foreground (OpenCV default
    # 400); raising it suppresses sensor noise and small exposure drift.
    # ``knn_history`` is how many frames a still object takes to be absorbed
    # into the background (OpenCV default 500 = ~17 s at 30 fps).
    knn_history: int = 500
    knn_dist2_threshold: float = 800.0
    # Mask cleanup after subtraction (all at ``frame_size`` resolution), in
    # order: median blur kills isolated specks; opening removes thin noise;
    # closing (``mask_close_kernel`` px) fills holes inside the silhouette;
    # components below ``mask_min_component_area`` px are dropped. The last
    # two stages grow the blob and are off by default: ``mask_dilate_iterations``
    # expands survivors, and ``mask_fill_mode`` paints each blob's convex
    # hull ("hull"), fills only its interior holes ("contour"), or does
    # nothing ("none").
    mask_median_kernel: int = 5
    mask_open_iterations: int = 1
    mask_close_kernel: int = 5
    mask_close_iterations: int = 2
    mask_min_component_area: int = 30
    mask_dilate_iterations: int = 0
    mask_fill_mode: str = "none"


@dataclass(frozen=True)
class SamplingSettings:
    """Docx 3.2 "Sampling": velocity-adaptive frame rate + recency-weighted frame summation.

    ``decay_weight`` is the 70% weight the docx assigns to the running
    accumulation each time a new (more recent) frame is folded in, so older
    frames fade out. ``trim_start``/``trim_end`` drop the first/last three
    sampled frames per clip, as specified in the docx, to reduce training noise.
    """

    max_samples_per_clip: int = 15
    trim_start: int = 3
    trim_end: int = 3
    decay_weight: float = 0.7
    optical_flow_pyr_scale: float = 0.5
    optical_flow_levels: int = 1
    optical_flow_winsize: int = 15
    optical_flow_iterations: int = 2
    optical_flow_poly_n: int = 5
    optical_flow_poly_sigma: float = 1.1


@dataclass(frozen=True)
class NetworkSettings:
    """Docx 3.3 "Network Implementation and Training": shallow AlexNet, tuned for embedded inference."""

    input_shape: tuple[int, int, int] = (227, 227, 3)
    num_classes: int = 2
    # Inference model: a local ``.keras`` path or ``hf://namespace/repo[/file]``.
    # Defaults to the published pretrained classifier; ``train`` writes its
    # own model to ``PathSettings.trained_model_path`` and ``--model`` selects it.
    model_source: str = "hf://sdrfsh/alexnet-door-entry-classifier"
    model_filename: str = "alexnet.keras"
    # Output-index order of the pretrained Hub model (0 = passing by, 1 = entering).
    # Locally trained models carry their own order in ``label_map.json``.
    pretrained_classes: tuple[str, ...] = ("passing_by", "entering")
    hidden_activation: str = "tanh"
    output_activation: str = "softmax"
    kernel_initializer: str = "glorot_normal"
    learning_rate: float = 0.01
    momentum: float = 0.9
    batch_size: int = 64
    epochs: int = 30
    train_ratio: float = 0.6
    validation_ratio: float = 0.3
    test_ratio: float = 0.1


DOOR_SIDES: tuple[str, ...] = ("right", "left", "top", "bottom")


@dataclass(frozen=True)
class SceneSettings:
    """Where the door sits in the camera image and how a door approach is
    recognised from the subject's heading (``DoorApproachDecider``).

    Purely 2-D: the door is one of the four frame edges and depth along the
    camera axis is not modelled. The subject's centroid is tracked across the
    window; if it travels at least ``approach_min_travel`` (a fraction of the
    frame size) toward ``door_side``, with a heading within
    ``approach_max_angle_deg`` of that edge's normal, the decision is
    "entering", otherwise "passing_by". ``trajectory_smoothing_frames``
    centroids are averaged at each end of the window to suppress mask jitter.
    Masks with fewer than ``min_foreground_pixels`` are ignored, as are masks
    where more than ``max_foreground_fraction`` of the frame is foreground
    (the background model has not settled, or the exposure just jumped).
    """

    door_side: str = "right"
    approach_min_travel: float = 0.10
    approach_max_angle_deg: float = 60.0
    trajectory_smoothing_frames: int = 3
    min_foreground_pixels: int = 30
    max_foreground_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.door_side not in DOOR_SIDES:
            raise ValueError(f"door_side must be one of {DOOR_SIDES}, got {self.door_side!r}")
        if not 0.0 < self.approach_max_angle_deg < 180.0:
            raise ValueError("approach_max_angle_deg must be between 0 and 180 (exclusive)")


DECISION_MODES: tuple[str, ...] = ("heading", "network")


@dataclass(frozen=True)
class LiveSettings:
    camera_source: int | str = 0
    window_seconds: float = 3.0
    warmup_seconds: float = 3.0
    cooldown_seconds: float = 2.5
    # Foreground pixels (of 160x90 = 14,400) needed to start a window. A quiet
    # webcam scene still shows ~400-800 from sensor noise, and auto-exposure
    # swings reach several thousand, so this demands ~14% of the frame to
    # change; lower it for a fixed, exposure-locked camera.
    motion_pixel_threshold: int = 2000
    fps_probe_frames: int = 60
    queue_max_frames: int = 30
    preview: bool = True
    # "heading": decide from the tracked subject's direction relative to
    # ``SceneSettings.door_side``. "network": classify the motion image with
    # the AlexNet model (ignores the door side).
    decision_mode: str = "heading"

    def __post_init__(self) -> None:
        if self.decision_mode not in DECISION_MODES:
            raise ValueError(f"decision_mode must be one of {DECISION_MODES}, got {self.decision_mode!r}")


@dataclass(frozen=True)
class Settings:
    paths: PathSettings = field(default_factory=PathSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    scene: SceneSettings = field(default_factory=SceneSettings)
    live: LiveSettings = field(default_factory=LiveSettings)


def load_settings() -> Settings:
    """Single entry point for obtaining configuration. Swap this out for a
    file/env-based loader later without touching any call site."""
    return Settings()
