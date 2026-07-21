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


@dataclass(frozen=True)
class Settings:
    paths: PathSettings = field(default_factory=PathSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)


def load_settings() -> Settings:
    """Single entry point for obtaining configuration. Swap this out for a
    file/env-based loader later without touching any call site."""
    return Settings()
