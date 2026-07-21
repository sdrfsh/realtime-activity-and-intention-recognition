import random
from pathlib import Path

from config import NetworkSettings
from modeling.dataset_splitter import DatasetSplitter


def _samples(n: int) -> list[tuple[Path, str]]:
    return [(Path(f"clip_{i}.jpg"), "walk" if i % 2 == 0 else "run") for i in range(n)]


def test_split_respects_configured_ratios():
    settings = NetworkSettings(train_ratio=0.6, validation_ratio=0.3, test_ratio=0.1)
    splitter = DatasetSplitter(settings, rng=random.Random(42))

    result = splitter.split(_samples(100))

    assert len(result.train) == 60
    assert len(result.validation) == 30
    assert len(result.test) == 10


def test_split_partitions_all_samples_with_no_overlap():
    settings = NetworkSettings()
    splitter = DatasetSplitter(settings, rng=random.Random(7))

    samples = _samples(37)
    result = splitter.split(samples)

    reconstructed = set(result.train) | set(result.validation) | set(result.test)
    assert reconstructed == set(samples)
    assert len(result.train) + len(result.validation) + len(result.test) == len(samples)


def test_split_is_deterministic_given_same_rng_seed():
    settings = NetworkSettings()
    samples = _samples(50)

    result_a = DatasetSplitter(settings, rng=random.Random(123)).split(samples)
    result_b = DatasetSplitter(settings, rng=random.Random(123)).split(samples)

    assert result_a.train == result_b.train
    assert result_a.validation == result_b.validation
    assert result_a.test == result_b.test
