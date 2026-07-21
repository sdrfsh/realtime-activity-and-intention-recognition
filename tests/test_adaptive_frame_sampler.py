from config import SamplingSettings
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler


def _sampler(**overrides) -> AdaptiveFrameSampler:
    return AdaptiveFrameSampler(SamplingSettings(**overrides))


def test_faster_velocity_yields_smaller_sampling_rate():
    sampler = _sampler(max_samples_per_clip=15)
    slow_rate = sampler.compute_sampling_rate(frame_count=150, velocity=0.5)
    fast_rate = sampler.compute_sampling_rate(frame_count=150, velocity=2.0)
    assert fast_rate < slow_rate


def test_zero_velocity_falls_back_to_fixed_rate():
    sampler = _sampler(max_samples_per_clip=15)
    rate = sampler.compute_sampling_rate(frame_count=150, velocity=0.0)
    assert rate == 150 // 15


def test_sampling_rate_is_never_less_than_one():
    sampler = _sampler(max_samples_per_clip=15)
    rate = sampler.compute_sampling_rate(frame_count=10, velocity=100.0)
    assert rate >= 1


def test_select_frame_indices_caps_at_max_samples():
    sampler = _sampler(max_samples_per_clip=5, trim_start=0, trim_end=0)
    indices = sampler.select_frame_indices(frame_count=100, sampling_rate=1)
    assert indices == [0, 1, 2, 3, 4]


def test_select_frame_indices_trims_start_and_end():
    sampler = _sampler(max_samples_per_clip=15, trim_start=3, trim_end=3)
    indices = sampler.select_frame_indices(frame_count=150, sampling_rate=10)
    full = list(range(0, 150, 10))[:15]
    assert indices == full[3 : len(full) - 3]


def test_select_frame_indices_skips_trim_when_too_few_samples():
    sampler = _sampler(max_samples_per_clip=15, trim_start=3, trim_end=3)
    indices = sampler.select_frame_indices(frame_count=20, sampling_rate=15)
    assert indices == [0, 15]
