from pathlib import Path

from data.label_map_repository import LabelMapRepository


def test_load_returns_empty_list_when_file_missing(tmp_path: Path):
    repository = LabelMapRepository(tmp_path / "missing.json")
    assert repository.load() == []


def test_save_then_load_round_trips_class_order(tmp_path: Path):
    repository = LabelMapRepository(tmp_path / "nested" / "label_map.json")
    repository.save(["entering", "passing_by"])

    assert repository.load() == ["entering", "passing_by"]
