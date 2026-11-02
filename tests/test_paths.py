"""The layout constants other modules depend on."""

from __future__ import annotations

from src import paths


def test_root_contains_expected_layout() -> None:
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert (paths.ROOT / "src").is_dir()


def test_interval_constants_are_consistent() -> None:
    assert paths.DT_HOURS * paths.INTERVALS_PER_DAY_5MIN == 24.0
    assert paths.INTERVALS_PER_DAY_5MIN == 6 * paths.INTERVALS_PER_DAY_30MIN
