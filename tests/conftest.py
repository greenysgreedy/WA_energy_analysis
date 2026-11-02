"""Shared pytest configuration.

Tests that touch real data under ``data/`` are marked ``slow`` and skipped unless
``--run-slow`` is passed, so the toy-day dispatch tests stay fast enough to run
before every real-data execution.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests that read real AEMO data from data/",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow and data in data/processed/")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
