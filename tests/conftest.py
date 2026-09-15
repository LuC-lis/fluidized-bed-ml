"""Pytest configuration: make ``src/`` importable and register the slow marker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs the multi-hundred-MB raw recordings (set FBFML_SLOW_TESTS=1)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if os.environ.get("FBFML_SLOW_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="slow test: set FBFML_SLOW_TESTS=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
