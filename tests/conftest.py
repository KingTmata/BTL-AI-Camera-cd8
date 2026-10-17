"""Classify test suites without skipping or filtering any collected test."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        relative = Path(item.path).relative_to(config.rootpath)
        marker = pytest.mark.integration if "integration" in relative.parts else pytest.mark.unit
        item.add_marker(marker)
