from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return Path(__file__).parent / "fixtures"
