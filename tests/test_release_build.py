from __future__ import annotations

from tools import release_check


def test_release_build_configuration_is_isolated_and_hardened() -> None:
    release_check.check_release_build_configuration()


def test_release_requirements_are_exactly_pinned() -> None:
    entries = [
        line.strip()
        for line in release_check.RELEASE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert entries
    assert all(entry.count("==") == 1 for entry in entries)
    assert len({entry.split("==", 1)[0].lower() for entry in entries}) == len(entries)
