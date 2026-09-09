from __future__ import annotations

from pathlib import Path

import pytest

from animation_converter.detection import detect_format
from animation_converter.models import SourceFormat
from animation_converter.registry import default_registry


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("ostim_sa", SourceFormat.OSTIM_SA),
        ("osa_namespaced", SourceFormat.OSA_OSEX),
        ("slal", SourceFormat.SLAL),
        ("flowergirls", SourceFormat.FLOWERGIRLS),
    ],
)
def test_detects_format_from_layout_and_metadata(fixture_root: Path, fixture_name: str, expected: SourceFormat) -> None:
    selected, _ = detect_format(fixture_root / fixture_name, default_registry())
    assert selected.proposed_format == expected
    assert selected.confidence >= 0.8
    assert selected.evidence
    assert selected.files_inspected


def test_does_not_infer_format_from_directory_name(tmp_path: Path) -> None:
    misleading = tmp_path / "Definitely_OStim_Scenes"
    misleading.mkdir()
    selected, _ = detect_format(misleading, default_registry())
    assert selected.proposed_format == SourceFormat.AUTO
    assert selected.confidence == 0


def test_reports_conflicting_evidence(fixture_root: Path, tmp_path: Path) -> None:
    mixed = tmp_path / "mixed"
    (mixed / "one").mkdir(parents=True)
    (mixed / "two").mkdir(parents=True)
    source_scene = next((fixture_root / "ostim_sa").rglob("ExampleIdle.json"))
    source_slal = next((fixture_root / "slal").rglob("FixturePack.json"))
    (mixed / "one" / "ExampleIdle.json").write_bytes(source_scene.read_bytes())
    (mixed / "two" / "FixturePack.json").write_bytes(source_slal.read_bytes())

    selected, candidates = detect_format(mixed, default_registry())
    assert selected.proposed_format in {SourceFormat.OSTIM_SA, SourceFormat.SLAL}
    assert selected.conflicting_evidence
    assert sum(candidate.confidence > 0 for candidate in candidates) >= 2
