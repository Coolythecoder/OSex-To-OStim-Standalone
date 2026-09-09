from __future__ import annotations

import json
from pathlib import Path

from animation_converter.cli import main
from animation_converter.service import EXIT_SUCCESS, EXIT_WARNINGS


def test_schema_dump_reports_pinned_commit(capsys) -> None:
    exit_code = main(["schema-dump", "ostim-sa"])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_SUCCESS
    assert len(output["schemaCommit"]) == 40
    assert output["offsetFields"] == ["x", "y", "z", "r"]


def test_inspect_prints_detection_evidence(fixture_root: Path, capsys) -> None:
    exit_code = main(["inspect", str(fixture_root / "ostim_sa")])
    output = json.loads(capsys.readouterr().out)
    assert exit_code in {EXIT_SUCCESS, EXIT_WARNINGS}
    assert output["selected"]["proposedFormat"] == "ostim-sa"
    assert output["selected"]["evidence"]


def test_dry_run_does_not_create_conversion_output(fixture_root: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "should-not-exist"
    exit_code = main(
        [
            "convert",
            str(fixture_root / "ostim_sa"),
            "--to",
            "ostim-sa",
            "--output",
            str(output),
            "--dry-run",
            "--pack-id",
            "ExamplePack",
        ]
    )
    capsys.readouterr()
    assert exit_code in {EXIT_SUCCESS, EXIT_WARNINGS}
    assert not output.exists()
