from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from animation_converter.models import ConversionMode, SourceFormat
from animation_converter.packaging import (
    PackageMode,
    PackagingError,
    ValidatedOutput,
    deterministic_zip,
    package_validated_output,
    validate_install_layout,
)
from animation_converter.service import ConversionRequest, ConverterService


def test_deterministic_zip_has_stable_bytes_and_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "Data" / "z").mkdir(parents=True)
    (source / "Data" / "z" / "second.txt").write_text("second", encoding="utf-8")
    (source / "Data" / "a.txt").write_text("first", encoding="utf-8")
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"
    first_names = deterministic_zip(source, first)
    second_names = deterministic_zip(source, second)
    assert first_names == ("Data/a.txt", "Data/z/second.txt")
    assert first_names == second_names
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with ZipFile(first) as archive:
        assert [info.date_time for info in archive.infolist()] == [(1980, 1, 1, 0, 0, 0)] * 2


def test_packaging_requires_validated_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(PackagingError, match="requires"):
        package_validated_output(
            ValidatedOutput(source, False, "ostim-sa", False),
            tmp_path / "output.zip",
            PackageMode.ZIP,
        )


def test_directory_finalization_preserves_data_layout(tmp_path: Path, fixture_root: Path) -> None:
    source = fixture_root / "ostim_sa"
    output = tmp_path / "output"
    result = package_validated_output(
        ValidatedOutput(source, True, "ostim-sa", False),
        output,
        PackageMode.DIRECTORY,
    )
    assert result.path == output
    assert validate_install_layout(output, "ostim-sa")
    assert "Data/SKSE/Plugins/OStim/scenes/ExamplePack/ExampleIdle.json" in result.entry_names


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("new", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "file.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        package_validated_output(ValidatedOutput(source, True, "ostim-sa", False), output, PackageMode.DIRECTORY)
    assert (output / "file.txt").read_text(encoding="utf-8") == "old"


def test_service_zip_and_manifest_are_reproducible(fixture_root: Path, tmp_path: Path) -> None:
    service = ConverterService()
    outputs = [tmp_path / "one.zip", tmp_path / "two.zip"]
    for output in outputs:
        result = service.convert(
            ConversionRequest(
                fixture_root / "ostim_sa",
                SourceFormat.OSTIM_SA,
                output,
                SourceFormat.OSTIM_SA,
                ConversionMode.NORMAL,
                PackageMode.ZIP,
                "none",
                "ExamplePack",
                "Example Pack",
            )
        )
        assert result.output_written
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with ZipFile(outputs[0]) as first, ZipFile(outputs[1]) as second:
        assert first.namelist() == sorted(first.namelist(), key=str.casefold)
        assert first.read("conversion-manifest.json") == second.read("conversion-manifest.json")
