from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

import animation_converter.archive as archive_module
from animation_converter.archive import (
    ArchiveError,
    ArchiveLimits,
    ArchiveSecurityError,
    safe_extract_7z,
    safe_extract_zip,
    validate_member_name,
)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "folder/../../escape.txt",
        "/absolute.txt",
        r"C:\drive.txt",
        r"\\server\share\file.txt",
        r"\\?\C:\device.txt",
        r"folder\..\escape.txt",
        "folder/file.txt:stream",
    ],
)
def test_rejects_unsafe_member_names(name: str) -> None:
    with pytest.raises(ArchiveSecurityError):
        validate_member_name(name)


def test_zip_traversal_is_rejected_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "bad.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    destination = tmp_path / "staging"
    with pytest.raises(ArchiveSecurityError):
        safe_extract_zip(source, destination)
    assert not (tmp_path / "outside.txt").exists()


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(source, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ArchiveSecurityError, match="symlink"):
        safe_extract_zip(source, tmp_path / "out")


def test_archive_file_count_and_size_limits(tmp_path: Path) -> None:
    source = tmp_path / "many.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("one", b"1")
        archive.writestr("two", b"2")
    with pytest.raises(ArchiveSecurityError, match="files"):
        safe_extract_zip(source, tmp_path / "count", ArchiveLimits(max_files=1))
    with pytest.raises(ArchiveSecurityError, match="expands"):
        safe_extract_zip(source, tmp_path / "size", ArchiveLimits(max_uncompressed_bytes=1))


def test_extreme_compression_ratio_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "ratio.zip"
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("large.txt", b"0" * 100_000)
    with pytest.raises(ArchiveSecurityError, match="compression ratio"):
        safe_extract_zip(source, tmp_path / "ratio", ArchiveLimits(max_compression_ratio=2))


def test_duplicate_casefolded_output_paths_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("Folder/File.txt", "one")
        archive.writestr("folder/file.TXT", "two")
    with pytest.raises(ArchiveSecurityError, match="duplicate"):
        safe_extract_zip(source, tmp_path / "out")


def test_directory_entry_after_file_is_safe(tmp_path: Path) -> None:
    source = tmp_path / "late-directory.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("folder/file.txt", "content")
        archive.writestr("folder/", b"")
    destination = tmp_path / "out"
    safe_extract_zip(source, destination)
    assert (destination / "folder" / "file.txt").read_text(encoding="utf-8") == "content"


def test_absence_of_7z_is_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(archive_module, "find_7z", lambda explicit=None: None)
    with pytest.raises(ArchiveError, match="7-Zip is required"):
        safe_extract_7z(tmp_path / "source.7z", tmp_path / "out")


def test_failed_external_listing_reports_return_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "7z.exe"
    executable.write_bytes(b"placeholder")
    monkeypatch.setattr(archive_module, "find_7z", lambda explicit=None: executable)
    monkeypatch.setattr(
        archive_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, stdout="", stderr="bad archive"),
    )
    with pytest.raises(ArchiveError, match="exit 2"):
        safe_extract_7z(tmp_path / "source.7z", tmp_path / "out")


def test_solid_7z_uses_block_ratio_when_member_packed_sizes_are_blank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "7z.exe"
    executable.write_bytes(b"placeholder")
    listing = """----------
Path = first.hkx
Size = 100
Packed Size = 100
Folder = -
Block = 0

Path = second.hkx
Size = 100
Packed Size =
Folder = -
Block = 0
"""

    def fake_run(command, **kwargs):
        if "l" in command:
            return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(archive_module, "find_7z", lambda explicit=None: executable)
    monkeypatch.setattr(archive_module.subprocess, "run", fake_run)
    assert safe_extract_7z(
        tmp_path / "source.7z",
        tmp_path / "out",
        limits=ArchiveLimits(max_compression_ratio=3),
    ) == []


def test_solid_7z_rejects_an_over_limit_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "7z.exe"
    executable.write_bytes(b"placeholder")
    listing = """----------
Path = first.bin
Size = 10000
Packed Size = 10
Folder = -
Block = 4

Path = second.bin
Size = 10000
Packed Size =
Folder = -
Block = 4
"""
    monkeypatch.setattr(archive_module, "find_7z", lambda explicit=None: executable)
    monkeypatch.setattr(
        archive_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=listing, stderr=""),
    )
    with pytest.raises(ArchiveSecurityError, match="solid block 4"):
        safe_extract_7z(
            tmp_path / "source.7z",
            tmp_path / "out",
            limits=ArchiveLimits(max_compression_ratio=100),
        )
