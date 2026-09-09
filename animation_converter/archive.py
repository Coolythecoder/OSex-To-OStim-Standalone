"""Bounded, traversal-safe archive staging."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


class ArchiveError(RuntimeError):
    pass


class ArchiveSecurityError(ArchiveError):
    pass


@dataclass(frozen=True)
class ArchiveLimits:
    max_files: int = 20_000
    max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024
    max_compression_ratio: float = 1_000.0


DEFAULT_LIMITS = ArchiveLimits()
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar"}


def normalized_member_name(name: str) -> str:
    return name.replace("\\", "/")


def unsafe_member_reason(name: str) -> str | None:
    normalized = normalized_member_name(name)
    if not normalized or not normalized.strip():
        return "empty member name"
    if "\x00" in normalized:
        return "NUL characters are forbidden"
    if normalized.startswith(("/", "//", "\\", "\\\\", "//?/", "//./")):
        return "absolute, UNC, or device paths are forbidden"
    if re.match(r"^[A-Za-z]:", normalized):
        return "Windows drive paths are forbidden"
    parts = PurePosixPath(normalized).parts
    if any(part in {"..", "."} for part in parts):
        return "relative traversal components are forbidden"
    if any(":" in part for part in parts):
        return "colon-qualified or alternate-stream paths are forbidden"
    return None


def validate_member_name(name: str) -> PurePosixPath:
    reason = unsafe_member_reason(name)
    if reason:
        display = name.replace("\x00", "\\0")
        raise ArchiveSecurityError(f"Unsafe archive member {display!r}: {reason}")
    return PurePosixPath(normalized_member_name(name))


def _target_for_member(destination: Path, member: PurePosixPath) -> Path:
    target = destination.joinpath(*member.parts)
    root = destination.resolve()
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ArchiveSecurityError(f"Archive member escapes staging directory: {member}") from exc
    return target


def _is_zip_symlink(external_attr: int) -> bool:
    return stat.S_IFMT((external_attr >> 16) & 0xFFFF) == stat.S_IFLNK


def safe_extract_zip(archive: Path, destination: Path, limits: ArchiveLimits = DEFAULT_LIMITS) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    seen: set[str] = set()
    try:
        with ZipFile(archive, "r") as source:
            entries = source.infolist()
            file_entries = [entry for entry in entries if not entry.is_dir()]
            if len(file_entries) > limits.max_files:
                raise ArchiveSecurityError(f"Archive contains {len(file_entries)} files; limit is {limits.max_files}.")
            total_size = sum(entry.file_size for entry in file_entries)
            if total_size > limits.max_uncompressed_bytes:
                raise ArchiveSecurityError(
                    f"Archive expands to {total_size} bytes; limit is {limits.max_uncompressed_bytes}."
                )

            for entry in entries:
                member = validate_member_name(entry.filename)
                collision_key = member.as_posix().casefold().rstrip("/")
                if collision_key in seen:
                    raise ArchiveSecurityError(f"Archive contains a duplicate output path: {member}")
                seen.add(collision_key)
                if _is_zip_symlink(entry.external_attr):
                    raise ArchiveSecurityError(f"Archive contains a symlink: {member}")
                if entry.file_size:
                    ratio = entry.file_size / max(1, entry.compress_size)
                    if ratio > limits.max_compression_ratio:
                        raise ArchiveSecurityError(
                            f"Archive member {member} has compression ratio {ratio:.1f}; "
                            f"limit is {limits.max_compression_ratio:.1f}."
                        )

                target = _target_for_member(destination, member)
                if entry.is_dir():
                    if target.exists():
                        if target.is_dir():
                            continue
                        raise ArchiveSecurityError(f"Directory entry collides with a file: {member}")
                    target.mkdir(parents=True, exist_ok=False)
                    continue
                if target.exists():
                    raise ArchiveSecurityError(f"Extraction would overwrite an existing path: {member}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry, "r") as input_stream, target.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                extracted.append(target)
    except BadZipFile as exc:
        raise ArchiveError(f"Invalid ZIP archive: {archive.name}") from exc
    validate_extracted_tree(destination)
    return extracted


def find_7z(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    env_path = os.environ.get("AAC_7ZIP")
    if env_path:
        candidates.append(Path(env_path))
    for name in ("7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.extend(
        [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip" / "7z.exe",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "7-Zip" / "7z.exe",
        ]
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _parse_7z_listing(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_entries = False
    for line in text.splitlines():
        if line.startswith("----------"):
            in_entries = True
            current = {}
            continue
        if not in_entries:
            continue
        if not line.strip():
            if current.get("Path"):
                records.append(current)
            current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if current.get("Path"):
        records.append(current)
    return records


def _safe_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _validate_7z_compression_ratios(records: list[dict[str, str]], limits: ArchiveLimits) -> None:
    """Validate ratios without mistaking solid-block listings for one-byte entries.

    In ``7z l -slt`` output, only one member of a solid block normally carries
    the block's packed size. The remaining members have a blank ``Packed Size``
    field, so their ratios can only be checked as part of the complete block.
    """
    solid_blocks: dict[str, list[dict[str, str]]] = {}
    independent: list[dict[str, str]] = []
    for record in records:
        block = str(record.get("Block") or "").strip()
        if block:
            solid_blocks.setdefault(block, []).append(record)
        else:
            independent.append(record)

    for block, members in solid_blocks.items():
        size = sum(_safe_int(member.get("Size")) for member in members)
        packed = sum(_safe_int(member.get("Packed Size")) for member in members)
        if size and packed and size / packed > limits.max_compression_ratio:
            raise ArchiveSecurityError(
                f"Archive solid block {block} has compression ratio {size / packed:.1f}; "
                f"limit is {limits.max_compression_ratio:.1f}."
            )

    for record in independent:
        size = _safe_int(record.get("Size"))
        packed_value = str(record.get("Packed Size") or "").strip()
        if not size or not packed_value:
            continue
        packed = _safe_int(packed_value)
        if packed <= 0 or size / packed > limits.max_compression_ratio:
            member = normalized_member_name(record.get("Path", ""))
            ratio = size / max(1, packed)
            raise ArchiveSecurityError(
                f"Archive member {member} has compression ratio {ratio:.1f}; "
                f"limit is {limits.max_compression_ratio:.1f}."
            )


def safe_extract_7z(
    archive: Path,
    destination: Path,
    *,
    executable: Path | None = None,
    limits: ArchiveLimits = DEFAULT_LIMITS,
) -> list[Path]:
    seven_zip = find_7z(executable)
    if not seven_zip:
        raise ArchiveError(
            "7-Zip is required for .7z and .rar inputs. Install it, add 7z to PATH, "
            "or set AAC_7ZIP to the executable path."
        )
    listing = subprocess.run(
        [str(seven_zip), "l", "-slt", str(archive)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if listing.returncode != 0:
        raise ArchiveError(f"7-Zip could not list {archive.name} (exit {listing.returncode}): {listing.stderr.strip()}")
    records = _parse_7z_listing(listing.stdout)
    files = [record for record in records if record.get("Folder") != "+"]
    if len(files) > limits.max_files:
        raise ArchiveSecurityError(f"Archive file count exceeds {limits.max_files}.")
    total = sum(_safe_int(record.get("Size")) for record in files)
    if total > limits.max_uncompressed_bytes:
        raise ArchiveSecurityError(f"Archive uncompressed size exceeds {limits.max_uncompressed_bytes} bytes.")
    _validate_7z_compression_ratios(files, limits)
    seen: set[str] = set()
    for record in records:
        member = validate_member_name(record.get("Path", ""))
        key = member.as_posix().casefold().rstrip("/")
        if key in seen:
            raise ArchiveSecurityError(f"Archive contains a duplicate output path: {member}")
        seen.add(key)
        if record.get("Symbolic Link") or record.get("Hard Link"):
            raise ArchiveSecurityError(f"Archive contains a link entry: {member}")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ArchiveSecurityError("7-Zip extraction destination must be empty.")
    result = subprocess.run(
        [str(seven_zip), "x", str(archive), f"-o{destination}", "-y", "-aos"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise ArchiveError(
            f"7-Zip extraction failed (exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    validate_extracted_tree(destination)
    return sorted(path for path in destination.rglob("*") if path.is_file())


def validate_extracted_tree(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        try:
            path.resolve(strict=False).relative_to(resolved_root)
        except ValueError as exc:
            raise ArchiveSecurityError(f"Extracted path escapes staging directory: {path}") from exc
        if path.is_symlink():
            raise ArchiveSecurityError(f"Extracted tree contains a symlink: {path}")
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attributes & reparse_flag:
            raise ArchiveSecurityError(f"Extracted tree contains a reparse point: {path}")


@contextmanager
def staged_source(
    source: Path,
    *,
    limits: ArchiveLimits = DEFAULT_LIMITS,
    seven_zip: Path | None = None,
) -> Iterator[Path]:
    source = source.expanduser().resolve()
    if source.is_dir():
        yield source
        return
    if not source.is_file():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.suffix.lower() not in ARCHIVE_SUFFIXES:
        yield source
        return

    temp_root = Path(tempfile.mkdtemp(prefix="animation-converter-"))
    try:
        if source.suffix.lower() == ".zip":
            safe_extract_zip(source, temp_root, limits)
        else:
            safe_extract_7z(source, temp_root, executable=seven_zip, limits=limits)
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
