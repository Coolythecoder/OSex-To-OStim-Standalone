"""Validated directory finalization and deterministic ZIP packaging."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class PackageMode(str, Enum):
    NONE = "none"
    DIRECTORY = "directory"
    ZIP = "zip"


class PackagingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidatedOutput:
    staging_root: Path
    validated: bool
    target_format: str
    install_ready: bool


@dataclass(frozen=True)
class PackageResult:
    path: Path
    mode: PackageMode
    entry_names: tuple[str, ...]


def _safe_staging_files(root: Path) -> list[Path]:
    resolved = root.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PackagingError(f"Output staging tree contains a symlink: {path}")
        try:
            path.resolve(strict=False).relative_to(resolved)
        except ValueError as exc:
            raise PackagingError(f"Output path escapes staging root: {path}") from exc
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: PurePosixPath(*item.relative_to(root).parts).as_posix().casefold())


def validate_install_layout(root: Path, target_format: str) -> bool:
    if target_format == "ostim-sa":
        scene_root = root / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes"
        return scene_root.is_dir() and any(scene_root.rglob("*.json"))
    if target_format == "osa-osex":
        mesh_root = root / "Data" / "meshes" / "0SA" / "mod"
        return mesh_root.is_dir() and any(mesh_root.rglob("*.xml"))
    return False


def deterministic_zip(source: Path, destination: Path) -> tuple[str, ...]:
    files = _safe_staging_files(source)
    names: list[str] = []
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = PurePosixPath(*path.relative_to(source).parts).as_posix()
            if name.startswith("/") or ".." in PurePosixPath(name).parts:
                raise PackagingError(f"Unsafe ZIP entry name: {name}")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.flag_bits |= 0x800
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
            names.append(name)
    return tuple(names)


def _replace_atomically(prepared: Path, destination: Path, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(prepared, destination)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    os.replace(destination, backup)
    try:
        os.replace(prepared, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink(missing_ok=True)


def package_validated_output(
    output: ValidatedOutput,
    destination: Path,
    mode: PackageMode,
    *,
    overwrite: bool = False,
) -> PackageResult:
    if not output.validated:
        raise PackagingError("Packaging requires a conversion output that has completed validation.")
    _safe_staging_files(output.staging_root)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode in {PackageMode.NONE, PackageMode.DIRECTORY}:
        prepared = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        shutil.copytree(output.staging_root, prepared)
        try:
            _replace_atomically(prepared, destination, overwrite)
        finally:
            if prepared.exists():
                shutil.rmtree(prepared, ignore_errors=True)
        names = tuple(
            PurePosixPath(*path.relative_to(destination).parts).as_posix() for path in _safe_staging_files(destination)
        )
        return PackageResult(destination, mode, names)
    if mode != PackageMode.ZIP:
        raise PackagingError(f"Unsupported package mode: {mode}")
    if destination.suffix.casefold() != ".zip":
        destination = destination.with_suffix(".zip")
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".tmp.zip", dir=destination.parent)
    os.close(fd)
    prepared = Path(temp_name)
    try:
        names = deterministic_zip(output.staging_root, prepared)
        _replace_atomically(prepared, destination, overwrite)
    finally:
        prepared.unlink(missing_ok=True)
    return PackageResult(destination, mode, names)
