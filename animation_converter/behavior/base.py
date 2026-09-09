"""Behavior-writer contracts and explicit external-process support."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..diagnostics import DiagnosticCollection
from ..models import ConversionIR


@dataclass
class BehaviorWriteResult:
    writer: str
    written_files: list[Path] = field(default_factory=list)
    registration_complete: bool = False
    external_step_required: bool = True
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)


class BehaviorWriter(Protocol):
    name: str

    def write(self, ir: ConversionIR, destination: Path, pack_id: str) -> BehaviorWriteResult: ...


@dataclass(frozen=True)
class ExternalGeneratorResult:
    executable: str
    arguments: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


def run_external_generator(
    executable: Path,
    arguments: Sequence[str],
    *,
    working_directory: Path | None = None,
    timeout: float | None = None,
) -> ExternalGeneratorResult:
    """Run only after an explicit user action; shell execution is never used."""
    if not executable.is_file():
        raise FileNotFoundError(f"Behavior generator executable not found: {executable}")
    command = [str(executable), *map(str, arguments)]
    completed = subprocess.run(
        command,
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )
    return ExternalGeneratorResult(
        str(executable), tuple(map(str, arguments)), completed.returncode, completed.stdout, completed.stderr
    )
