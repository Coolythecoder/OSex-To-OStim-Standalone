"""Nemesis manifest-only writer."""

from __future__ import annotations

from pathlib import Path

from ..diagnostics import DiagnosticCollection
from ..models import ConversionIR
from .base import BehaviorWriteResult
from .pandora import write_registration_manifest


class NemesisBehaviorWriter:
    name = "nemesis"

    def write(self, ir: ConversionIR, destination: Path, pack_id: str) -> BehaviorWriteResult:
        diagnostics = DiagnosticCollection()
        output = write_registration_manifest(ir, destination, pack_id, self.name)
        diagnostics.warning(
            "NEMESIS_EXTERNAL_GENERATION_REQUIRED",
            "A validated event manifest was written, but no invented Nemesis patch fragments were emitted.",
            category="behavior-registration completeness",
            remediation="Use a verified Nemesis patch workflow and run Nemesis explicitly.",
        )
        return BehaviorWriteResult(self.name, [output], False, True, diagnostics)
