"""Adapter contracts and shared contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..diagnostics import DiagnosticCollection
from ..models import ConversionIR, ConversionMode, SourceFormat


@dataclass
class DetectionResult:
    proposed_format: SourceFormat
    confidence: float
    evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    files_inspected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "proposedFormat": self.proposed_format.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "conflictingEvidence": self.conflicting_evidence,
            "filesInspected": self.files_inspected,
        }


@dataclass(frozen=True)
class CapabilityDescriptor:
    format: SourceFormat
    import_supported: bool
    export_supported: bool
    round_trip_supported: bool
    supported_actor_counts: str
    navigation_support: str
    furniture_support: str
    annotation_support: str
    expected_losses: tuple[str, ...] = ()
    required_behavior_generator: str | None = None
    experimental_export: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format.value,
            "import": self.import_supported,
            "export": self.export_supported,
            "roundTrip": self.round_trip_supported,
            "supportedActorCounts": self.supported_actor_counts,
            "navigation": self.navigation_support,
            "furniture": self.furniture_support,
            "annotations": self.annotation_support,
            "expectedLosses": list(self.expected_losses),
            "requiredBehaviorGenerator": self.required_behavior_generator,
            "experimentalExport": self.experimental_export,
        }


@dataclass
class ParseContext:
    mode: ConversionMode = ConversionMode.NORMAL
    pack_id: str | None = None
    display_name: str | None = None
    copy_assets: bool = True
    source_root: Path | None = None
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)


@dataclass
class EmitContext:
    mode: ConversionMode = ConversionMode.NORMAL
    pack_id: str | None = None
    display_name: str | None = None
    copy_assets: bool = True
    overwrite: bool = False
    preserve_unknown_fields: bool = True
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)


@dataclass
class EmitResult:
    destination: Path
    written_files: list[Path] = field(default_factory=list)
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)
    scene_id_map: dict[str, str] = field(default_factory=dict)
    event_map: dict[str, str] = field(default_factory=dict)
    install_ready: bool = False
    output_pack_id: str | None = None


class Adapter(Protocol):
    format: SourceFormat

    def detect(self, path_or_tree: Path) -> DetectionResult: ...

    def parse(self, source: Path, context: ParseContext) -> ConversionIR: ...

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection: ...

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult: ...

    def capabilities(self) -> CapabilityDescriptor: ...
