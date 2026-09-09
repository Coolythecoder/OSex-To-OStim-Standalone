"""Evidence-based source-format detection."""

from __future__ import annotations

from pathlib import Path

from .adapters.base import DetectionResult
from .models import SourceFormat
from .registry import AdapterRegistry


class DetectionError(RuntimeError):
    pass


def detect_format(path_or_tree: Path, registry: AdapterRegistry) -> tuple[DetectionResult, list[DetectionResult]]:
    results = [adapter.detect(path_or_tree) for adapter in registry.all()]
    candidates = [result for result in results if result.confidence > 0]
    if not candidates:
        inspected = sorted({path for result in results for path in result.files_inspected})
        return (
            DetectionResult(
                proposed_format=SourceFormat.AUTO,
                confidence=0.0,
                evidence=["No recognized scene metadata signature was found."],
                files_inspected=inspected,
            ),
            results,
        )

    candidates.sort(key=lambda item: (-item.confidence, item.proposed_format.value))
    selected = candidates[0]
    conflicts = [
        f"{candidate.proposed_format.value} scored {candidate.confidence:.2f}: " + "; ".join(candidate.evidence[:2])
        for candidate in candidates[1:]
        if candidate.confidence >= max(0.5, selected.confidence - 0.15)
    ]
    selected = DetectionResult(
        proposed_format=selected.proposed_format,
        confidence=selected.confidence,
        evidence=list(selected.evidence),
        conflicting_evidence=conflicts,
        files_inspected=list(selected.files_inspected),
    )
    return selected, results


def require_detected_format(
    path_or_tree: Path, registry: AdapterRegistry, minimum_confidence: float = 0.5
) -> DetectionResult:
    selected, _ = detect_format(path_or_tree, registry)
    if selected.proposed_format == SourceFormat.AUTO or selected.confidence < minimum_confidence:
        raise DetectionError(
            "Could not identify a supported source format with enough confidence. "
            "Use --from only after confirming the source metadata layout."
        )
    return selected
