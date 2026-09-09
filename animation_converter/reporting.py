"""Conversion manifests and human/machine-readable reports."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from . import OSTIM_SA_SCHEMA_VERSION, __version__
from .adapters.base import EmitResult
from .diagnostics import DiagnosticCollection
from .models import ConversionIR, ConversionManifest, LossKind, SourceFormat
from .validation import InstallReadiness


def deterministic_timestamp(deterministic: bool = True) -> tuple[str, dict[str, Any]]:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        try:
            value = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
            return value.isoformat().replace("+00:00", "Z"), {
                "timestampSource": "SOURCE_DATE_EPOCH",
                "sourceDateEpoch": int(epoch),
            }
        except (ValueError, OverflowError, OSError):
            pass
    if deterministic:
        return "1980-01-01T00:00:00Z", {
            "timestampSource": "fixed-reproducible-zip-epoch",
            "zipEntryTimestamp": "1980-01-01T00:00:00Z",
        }
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), {"timestampSource": "system-clock"}


def build_manifest(
    ir: ConversionIR,
    target_format: SourceFormat,
    emit_result: EmitResult,
    diagnostics: DiagnosticCollection,
    *,
    deterministic: bool = True,
) -> ConversionManifest:
    timestamp, deterministic_metadata = deterministic_timestamp(deterministic)
    generated = [loss.to_dict() for loss in ir.losses if loss.kind == LossKind.GENERATED]
    discarded = [loss.to_dict() for loss in ir.losses if loss.kind == LossKind.DISCARDED]
    unsupported = [loss.to_dict() for loss in ir.losses if loss.kind == LossKind.UNSUPPORTED]
    inferred = [loss.to_dict() for loss in ir.losses if loss.kind == LossKind.INFERRED]
    scene_by_source = {scene.source_id: scene for scene in ir.graph.nodes}
    scene_map = {
        source_id: {
            "irId": scene_by_source[source_id].id,
            "targetId": target_id,
        }
        for source_id, target_id in sorted(emit_result.scene_id_map.items())
        if source_id in scene_by_source
    }
    asset_map = {
        PurePosixPath(asset.source_path.replace("\\", "/")).as_posix(): asset.target_path or "" for asset in ir.assets
    }
    preserved_provenance = {
        scene.source_id: {
            "scene": scene.provenance.to_dict(),
            "speeds": [speed.provenance.to_dict() for speed in scene.speeds],
            "actors": [actor.provenance.to_dict() for actor in scene.actors],
            "actions": [action.provenance.to_dict() for action in scene.actions],
            "navigations": [navigation.provenance.to_dict() for navigation in scene.navigations],
        }
        for scene in sorted(ir.graph.nodes, key=lambda item: item.source_id.casefold())
    }
    manifest = ConversionManifest(
        converter_version=__version__,
        source_framework=ir.source.format,
        target_framework=target_format,
        target_schema_version=OSTIM_SA_SCHEMA_VERSION if target_format == SourceFormat.OSTIM_SA else "adapter-2.0",
        source_file_hashes=ir.source.file_hashes,
        source_pack_identifier=ir.pack.identifier,
        output_pack_identifier=emit_result.output_pack_id or ir.pack.identifier,
        scene_id_map=scene_map,
        animation_event_map=emit_result.event_map,
        asset_path_map=asset_map,
        preserved_provenance=preserved_provenance,
        generated_values=generated,
        discarded_values=discarded,
        unsupported_values=unsupported,
        inferred_values=inferred,
        warnings=[item.message for item in diagnostics if item.severity.name == "WARNING"],
        conversion_timestamp=timestamp,
        deterministic_build=deterministic,
        deterministic_metadata=deterministic_metadata,
    )
    ir.manifest = manifest
    return manifest


def build_report(
    ir: ConversionIR,
    target_format: SourceFormat,
    diagnostics: DiagnosticCollection,
    readiness: InstallReadiness,
    manifest: ConversionManifest,
    *,
    behavior_writer: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    transition_count = sum(scene.is_transition for scene in ir.graph.nodes)
    return {
        "converterVersion": __version__,
        "sourceFramework": ir.source.format.value,
        "sourceDetectedVersion": ir.source.detected_version,
        "targetFramework": target_format.value,
        "targetSchemaVersion": manifest.target_schema_version,
        "conversionQuality": ir.quality.value,
        "dryRun": dry_run,
        "counts": {
            "scenes": len(ir.graph.nodes),
            "transitions": transition_count,
            "ordinaryScenes": len(ir.graph.nodes) - transition_count,
            "actors": sum(len(scene.actors) for scene in ir.graph.nodes),
            "animationEvents": len(ir.events),
            "animationAssets": len(ir.assets),
            "navigations": sum(len(scene.navigations) for scene in ir.graph.nodes),
            "sequences": len(ir.sequences),
            "losses": len(ir.losses),
        },
        "behaviorWriter": behavior_writer,
        "installReadiness": readiness.to_dict(),
        "diagnosticCounts": diagnostics.counts(),
        "diagnostics": diagnostics.to_list(),
        "losses": [loss.to_dict() for loss in ir.losses],
        "manifest": manifest.to_dict(),
        "limitations": [
            "No ESP/ESL/ESM, quests, dialogue, Papyrus, framework menus, or skeleton retargeting is performed.",
            "HKX files are copied byte-for-byte and are not repaired or converted between Skyrim editions.",
            "Visual in-game alignment cannot be guaranteed.",
        ],
    }


def report_to_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    readiness = report["installReadiness"]
    lines = [
        f"Animation Converter {report['converterVersion']}",
        f"Source: {report['sourceFramework']}",
        f"Target: {report['targetFramework']} ({report['targetSchemaVersion']})",
        f"Quality: {report['conversionQuality']}",
        f"Scenes: {counts['scenes']} ({counts['ordinaryScenes']} ordinary, {counts['transitions']} transitions)",
        f"Animation events/assets: {counts['animationEvents']}/{counts['animationAssets']}",
        f"Install ready: {'yes' if readiness['installReady'] else 'no'}",
    ]
    if readiness["blockers"]:
        lines.append("Install-readiness blockers: " + ", ".join(readiness["blockers"]))
    lines.append("")
    lines.append("Diagnostics:")
    if not report["diagnostics"]:
        lines.append("  none")
    for diagnostic in report["diagnostics"]:
        location = f" [{diagnostic.get('sourceFile')}]" if diagnostic.get("sourceFile") else ""
        lines.append(f"  {diagnostic['level']} {diagnostic['code']}{location}: {diagnostic['message']}")
        if diagnostic.get("suggestedRemediation"):
            lines.append(f"    Suggested: {diagnostic['suggestedRemediation']}")
    lines.append("")
    lines.append("Loss accounting:")
    if not report["losses"]:
        lines.append("  none")
    for loss in report["losses"]:
        lines.append(f"  {loss['kind']} {loss['code']}: {loss['message']}")
    return "\n".join(lines) + "\n"


def write_reports(
    destination: Path,
    report: dict[str, Any],
    manifest: ConversionManifest,
    *,
    pack_id: str | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    report_json = destination / "conversion-report.json"
    report_txt = destination / "conversion-report.txt"
    manifest_json = destination / "conversion-manifest.json"
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_txt.write_text(report_to_text(report), encoding="utf-8", newline="\n")
    manifest_json.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    written = [report_json, report_txt, manifest_json]
    if pack_id:
        metadata = destination / "Data" / "SKSE" / "Plugins" / "OStim" / "converter_metadata" / pack_id
        metadata.mkdir(parents=True, exist_ok=True)
        for source in (report_json, report_txt, manifest_json):
            target = metadata / source.name
            target.write_bytes(source.read_bytes())
            written.append(target)
    return written
