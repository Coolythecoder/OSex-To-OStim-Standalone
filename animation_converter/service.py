"""Shared conversion application service for CLI and GUI."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .adapters.base import DetectionResult, EmitContext, EmitResult, ParseContext
from .archive import staged_source
from .assets import (
    assign_ostim_target_paths,
    copy_assets_verified,
    copy_source_documents,
    discover_animation_assets,
    source_file_hashes,
)
from .behavior import behavior_writer
from .behavior.base import BehaviorWriteResult
from .detection import DetectionError, detect_format, require_detected_format
from .diagnostics import DiagnosticCollection
from .graph import remap_scene_references
from .models import (
    ActorSlot,
    AnimationEvent,
    ConversionIR,
    ConversionLoss,
    ConversionMode,
    ConversionQuality,
    FurnitureRequirement,
    LossKind,
    PackMetadata,
    Provenance,
    SceneGraph,
    SceneNode,
    SourceDescriptor,
    SourceFormat,
    SpeedVariant,
    semantic_digest,
    stable_internal_id,
)
from .packaging import (
    PackageMode,
    PackageResult,
    ValidatedOutput,
    package_validated_output,
    validate_install_layout,
)
from .registry import AdapterRegistry, default_registry
from .reporting import build_manifest, build_report, report_to_text, write_reports
from .validation import InstallReadiness, calculate_install_readiness, mode_allows_output

EXIT_SUCCESS = 0
EXIT_WARNINGS = 1
EXIT_VALIDATION_FAILURE = 2
EXIT_UNSUPPORTED = 3
EXIT_EXTRACTION_FAILURE = 4
EXIT_INTERNAL_ERROR = 5


class UnsupportedConversionError(RuntimeError):
    pass


class ConversionCancelled(RuntimeError):
    pass


def _source_display_name(path: Path) -> str | None:
    name = path.stem if path.suffix else path.name
    return name.strip() or None


@dataclass
class ConversionRequest:
    input_path: Path
    target_format: SourceFormat
    output_path: Path | None = None
    source_format: SourceFormat = SourceFormat.AUTO
    mode: ConversionMode = ConversionMode.NORMAL
    package_mode: PackageMode = PackageMode.DIRECTORY
    behavior: str = "none"
    pack_id: str | None = None
    display_name: str | None = None
    copy_assets: bool = True
    dry_run: bool = False
    report_path: Path | None = None
    roundtrip_sidecar: bool = True
    overwrite: bool = False
    deterministic: bool = True
    seven_zip: Path | None = None
    cancel_event: Any | None = None


@dataclass
class InspectionResult:
    selected: DetectionResult
    candidates: list[DetectionResult]
    ir: ConversionIR | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "selected": self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        if self.ir:
            data["counts"] = {
                "scenes": len(self.ir.graph.nodes),
                "actors": sum(len(scene.actors) for scene in self.ir.graph.nodes),
                "transitions": sum(scene.is_transition for scene in self.ir.graph.nodes),
                "animationEvents": len(self.ir.events),
                "animationAssets": len(self.ir.assets),
            }
            data["diagnostics"] = self.ir.diagnostics.to_list()
        return data


@dataclass
class ConversionResult:
    request: ConversionRequest
    ir: ConversionIR
    diagnostics: DiagnosticCollection
    report: dict[str, Any]
    readiness: InstallReadiness
    emit_result: EmitResult
    package_result: PackageResult | None = None
    output_written: bool = False
    unsupported: bool = False

    @property
    def exit_code(self) -> int:
        if self.unsupported:
            return EXIT_UNSUPPORTED
        if not self.output_written and not self.request.dry_run:
            return EXIT_VALIDATION_FAILURE
        if self.diagnostics.has_errors:
            return EXIT_VALIDATION_FAILURE
        if self.diagnostics.has_warnings:
            return EXIT_WARNINGS
        return EXIT_SUCCESS


@dataclass
class RoundTripResult:
    first: ConversionResult
    second: ConversionResult | None
    source_digest: str
    roundtrip_digest: str | None
    semantically_equal: bool


def _deduplicate_diagnostics(*collections: DiagnosticCollection) -> DiagnosticCollection:
    result = DiagnosticCollection()
    seen: set[tuple[Any, ...]] = set()
    for collection in collections:
        for item in collection:
            key = (
                item.severity,
                item.code,
                item.message,
                item.category,
                item.source_file,
                item.object_id,
            )
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def _empty_emit(destination: Path) -> EmitResult:
    return EmitResult(destination=destination)


def _check_cancelled(request: ConversionRequest) -> None:
    if request.cancel_event is not None and request.cancel_event.is_set():
        raise ConversionCancelled("Conversion cancelled before output finalization.")


def _load_conversion_manifest(root: Path) -> dict[str, Any] | None:
    candidates = [root / "conversion-manifest.json"]
    candidates.extend(
        sorted(
            root.glob("Data/SKSE/Plugins/OStim/converter_metadata/*/conversion-manifest.json"),
            key=lambda item: str(item).casefold(),
        )
        if root.is_dir()
        else []
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _merge_preserved_provenance(target: Provenance, raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    source_path = raw.get("sourcePath")
    if isinstance(source_path, str):
        normalized = PurePosixPath(source_path.replace("\\", "/"))
        if not normalized.is_absolute() and ".." not in normalized.parts:
            target.source_path = normalized.as_posix()
    if isinstance(raw.get("sourceIdentifier"), str):
        target.source_identifier = raw["sourceIdentifier"]
    if isinstance(raw.get("sourceFormat"), str):
        try:
            target.source_format = SourceFormat(raw["sourceFormat"])
        except ValueError:
            pass
    if isinstance(raw.get("location"), str):
        target.location = raw["location"]
    if isinstance(raw.get("extras"), dict):
        target.extras.update(raw["extras"])
    for source_key, target_list in (
        ("unknownElements", target.unknown_elements),
        ("normalizations", target.normalizations),
        ("warnings", target.warnings),
    ):
        values = raw.get(source_key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value not in target_list:
                    target_list.append(value)


def _restore_source_ids_from_manifest(ir: ConversionIR, root: Path, target_format: SourceFormat) -> None:
    manifest = _load_conversion_manifest(root)
    if not manifest or manifest.get("sourceFramework") != target_format.value:
        return
    mappings = manifest.get("sceneIdMappings")
    if not isinstance(mappings, dict):
        return
    reverse: dict[str, str] = {}
    for original, mapping in mappings.items():
        if isinstance(original, str) and isinstance(mapping, dict) and isinstance(mapping.get("targetId"), str):
            reverse[mapping["targetId"]] = original
    if not reverse:
        return
    preserved = manifest.get("preservedSourceProvenance")
    preserved = preserved if isinstance(preserved, dict) else {}
    folded = {key.casefold(): value for key, value in reverse.items()}

    def restore(value: str | None) -> str | None:
        if value is None:
            return None
        return folded.get(value.casefold(), value)

    for scene in ir.graph.nodes:
        current = scene.source_id
        restored = restore(current) or current
        raw_provenance = preserved.get(restored)
        if isinstance(raw_provenance, dict):
            _merge_preserved_provenance(scene.provenance, raw_provenance.get("scene"))
            for key, objects in (
                ("speeds", scene.speeds),
                ("actors", scene.actors),
                ("actions", scene.actions),
                ("navigations", scene.navigations),
            ):
                raw_objects = raw_provenance.get(key)
                if isinstance(raw_objects, list):
                    for object_value, raw_value in zip(objects, raw_objects, strict=False):
                        _merge_preserved_provenance(object_value.provenance, raw_value)
        if current.casefold() in folded:
            scene.provenance.extras["restoredFromManifest"] = True
            scene.provenance.extras["manifestSourceFormat"] = target_format.value
        if restored != current:
            scene.provenance.normalizations.append(
                f"Restored source scene ID {restored!r} from conversion-manifest.json."
            )
            scene.source_id = restored
            scene.target_id = None
        scene.transition_destination = restore(scene.transition_destination)
        scene.transition_origin = restore(scene.transition_origin)
        for edge in scene.navigations:
            edge.destination = restore(edge.destination)
            edge.origin = restore(edge.origin)
        scene.auto_transitions = {key: restore(value) or value for key, value in scene.auto_transitions.items()}
        for actor in scene.actors:
            actor.auto_transitions = {key: restore(value) or value for key, value in actor.auto_transitions.items()}
    for sequence in ir.sequences:
        for entry in sequence.entries:
            entry.scene_id = restore(entry.scene_id) or entry.scene_id
    ir.diagnostics.info(
        "MANIFEST_SOURCE_IDS_RESTORED",
        "Restored original source scene identifiers from conversion-manifest.json.",
        category="unsupported or lost data",
    )


def _salvage_ir(
    root: Path,
    source_format: SourceFormat,
    pack_id: str,
    display_name: str,
    diagnostics: DiagnosticCollection,
) -> ConversionIR:
    assets = discover_animation_assets(root)
    nodes: list[SceneNode] = []
    events: list[AnimationEvent] = []
    used: set[str] = set()
    for asset in assets:
        stem = PurePosixPath(asset.source_path).stem
        base = stem
        suffix = 2
        while base.casefold() in used:
            base = f"{stem}_{suffix}"
            suffix += 1
        used.add(base.casefold())
        event_name = f"Salvage_{base}"
        asset.event_name = event_name
        asset.actor_index = 0
        event = AnimationEvent(
            hashlib.sha256(f"salvage-event\0{asset.id}".encode()).hexdigest()[:24],
            event_name,
            {0: asset.id},
            provenance=Provenance(
                source_path=asset.source_path,
                source_identifier=stem,
                source_format=source_format,
                warnings=["Generated only because explicit salvage mode was selected."],
            ),
        )
        scene_id = f"Salvage_{base}"
        nodes.append(
            SceneNode(
                id=stable_internal_id(source_format, asset.source_path, scene_id),
                source_id=scene_id,
                name=f"SALVAGED ASSET: {stem}",
                modpack=display_name,
                length=1.0,
                speeds=[SpeedVariant(event_name)],
                actors=[ActorSlot(0)],
                furniture=FurnitureRequirement("none"),
                tags=["salvaged", "incomplete"],
                no_random_selection=True,
                salvaged=True,
                provenance=Provenance(
                    source_path=asset.source_path,
                    source_identifier=stem,
                    source_format=source_format,
                    extras={"assetPlaceholder": True},
                    warnings=["This is not complete scene metadata and must not be treated as install ready."],
                ),
            )
        )
        events.append(event)
    diagnostics.warning(
        "SALVAGE_ASSET_PLACEHOLDERS",
        f"Explicit salvage mode created {len(nodes)} visibly incomplete asset placeholder(s).",
        category="unsupported or lost data",
        remediation="Recover real source scene metadata before producing an installable pack.",
    )
    return ConversionIR(
        pack=PackMetadata(pack_id, display_name),
        source=SourceDescriptor(source_format, root.name),
        graph=SceneGraph(nodes),
        events=events,
        assets=assets,
        losses=[
            ConversionLoss(
                LossKind.GENERATED,
                "SALVAGE_PLACEHOLDER",
                "One non-installable placeholder scene was generated for each discovered HKX asset.",
            )
        ],
        diagnostics=diagnostics,
        quality=ConversionQuality.SALVAGE,
    )


def _remap_events(ir: ConversionIR, event_map: dict[str, str]) -> None:
    folded = {key.casefold(): value for key, value in event_map.items()}
    for scene in ir.graph.nodes:
        for speed in scene.speeds:
            speed.animation = folded.get(speed.animation.casefold(), speed.animation)
    for event in ir.events:
        old_name = event.name
        event.name = folded.get(event.name.casefold(), event.name)
        for asset_id in event.actor_assets.values():
            for asset in ir.assets:
                if asset.id == asset_id:
                    asset.event_name = event.name
        if event.name != old_name:
            event.provenance.normalizations.append(f"Mapped event {old_name!r} to {event.name!r}.")


def _write_external_report(
    request: ConversionRequest,
    report: dict[str, Any],
    manifest: Any,
    *,
    default_on_failure: bool = False,
) -> list[Path]:
    if request.report_path is None:
        if not default_on_failure or request.output_path is None:
            return []
        output = request.output_path.expanduser().resolve()
        stem = output.stem if output.suffix else output.name
        report_json = output.parent / f"{stem}-conversion-report.json"
        report_txt = output.parent / f"{stem}-conversion-report.txt"
        manifest_json = output.parent / f"{stem}-conversion-manifest.json"
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_txt.write_text(report_to_text(report), encoding="utf-8")
        manifest_json.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [report_json, report_txt, manifest_json]
    path = request.report_path.expanduser().resolve()
    if path.suffix.casefold() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        text_path = path.with_suffix(".txt")
        manifest_path = path.with_name(f"{path.stem}-manifest.json")
        text_path.write_text(report_to_text(report), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [path, text_path, manifest_path]
    return write_reports(path, report, manifest)


class ConverterService:
    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def inspect(self, input_path: Path, source_format: SourceFormat = SourceFormat.AUTO) -> InspectionResult:
        with staged_source(input_path) as root:
            selected, candidates = detect_format(root, self.registry)
            if source_format != SourceFormat.AUTO:
                selected = next(
                    (candidate for candidate in candidates if candidate.proposed_format == source_format),
                    DetectionResult(source_format, 1.0, ["Format selected explicitly."], [], []),
                )
            if selected.proposed_format == SourceFormat.AUTO:
                return InspectionResult(selected, candidates, None)
            adapter = self.registry.get(selected.proposed_format)
            context = ParseContext(source_root=root, display_name=_source_display_name(input_path))
            ir = adapter.parse(root, context)
            ir.source.file_hashes = source_file_hashes(input_path if input_path.is_file() else root)
            return InspectionResult(selected, candidates, ir)

    def validate(
        self, input_path: Path, source_format: SourceFormat = SourceFormat.AUTO
    ) -> tuple[ConversionIR, DiagnosticCollection]:
        with staged_source(input_path) as root:
            selected = (
                require_detected_format(root, self.registry)
                if source_format == SourceFormat.AUTO
                else DetectionResult(source_format, 1.0, ["Format selected explicitly."], [], [])
            )
            adapter = self.registry.get(selected.proposed_format)
            parse_context = ParseContext(source_root=root, display_name=_source_display_name(input_path))
            ir = adapter.parse(root, parse_context)
            ir.source.file_hashes = source_file_hashes(input_path if input_path.is_file() else root)
            return ir, adapter.validate(ir, parse_context)

    def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format in {
            SourceFormat.AUTO,
            SourceFormat.OSTIM_LEGACY,
            SourceFormat.SLAL,
            SourceFormat.FLOWERGIRLS,
        }:
            raise UnsupportedConversionError(f"Export to {request.target_format.value} is not supported.")
        if request.output_path is None and not request.dry_run:
            raise ValueError("An output path is required unless --dry-run is used.")

        staging = Path(tempfile.mkdtemp(prefix="animation-converter-output-"))
        output_destination = request.output_path or staging / "dry-run-output"
        try:
            _check_cancelled(request)
            with staged_source(request.input_path, seven_zip=request.seven_zip) as root:
                detection_diagnostics = DiagnosticCollection()
                if request.source_format == SourceFormat.AUTO:
                    try:
                        selected = require_detected_format(root, self.registry)
                        source_format = selected.proposed_format
                    except DetectionError:
                        if request.mode != ConversionMode.SALVAGE:
                            raise
                        source_format = SourceFormat.OSA_OSEX
                        selected = DetectionResult(
                            source_format,
                            0.0,
                            ["No scene metadata detected; explicit salvage mode will inspect HKX assets only."],
                            [],
                            [],
                        )
                else:
                    source_format = request.source_format
                    selected = DetectionResult(source_format, 1.0, ["Format selected explicitly."], [], [])

                parse_display_name = request.display_name
                if request.pack_id is None and parse_display_name is None:
                    parse_display_name = _source_display_name(request.input_path)
                parse_context = ParseContext(
                    mode=request.mode,
                    pack_id=request.pack_id,
                    display_name=parse_display_name,
                    copy_assets=request.copy_assets,
                    source_root=root,
                    diagnostics=detection_diagnostics,
                )
                source_adapter = self.registry.get(source_format)
                ir = source_adapter.parse(root, parse_context)
                _check_cancelled(request)
                ir.source.detection_evidence = selected.evidence
                ir.source.file_hashes = source_file_hashes(request.input_path if request.input_path.is_file() else root)

                if not ir.graph.nodes and request.mode == ConversionMode.SALVAGE:
                    ir = _salvage_ir(
                        root,
                        source_format,
                        request.pack_id or ir.pack.identifier or "SalvagedPack",
                        request.display_name or ir.pack.display_name or "Salvaged Pack",
                        detection_diagnostics,
                    )
                    ir.source.file_hashes = source_file_hashes(
                        request.input_path if request.input_path.is_file() else root
                    )

                pre_validation = source_adapter.validate(ir, parse_context)
                _check_cancelled(request)
                target_adapter = self.registry.get(request.target_format)
                if not target_adapter.capabilities().export_supported:
                    raise UnsupportedConversionError(
                        f"The {request.target_format.value} adapter does not support export."
                    )

                # Parsing and graph validation complete before target files are emitted.
                if not mode_allows_output(request.mode, pre_validation, ir):
                    emit_result = _empty_emit(staging)
                    readiness = calculate_install_readiness(
                        ir, pre_validation, behavior_complete=False, layout_valid=False
                    )
                    manifest = build_manifest(
                        ir, request.target_format, emit_result, pre_validation, deterministic=request.deterministic
                    )
                    report = build_report(
                        ir,
                        request.target_format,
                        pre_validation,
                        readiness,
                        manifest,
                        behavior_writer=request.behavior,
                        dry_run=request.dry_run,
                    )
                    _write_external_report(
                        request,
                        report,
                        manifest,
                        default_on_failure=not request.dry_run,
                    )
                    return ConversionResult(
                        request,
                        ir,
                        pre_validation,
                        report,
                        readiness,
                        emit_result,
                        output_written=False,
                    )

                if request.roundtrip_sidecar:
                    _restore_source_ids_from_manifest(ir, root, request.target_format)

                emit_context = EmitContext(
                    mode=request.mode,
                    pack_id=request.pack_id or ir.pack.identifier,
                    display_name=request.display_name or ir.pack.display_name,
                    copy_assets=request.copy_assets,
                    overwrite=request.overwrite,
                    diagnostics=DiagnosticCollection(),
                )
                emit_result = target_adapter.emit(ir, staging, emit_context)
                _check_cancelled(request)
                original_scene_map = dict(emit_result.scene_id_map)
                original_event_map = dict(emit_result.event_map)
                effective_pack_id = emit_result.output_pack_id or request.pack_id or ir.pack.identifier
                if request.target_format == SourceFormat.OSTIM_SA:
                    assign_ostim_target_paths(
                        ir,
                        effective_pack_id,
                        original_event_map,
                    )

                if request.copy_assets:
                    copy_diagnostics = DiagnosticCollection()
                    copy_assets_verified(ir, root, staging, copy_diagnostics)
                    _check_cancelled(request)
                else:
                    copy_diagnostics = DiagnosticCollection()
                    copy_diagnostics.warning(
                        "ASSET_COPY_DISABLED",
                        "Asset copying was disabled; the output cannot be install ready.",
                        category="asset existence",
                    )

                remap_scene_references(ir, original_scene_map)
                _remap_events(ir, original_event_map)
                pack_id = effective_pack_id
                if request.target_format == SourceFormat.OSTIM_SA:
                    behavior_result = behavior_writer(request.behavior).write(ir, staging, pack_id)
                    docs_destination = (
                        staging / "Data" / "SKSE" / "Plugins" / "OStim" / "converter_metadata" / pack_id / "source_docs"
                    )
                else:
                    behavior_result = BehaviorWriteResult(
                        "source-framework",
                        registration_complete=False,
                        external_step_required=True,
                    )
                    behavior_result.diagnostics.warning(
                        "OSA_BEHAVIOR_EXTERNAL_STEP",
                        "OSA/OSex output requires framework-specific animation registration outside this converter.",
                        category="behavior-registration completeness",
                    )
                    docs_destination = staging / "Documentation" / pack_id / "source_docs"
                document_diagnostics = DiagnosticCollection()
                copy_source_documents(root, docs_destination, document_diagnostics)

                post_validation = target_adapter.validate(ir, emit_context)
                diagnostics = _deduplicate_diagnostics(
                    pre_validation,
                    emit_result.diagnostics,
                    copy_diagnostics,
                    behavior_result.diagnostics,
                    document_diagnostics,
                    post_validation,
                )
                layout_valid = validate_install_layout(staging, request.target_format.value)
                if not layout_valid:
                    diagnostics.error(
                        "INSTALL_LAYOUT_INVALID",
                        "Converted files are not under the required target Data directory layout.",
                        category="install directory layout",
                        can_continue=False,
                    )
                readiness = calculate_install_readiness(
                    ir,
                    diagnostics,
                    behavior_complete=behavior_result.registration_complete,
                    layout_valid=layout_valid,
                )
                emit_result.scene_id_map = original_scene_map
                emit_result.event_map = original_event_map
                emit_result.install_ready = readiness.install_ready
                manifest = build_manifest(
                    ir,
                    request.target_format,
                    emit_result,
                    diagnostics,
                    deterministic=request.deterministic,
                )
                report = build_report(
                    ir,
                    request.target_format,
                    diagnostics,
                    readiness,
                    manifest,
                    behavior_writer=behavior_result.writer,
                    dry_run=request.dry_run,
                )
                write_reports(
                    staging,
                    report,
                    manifest,
                    pack_id=pack_id if request.target_format == SourceFormat.OSTIM_SA else None,
                )
                _write_external_report(request, report, manifest)
                _check_cancelled(request)

                if request.dry_run:
                    return ConversionResult(
                        request,
                        ir,
                        diagnostics,
                        report,
                        readiness,
                        emit_result,
                        output_written=False,
                    )
                if not mode_allows_output(request.mode, diagnostics, ir):
                    return ConversionResult(
                        request,
                        ir,
                        diagnostics,
                        report,
                        readiness,
                        emit_result,
                        output_written=False,
                    )
                package_result = package_validated_output(
                    ValidatedOutput(staging, True, request.target_format.value, readiness.install_ready),
                    output_destination,
                    request.package_mode,
                    overwrite=request.overwrite,
                )
                return ConversionResult(
                    request,
                    ir,
                    diagnostics,
                    report,
                    readiness,
                    emit_result,
                    package_result,
                    output_written=True,
                )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def roundtrip(
        self,
        request: ConversionRequest,
        *,
        through: SourceFormat,
    ) -> RoundTripResult:
        inspection = self.inspect(request.input_path, request.source_format)
        if inspection.ir is None:
            raise DetectionError("Round trip requires parseable scene metadata.")
        source_ir = copy.deepcopy(inspection.ir)
        source_digest = semantic_digest(source_ir)
        source_format = inspection.selected.proposed_format
        with tempfile.TemporaryDirectory(prefix="animation-converter-roundtrip-") as temp:
            temp_root = Path(temp)
            through_output = temp_root / "through"
            first_request = copy.copy(request)
            first_request.target_format = through
            first_request.output_path = through_output
            first_request.package_mode = PackageMode.DIRECTORY
            first_request.dry_run = False
            first_request.overwrite = True
            first_request.mode = ConversionMode.BEST_EFFORT
            first = self.convert(first_request)
            if not first.output_written:
                return RoundTripResult(first, None, source_digest, None, False)
            second_request = copy.copy(request)
            second_request.input_path = through_output
            second_request.source_format = through
            second_request.target_format = source_format
            second_request.output_path = request.output_path
            second_request.mode = ConversionMode.BEST_EFFORT
            second = self.convert(second_request)
            if not second.output_written and not second.request.dry_run:
                return RoundTripResult(first, second, source_digest, None, False)
            roundtrip_digest = semantic_digest(second.ir)
            return RoundTripResult(first, second, source_digest, roundtrip_digest, source_digest == roundtrip_digest)
