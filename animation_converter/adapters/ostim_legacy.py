"""Importer for the original converter's non-loader id/pack/poses/clips JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from ..assets import discover_animation_assets
from ..diagnostics import DiagnosticCollection
from ..models import (
    ActorSlot,
    AnimationEvent,
    ConversionIR,
    ConversionLoss,
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
    Transform,
    stable_internal_id,
)
from ..validation import validate_ir
from .base import CapabilityDescriptor, DetectionResult, EmitContext, EmitResult, ParseContext


def _relative(path: Path, root: Path) -> str:
    try:
        return PurePosixPath(*path.relative_to(root).parts).as_posix()
    except ValueError:
        return path.name


def _json_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.casefold() == ".json" else []
    return sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.casefold() == ".json"),
        key=lambda item: str(item).casefold(),
    )


def _is_legacy_shape(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "poses" in value:
        return True
    return "id" in value and "pack" in value and "clips" in value


def _clips(pose: Any) -> list[dict[str, Any]]:
    if not isinstance(pose, dict):
        return []
    value = pose.get("clips", [])
    return [clip for clip in value if isinstance(clip, dict)] if isinstance(value, list) else []


class OStimLegacyAdapter:
    format = SourceFormat.OSTIM_LEGACY

    def detect(self, path_or_tree: Path) -> DetectionResult:
        root = path_or_tree if path_or_tree.is_dir() else path_or_tree.parent
        inspected: list[str] = []
        matches = 0
        for path in _json_files(path_or_tree)[:64]:
            inspected.append(_relative(path, root))
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            matches += int(_is_legacy_shape(data))
        evidence = [f"{matches} file(s) use the obsolete id/pack/poses/clips converter shape."] if matches else []
        return DetectionResult(self.format, 0.99 if matches else 0.0, evidence, [], inspected)

    def parse(self, source: Path, context: ParseContext) -> ConversionIR:
        root = source if source.is_dir() else source.parent
        diagnostics = context.diagnostics
        assets = discover_animation_assets(root)
        assets_by_name: dict[str, list] = {}
        for asset in assets:
            assets_by_name.setdefault(PurePosixPath(asset.source_path).name.casefold(), []).append(asset)
        nodes: list[SceneNode] = []
        events: list[AnimationEvent] = []
        display_name = context.display_name or context.pack_id or root.name
        for path in _json_files(source):
            source_file = _relative(path, root)
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except UnicodeDecodeError as exc:
                diagnostics.error(
                    "LEGACY_JSON_ENCODING",
                    f"Legacy JSON is not UTF-8: {exc}",
                    category="JSON syntax",
                    source_file=source_file,
                )
                continue
            except (OSError, json.JSONDecodeError) as exc:
                diagnostics.error(
                    "LEGACY_JSON_MALFORMED",
                    f"Could not parse legacy JSON: {exc}",
                    category="JSON syntax",
                    source_file=source_file,
                )
                continue
            if not _is_legacy_shape(data):
                continue
            source_id = str(data.get("id") or path.stem)
            display_name = context.display_name or str(data.get("pack") or display_name)
            poses = data.get("poses") if isinstance(data.get("poses"), list) else [data]
            speeds: list[SpeedVariant] = []
            actor_count = 0
            actors_by_index: dict[int, ActorSlot] = {}
            for pose_index, pose in enumerate(poses):
                clips = _clips(pose)
                event_name = f"{source_id}_P{pose_index + 1}"
                event = AnimationEvent(
                    hashlib.sha256(f"legacy-event\0{source_file}\0{pose_index}".encode()).hexdigest()[:24],
                    event_name,
                    provenance=Provenance(
                        source_path=source_file,
                        source_identifier=str(pose_index),
                        source_format=SourceFormat.OSTIM_LEGACY,
                    ),
                )
                for clip_index, clip in enumerate(clips):
                    actor_index = clip.get("actorIndex", clip.get("actor", clip_index))
                    actor_index = actor_index if isinstance(actor_index, int) and actor_index >= 0 else clip_index
                    actor_count = max(actor_count, actor_index + 1)
                    actor = actors_by_index.setdefault(actor_index, ActorSlot(actor_index))
                    file_value = clip.get("file") or clip.get("path") or clip.get("hkx")
                    if isinstance(file_value, str):
                        candidates = assets_by_name.get(
                            PurePosixPath(file_value.replace("\\", "/")).name.casefold(), []
                        )
                        if len(candidates) == 1:
                            asset = candidates[0]
                            asset.event_name = event_name
                            asset.actor_index = actor_index
                            event.actor_assets[actor_index] = asset.id
                    explicit_offset = any(key in clip for key in ("x", "y", "z", "r", "rx", "ry", "rz", "pos", "rot"))
                    if explicit_offset:
                        pos = clip.get("pos") if isinstance(clip.get("pos"), (list, tuple)) else []
                        rot = clip.get("rot") if isinstance(clip.get("rot"), (list, tuple)) else []
                        values = {
                            "x": clip.get("x", pos[0] if len(pos) > 0 else 0.0),
                            "y": clip.get("y", pos[1] if len(pos) > 1 else 0.0),
                            "z": clip.get("z", pos[2] if len(pos) > 2 else 0.0),
                            "r": clip.get("r", clip.get("rz", rot[2] if len(rot) > 2 else 0.0)),
                        }
                        transform = Transform(
                            provenance=Provenance(
                                source_path=source_file,
                                source_identifier=f"{source_id}:clip:{clip_index}",
                                source_format=SourceFormat.OSTIM_LEGACY,
                            )
                        )
                        for key, value in values.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                setattr(transform, key, float(value))
                                transform.provided_fields.add(key)
                            else:
                                transform.provenance.extras["parseFailed"] = True
                        actor.offset = transform
                    elif any(key in clip for key in ("pos_x", "rot_x")):
                        # This was the prototype's default-filled field family. Without
                        # source provenance, all zeros are not evidence of alignment.
                        actor.offset = Transform(
                            provenance=Provenance(
                                source_path=source_file,
                                source_identifier=f"{source_id}:clip:{clip_index}",
                                source_format=SourceFormat.OSTIM_LEGACY,
                                extras={"inferredFromMissingSource": True},
                            )
                        )
                if clips:
                    speeds.append(SpeedVariant(event_name))
                    events.append(event)
            actors = [actors_by_index.get(index, ActorSlot(index)) for index in range(max(1, actor_count))]
            node = SceneNode(
                id=stable_internal_id(SourceFormat.OSTIM_LEGACY, source_file, source_id),
                source_id=source_id,
                name=str(data.get("name") or source_id),
                modpack=display_name,
                length=float(data.get("length", 6.0)) if isinstance(data.get("length", 6.0), (int, float)) else 6.0,
                speeds=speeds,
                actors=actors,
                furniture=FurnitureRequirement("none"),
                provenance=Provenance(
                    source_path=source_file,
                    source_identifier=source_id,
                    source_format=SourceFormat.OSTIM_LEGACY,
                    extras={"legacyRaw": data},
                    warnings=["Legacy converter JSON is not a valid OStim loader scene."],
                ),
            )
            nodes.append(node)

        diagnostics.warning(
            "LEGACY_INTERMEDIATE_IMPORTED",
            "Imported obsolete converter JSON as a lossy intermediate; direct HKX paths were converted into event mappings.",
            category="unsupported or lost data",
        )
        return ConversionIR(
            pack=PackMetadata(context.pack_id or display_name, display_name),
            source=SourceDescriptor(SourceFormat.OSTIM_LEGACY, root.name),
            graph=SceneGraph(nodes),
            events=events,
            assets=assets,
            losses=[
                ConversionLoss(
                    LossKind.UNSUPPORTED,
                    "LEGACY_SCENE_SEMANTICS_MISSING",
                    "The prototype format did not preserve current OStim navigation and action semantics.",
                )
            ],
            diagnostics=diagnostics,
            quality=ConversionQuality.LOSSY,
        )

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection:
        return validate_ir(ir, include_assets=True, enforce_ostim_scene_ids=False)

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult:
        raise NotImplementedError(
            "Legacy malformed JSON export is intentionally unsupported; export current OStim SA instead."
        )

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            format=self.format,
            import_supported=True,
            export_supported=False,
            round_trip_supported=False,
            supported_actor_counts="Recovered from clip actor indices.",
            navigation_support="Not present in the prototype shape.",
            furniture_support="Not present in the prototype shape.",
            annotation_support="Not present in the prototype shape.",
            expected_losses=("Navigation, actions, event registration, and offset provenance may be absent.",),
            required_behavior_generator="pandora or nemesis after OStim export",
        )
