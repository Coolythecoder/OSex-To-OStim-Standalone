"""Current OStim Standalone scene and sequence adapter."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .. import OSTIM_SA_SCHEMA_COMMIT, OSTIM_SA_SCHEMA_VERSION
from ..assets import discover_animation_assets, link_events_to_assets
from ..diagnostics import DiagnosticCollection
from ..models import (
    ActorSlot,
    ConversionIR,
    ConversionLoss,
    ConversionQuality,
    FurnitureRequirement,
    LossKind,
    NavigationEdge,
    PackMetadata,
    Provenance,
    SceneAction,
    SceneGraph,
    SceneNode,
    Sequence,
    SequenceEntry,
    SourceDescriptor,
    SourceFormat,
    SpeedVariant,
    Transform,
    TransitionNode,
    stable_internal_id,
    unique_preserving_order,
)
from ..validation import validate_ir
from .base import CapabilityDescriptor, DetectionResult, EmitContext, EmitResult, ParseContext

SCENE_MARKER = ("skse", "plugins", "ostim", "scenes")
SEQUENCE_MARKER = ("skse", "plugins", "ostim", "sequences")
SCENE_KEYS = {
    "name",
    "modpack",
    "length",
    "destination",
    "origin",
    "navigations",
    "speeds",
    "defaultSpeed",
    "noRandomSelection",
    "fadeOnEntry",
    "furniture",
    "offset",
    "scaleOffsetWithFurniture",
    "tags",
    "autoTransitions",
    "actors",
    "actions",
    "priority",
    "description",
    "icon",
    "border",
    "noWarnings",
}
SCENE_ALIASES = {"modPack"}
LEGACY_SHAPE_KEYS = {"id", "pack", "poses"}
NAVIGATION_KEYS = {"destination", "origin", "priority", "description", "icon", "border", "noWarnings"}
SPEED_KEYS = {"animation", "playbackSpeed", "displaySpeed"}
ACTOR_KEYS = {
    "type",
    "intendedSex",
    "sosBend",
    "tngBend",
    "scale",
    "scaleHeight",
    "animationIndex",
    "underlyingExpression",
    "expressionAction",
    "expressionOverride",
    "lookUp",
    "lookDown",
    "lookLeft",
    "lookRight",
    "noStrip",
    "feetOnGround",
    "offset",
    "requirements",
    "tags",
    "autoTransitions",
}
ACTION_KEYS = {"type", "actor", "target", "performer", "muted", "doPeaks", "peaksAnnotated"}
SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.-]+")


def _relative(path: Path, root: Path) -> str:
    try:
        return PurePosixPath(*path.relative_to(root).parts).as_posix()
    except ValueError:
        return path.name


def _marker_index(path: Path, marker: tuple[str, ...]) -> int | None:
    parts = [part.casefold() for part in path.parts]
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return index
    return None


def _read_json(path: Path, diagnostics: DiagnosticCollection, root: Path) -> Any | None:
    source_file = _relative(path, root)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        diagnostics.error(
            "JSON_ENCODING_UNSUPPORTED",
            f"JSON is not valid UTF-8: {exc}",
            category="JSON syntax",
            source_file=source_file,
            remediation="Re-save the source metadata as UTF-8.",
            can_continue=False,
        )
        return None
    except OSError as exc:
        diagnostics.error(
            "JSON_READ_FAILED",
            f"Could not read JSON: {exc}",
            category="JSON syntax",
            source_file=source_file,
            can_continue=False,
        )
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        diagnostics.error(
            "JSON_MALFORMED",
            f"Malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            category="JSON syntax",
            source_file=source_file,
            can_continue=False,
        )
        return None


def _looks_like_scene(data: Any) -> bool:
    return isinstance(data, dict) and bool(set(data) & (SCENE_KEYS | SCENE_ALIASES | LEGACY_SHAPE_KEYS))


def discover_scene_json_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.casefold() == ".json" else []
    all_json = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".json"),
        key=lambda item: str(item).casefold(),
    )
    marked = [path for path in all_json if _marker_index(path, SCENE_MARKER) is not None]
    if marked:
        return marked
    if root.name.casefold() in {"scene", "scenes"}:
        return all_json
    candidates: list[Path] = []
    for path in all_json:
        if path.name.casefold() in {
            "conversion-report.json",
            "conversion-manifest.json",
            "conversion_report.json",
        }:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _looks_like_scene(data) and "animations" not in data:
            candidates.append(path)
    return candidates


def discover_sequence_json_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".json" and _marker_index(path, SEQUENCE_MARKER) is not None
        ),
        key=lambda item: str(item).casefold(),
    )


def _scene_relative_directory(path: Path) -> str:
    marker = _marker_index(path, SCENE_MARKER)
    if marker is None:
        return PurePosixPath(*path.parent.parts[-1:]).as_posix() if path.parent.name else ""
    after_marker = path.parts[marker + len(SCENE_MARKER) : -1]
    return PurePosixPath(*after_marker).as_posix() if after_marker else ""


def _unknown_fields(data: dict[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in known}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str)}


def _parse_transform(
    value: Any,
    diagnostics: DiagnosticCollection,
    *,
    source_file: str,
    object_id: str,
    location: str,
) -> Transform | None:
    provenance = Provenance(
        source_path=source_file,
        source_identifier=object_id,
        source_format=SourceFormat.OSTIM_SA,
        location=location,
    )
    if not isinstance(value, dict):
        provenance.extras["parseFailed"] = True
        diagnostics.error(
            "OFFSET_SHAPE_INVALID",
            f"{location} must be an object with x, y, z, and r numeric fields.",
            category="offsets and alignment",
            source_file=source_file,
            object_id=object_id,
        )
        return Transform(provenance=provenance)
    transform = Transform(provenance=provenance)
    unknown = _unknown_fields(value, {"x", "y", "z", "r"})
    if unknown:
        provenance.extras["jsonUnknown"] = unknown
    for field_name in ("x", "y", "z", "r"):
        if field_name not in value:
            continue
        parsed = _number(value[field_name])
        if parsed is None:
            provenance.extras["parseFailed"] = True
            diagnostics.error(
                "OFFSET_VALUE_INVALID",
                f"{location}.{field_name} must be numeric.",
                category="offsets and alignment",
                source_file=source_file,
                object_id=object_id,
            )
            continue
        setattr(transform, field_name, parsed)
        transform.provided_fields.add(field_name)
    return transform


def _parse_navigation(
    value: Any,
    index: int,
    source_file: str,
    scene_id: str,
    diagnostics: DiagnosticCollection,
) -> NavigationEdge | None:
    if not isinstance(value, dict):
        diagnostics.error(
            "NAVIGATION_SHAPE_INVALID",
            f"Navigation {index} must be an object.",
            category="schema shape",
            source_file=source_file,
            object_id=scene_id,
        )
        return None
    provenance = Provenance(
        source_path=source_file,
        source_identifier=f"{scene_id}:navigation:{index}",
        source_format=SourceFormat.OSTIM_SA,
        location=f"navigations[{index}]",
        extras={"jsonUnknown": _unknown_fields(value, NAVIGATION_KEYS)},
    )
    return NavigationEdge(
        destination=_string(value.get("destination")),
        origin=_string(value.get("origin")),
        priority=_integer(value.get("priority")),
        description=_string(value.get("description")),
        icon=_string(value.get("icon")),
        border=_string(value.get("border")),
        no_warnings=_boolean(value.get("noWarnings")),
        provenance=provenance,
    )


def _parse_speed(
    value: Any,
    index: int,
    source_file: str,
    scene_id: str,
    diagnostics: DiagnosticCollection,
) -> SpeedVariant | None:
    if not isinstance(value, dict):
        diagnostics.error(
            "SPEED_SHAPE_INVALID",
            f"Speed {index} must be an object.",
            category="speed validity",
            source_file=source_file,
            object_id=scene_id,
        )
        return None
    animation = _string(value.get("animation"))
    if not animation:
        diagnostics.error(
            "SPEED_ANIMATION_MISSING",
            f"Speed {index} does not name an animation event.",
            category="speed validity",
            source_file=source_file,
            object_id=scene_id,
        )
        return None
    return SpeedVariant(
        animation=animation,
        playback_speed=_number(value.get("playbackSpeed")),
        display_speed=_number(value.get("displaySpeed")),
        provenance=Provenance(
            source_path=source_file,
            source_identifier=animation,
            source_format=SourceFormat.OSTIM_SA,
            location=f"speeds[{index}]",
            extras={"jsonUnknown": _unknown_fields(value, SPEED_KEYS)},
        ),
    )


def _parse_actor(
    value: Any,
    index: int,
    source_file: str,
    scene_id: str,
    diagnostics: DiagnosticCollection,
) -> ActorSlot:
    if not isinstance(value, dict):
        diagnostics.error(
            "ACTOR_SHAPE_INVALID",
            f"Actor {index} must be an object.",
            category="actor index validity",
            source_file=source_file,
            object_id=scene_id,
        )
        value = {}
    provenance = Provenance(
        source_path=source_file,
        source_identifier=f"{scene_id}:actor:{index}",
        source_format=SourceFormat.OSTIM_SA,
        location=f"actors[{index}]",
        extras={"jsonUnknown": _unknown_fields(value, ACTOR_KEYS)},
    )
    sos_bend = _integer(value.get("sosBend"))
    tng_bend = _integer(value.get("tngBend"))
    if sos_bend is not None and tng_bend is not None and sos_bend != tng_bend:
        diagnostics.warning(
            "ACTOR_BEND_ALIAS_CONFLICT",
            f"Actor {index} defines conflicting sosBend and tngBend values; both are preserved in the IR.",
            category="schema shape",
            source_file=source_file,
            object_id=scene_id,
        )
    return ActorSlot(
        index=index,
        type=_string(value.get("type")) or "npc",
        intended_sex=_string(value.get("intendedSex")),
        sos_bend=sos_bend,
        tng_bend=tng_bend,
        scale=_number(value.get("scale")),
        scale_height=_number(value.get("scaleHeight")),
        animation_index=_integer(value.get("animationIndex")),
        tags=_string_list(value.get("tags")),
        feet_on_ground=_boolean(value.get("feetOnGround")),
        no_strip=_boolean(value.get("noStrip")),
        offset=(
            _parse_transform(
                value.get("offset"),
                diagnostics,
                source_file=source_file,
                object_id=scene_id,
                location=f"actors[{index}].offset",
            )
            if "offset" in value
            else None
        ),
        requirements=_string_list(value.get("requirements")),
        auto_transitions=_string_map(value.get("autoTransitions")),
        underlying_expression=_string(value.get("underlyingExpression")),
        expression_action=_integer(value.get("expressionAction")),
        expression_override=_string(value.get("expressionOverride")),
        look_up=_integer(value.get("lookUp")),
        look_down=_integer(value.get("lookDown")),
        look_left=_integer(value.get("lookLeft")),
        look_right=_integer(value.get("lookRight")),
        provenance=provenance,
    )


def _parse_action(
    value: Any,
    index: int,
    source_file: str,
    scene_id: str,
    diagnostics: DiagnosticCollection,
) -> SceneAction | None:
    if not isinstance(value, dict):
        diagnostics.error(
            "ACTION_SHAPE_INVALID",
            f"Action {index} must be an object.",
            category="action validity",
            source_file=source_file,
            object_id=scene_id,
        )
        return None
    action_type = _string(value.get("type"))
    actor = _integer(value.get("actor"))
    if not action_type or actor is None:
        diagnostics.error(
            "ACTION_REQUIRED_FIELD_MISSING",
            f"Action {index} requires string type and integer actor fields.",
            category="action validity",
            source_file=source_file,
            object_id=scene_id,
        )
        return None
    return SceneAction(
        type=action_type,
        actor=actor,
        target=_integer(value.get("target")),
        performer=_integer(value.get("performer")),
        muted=_boolean(value.get("muted")),
        do_peaks=_boolean(value.get("doPeaks")),
        peaks_annotated=_boolean(value.get("peaksAnnotated")),
        provenance=Provenance(
            source_path=source_file,
            source_identifier=f"{scene_id}:action:{index}",
            source_format=SourceFormat.OSTIM_SA,
            location=f"actions[{index}]",
            extras={"jsonUnknown": _unknown_fields(value, ACTION_KEYS)},
        ),
    )


def _serialize_transform(transform: Transform) -> dict[str, float]:
    data = transform.to_ostim_dict()
    unknown = transform.provenance.extras.get("jsonUnknown")
    if isinstance(unknown, dict):
        for key in sorted(unknown):
            if key not in data:
                data[key] = unknown[key]
    return data


def _put_optional(data: dict[str, Any], key: str, value: Any, *, default: Any = None) -> None:
    if value is not None and value != default:
        data[key] = value


def _safe_identifier(value: str, fallback: str) -> str:
    sanitized = SAFE_IDENTIFIER.sub("_", value.strip()).strip("._-")
    if not sanitized:
        sanitized = fallback
    if not sanitized[0].isalnum():
        sanitized = f"x_{sanitized}"
    return sanitized[:180]


def _unique_identifier(base: str, used: set[str], source_key: str) -> str:
    candidate = base
    if candidate.casefold() not in used:
        used.add(candidate.casefold())
        return candidate
    suffix = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base[:171]}_{suffix}"
    index = 2
    while candidate.casefold() in used:
        candidate = f"{base[:168]}_{suffix}_{index}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _serialize_navigation(edge: NavigationEdge, reference_map: dict[str, str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if edge.destination:
        data["destination"] = reference_map.get(edge.destination, edge.destination)
    if edge.origin:
        data["origin"] = reference_map.get(edge.origin, edge.origin)
    _put_optional(data, "priority", edge.priority)
    _put_optional(data, "description", edge.description)
    _put_optional(data, "icon", edge.icon)
    _put_optional(data, "border", edge.border)
    _put_optional(data, "noWarnings", edge.no_warnings, default=False)
    unknown = edge.provenance.extras.get("jsonUnknown")
    if isinstance(unknown, dict):
        for key in sorted(unknown):
            if key not in data:
                data[key] = unknown[key]
    return data


def _serialize_actor(actor: ActorSlot, event_index: int, reference_map: dict[str, str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    _put_optional(data, "type", actor.type, default="npc")
    _put_optional(data, "intendedSex", actor.intended_sex)
    if actor.sos_bend is not None:
        data["sosBend"] = actor.sos_bend
    elif actor.tng_bend is not None:
        data["tngBend"] = actor.tng_bend
    _put_optional(data, "scale", actor.scale, default=1.0)
    _put_optional(data, "scaleHeight", actor.scale_height, default=120.748)
    _put_optional(data, "animationIndex", actor.animation_index, default=event_index)
    _put_optional(data, "underlyingExpression", actor.underlying_expression)
    _put_optional(data, "expressionAction", actor.expression_action)
    _put_optional(data, "expressionOverride", actor.expression_override)
    _put_optional(data, "lookUp", actor.look_up)
    _put_optional(data, "lookDown", actor.look_down)
    _put_optional(data, "lookLeft", actor.look_left)
    _put_optional(data, "lookRight", actor.look_right)
    _put_optional(data, "noStrip", actor.no_strip, default=False)
    _put_optional(data, "feetOnGround", actor.feet_on_ground)
    if actor.offset is not None and (actor.offset.was_provided or not actor.offset.is_neutral):
        data["offset"] = _serialize_transform(actor.offset)
    if actor.requirements:
        data["requirements"] = unique_preserving_order(actor.requirements)
    if actor.tags:
        data["tags"] = unique_preserving_order(actor.tags)
    if actor.auto_transitions:
        data["autoTransitions"] = {
            key: reference_map.get(value, value) for key, value in sorted(actor.auto_transitions.items())
        }
    unknown = actor.provenance.extras.get("jsonUnknown")
    if isinstance(unknown, dict):
        for key in sorted(unknown):
            if key not in data and key not in {"id", "pack", "poses", "clips"}:
                data[key] = unknown[key]
    return data


def _serialize_action(action: SceneAction) -> dict[str, Any]:
    data: dict[str, Any] = {"type": action.type, "actor": action.actor}
    _put_optional(data, "target", action.target, default=action.actor)
    _put_optional(data, "performer", action.performer, default=action.actor)
    _put_optional(data, "muted", action.muted, default=False)
    _put_optional(data, "doPeaks", action.do_peaks, default=True)
    _put_optional(data, "peaksAnnotated", action.peaks_annotated, default=False)
    unknown = action.provenance.extras.get("jsonUnknown")
    if isinstance(unknown, dict):
        for key in sorted(unknown):
            if key not in data:
                data[key] = unknown[key]
    return data


class OStimSAAdapter:
    format = SourceFormat.OSTIM_SA

    def detect(self, path_or_tree: Path) -> DetectionResult:
        files = discover_scene_json_files(path_or_tree)
        inspected: list[str] = []
        canonical = 0
        legacy = 0
        aliases = 0
        marker_count = 0
        root = path_or_tree if path_or_tree.is_dir() else path_or_tree.parent
        for path in files[:64]:
            inspected.append(_relative(path, root))
            if _marker_index(path, SCENE_MARKER) is not None:
                marker_count += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if LEGACY_SHAPE_KEYS <= set(data) or ("poses" in data and "clips" in json.dumps(data)):
                legacy += 1
            elif "speeds" in data and "actors" in data:
                canonical += 1
                aliases += int("modPack" in data and "modpack" not in data)
        evidence: list[str] = []
        confidence = 0.0
        if canonical:
            confidence = 0.97 if marker_count else 0.82
            evidence.append(f"{canonical} JSON file(s) use loader-shaped speeds and actors fields.")
        if marker_count:
            evidence.append(f"{marker_count} file(s) are under SKSE/Plugins/OStim/scenes.")
        if aliases:
            evidence.append(f"{aliases} file(s) use the accepted historical modPack alias.")
        conflicts = []
        if legacy:
            conflicts.append(f"{legacy} file(s) use the obsolete id/pack/poses/clips shape.")
            if not canonical:
                confidence = 0.1
        return DetectionResult(self.format, confidence, evidence, conflicts, inspected)

    def parse(self, source: Path, context: ParseContext) -> ConversionIR:
        root = source if source.is_dir() else source.parent
        diagnostics = context.diagnostics
        scene_files = discover_scene_json_files(source)
        duplicate_stems: dict[str, Path] = {}
        blocked: set[Path] = set()
        for path in scene_files:
            key = path.stem.casefold()
            if key in duplicate_stems:
                diagnostics.error(
                    "OSTIM_DUPLICATE_FILENAME",
                    f"Scene filenames collide globally: {_relative(duplicate_stems[key], root)} and {_relative(path, root)}.",
                    category="duplicate filenames",
                    source_file=_relative(path, root),
                    object_id=path.stem,
                    remediation="Rename one scene file; subdirectories do not namespace scene IDs.",
                    can_continue=False,
                )
                blocked.update({path, duplicate_stems[key]})
            else:
                duplicate_stems[key] = path

        nodes: list[SceneNode] = []
        modpacks: list[str] = []
        for path in scene_files:
            source_file = _relative(path, root)
            data = _read_json(path, diagnostics, root)
            if data is None:
                continue
            if not isinstance(data, dict):
                diagnostics.error(
                    "OSTIM_SCENE_NOT_OBJECT",
                    "OStim scene JSON must contain an object.",
                    category="schema shape",
                    source_file=source_file,
                    object_id=path.stem,
                )
                continue
            if LEGACY_SHAPE_KEYS <= set(data) or "poses" in data or "clips" in data:
                diagnostics.error(
                    "OSTIM_LEGACY_INTERMEDIATE_SHAPE",
                    "The id/pack/poses/clips object is a legacy converter intermediate, not current OStim scene JSON.",
                    category="schema shape",
                    source_file=source_file,
                    object_id=path.stem,
                    remediation="Import it as ostim-legacy, then export through the current OStim SA adapter.",
                    can_continue=False,
                )
                continue
            scene_id = path.stem
            modpack = _string(data.get("modpack"))
            normalizations: list[str] = []
            if modpack is None and isinstance(data.get("modPack"), str):
                modpack = data["modPack"]
                normalizations.append("Normalized historical modPack key to canonical modpack.")
                diagnostics.warning(
                    "OSTIM_MODPACK_ALIAS",
                    "Normalized historical modPack casing to canonical modpack.",
                    category="schema shape",
                    source_file=source_file,
                    object_id=scene_id,
                )
            modpack = modpack or context.display_name or context.pack_id or root.name
            modpacks.append(modpack)
            length = _number(data.get("length"))
            if length is None:
                length = 0.0
                diagnostics.error(
                    "OSTIM_LENGTH_MISSING",
                    "Scene length is missing or non-numeric.",
                    category="schema shape",
                    source_file=source_file,
                    object_id=scene_id,
                )
            raw_speeds = data.get("speeds")
            if not isinstance(raw_speeds, list):
                diagnostics.error(
                    "OSTIM_SPEEDS_SHAPE",
                    "Scene speeds must be an array.",
                    category="speed validity",
                    source_file=source_file,
                    object_id=scene_id,
                )
                raw_speeds = []
            speeds = [
                speed
                for index, value in enumerate(raw_speeds)
                for speed in [_parse_speed(value, index, source_file, scene_id, diagnostics)]
                if speed is not None
            ]
            raw_actors = data.get("actors")
            if not isinstance(raw_actors, list):
                diagnostics.error(
                    "OSTIM_ACTORS_SHAPE",
                    "Scene actors must be an array.",
                    category="actor index validity",
                    source_file=source_file,
                    object_id=scene_id,
                )
                raw_actors = []
            actors = [
                _parse_actor(value, index, source_file, scene_id, diagnostics) for index, value in enumerate(raw_actors)
            ]
            raw_navigations = data.get("navigations", [])
            if not isinstance(raw_navigations, list):
                diagnostics.error(
                    "OSTIM_NAVIGATIONS_SHAPE",
                    "Scene navigations must be an array.",
                    category="schema shape",
                    source_file=source_file,
                    object_id=scene_id,
                )
                raw_navigations = []
            navigations = [
                navigation
                for index, value in enumerate(raw_navigations)
                for navigation in [_parse_navigation(value, index, source_file, scene_id, diagnostics)]
                if navigation is not None
            ]
            raw_actions = data.get("actions", [])
            if not isinstance(raw_actions, list):
                diagnostics.error(
                    "OSTIM_ACTIONS_SHAPE",
                    "Scene actions must be an array.",
                    category="action validity",
                    source_file=source_file,
                    object_id=scene_id,
                )
                raw_actions = []
            actions = [
                action
                for index, value in enumerate(raw_actions)
                for action in [_parse_action(value, index, source_file, scene_id, diagnostics)]
                if action is not None
            ]
            provenance = Provenance(
                source_path=source_file,
                source_identifier=scene_id,
                source_format=SourceFormat.OSTIM_SA,
                location="$",
                extras={"jsonUnknown": _unknown_fields(data, SCENE_KEYS | SCENE_ALIASES)},
                normalizations=normalizations,
            )
            node_type = TransitionNode if isinstance(data.get("destination"), str) else SceneNode
            node = node_type(
                id=stable_internal_id(SourceFormat.OSTIM_SA, source_file, scene_id),
                source_id=scene_id,
                name=_string(data.get("name")) or scene_id,
                modpack=modpack,
                length=length,
                speeds=speeds,
                actors=actors,
                relative_directory=_scene_relative_directory(path),
                navigations=navigations,
                default_speed=_integer(data.get("defaultSpeed")) or 0,
                no_random_selection=_boolean(data.get("noRandomSelection")),
                fade_on_entry=_boolean(data.get("fadeOnEntry")),
                furniture=FurnitureRequirement(_string(data.get("furniture")) or "none"),
                offset=(
                    _parse_transform(
                        data.get("offset"),
                        diagnostics,
                        source_file=source_file,
                        object_id=scene_id,
                        location="offset",
                    )
                    if "offset" in data
                    else None
                ),
                scale_offset_with_furniture=_boolean(data.get("scaleOffsetWithFurniture")),
                tags=_string_list(data.get("tags")),
                auto_transitions=_string_map(data.get("autoTransitions")),
                actions=actions,
                transition_destination=_string(data.get("destination")),
                transition_origin=_string(data.get("origin")),
                transition_priority=_integer(data.get("priority")),
                transition_description=_string(data.get("description")),
                transition_icon=_string(data.get("icon")),
                transition_border=_string(data.get("border")),
                transition_no_warnings=_boolean(data.get("noWarnings")),
                provenance=provenance,
            )
            if path in blocked:
                node.provenance.warnings.append("Filename collides with another scene and cannot be exported safely.")
            nodes.append(node)

        sequences: list[Sequence] = []
        sequence_ids: set[str] = set()
        for path in discover_sequence_json_files(root):
            source_file = _relative(path, root)
            data = _read_json(path, diagnostics, root)
            if not isinstance(data, dict):
                continue
            sequence_id = path.stem
            if sequence_id.casefold() in sequence_ids:
                diagnostics.error(
                    "OSTIM_DUPLICATE_SEQUENCE_FILENAME",
                    f"Duplicate sequence filename {sequence_id!r}.",
                    category="duplicate filenames",
                    source_file=source_file,
                )
                continue
            sequence_ids.add(sequence_id.casefold())
            raw_entries = data.get("scenes")
            if not isinstance(raw_entries, list):
                diagnostics.error(
                    "OSTIM_SEQUENCE_SCENES_SHAPE",
                    "Sequence scenes must be an array.",
                    category="schema shape",
                    source_file=source_file,
                )
                continue
            entries: list[SequenceEntry] = []
            for index, raw_entry in enumerate(raw_entries):
                if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("id"), str):
                    diagnostics.error(
                        "OSTIM_SEQUENCE_ENTRY_SHAPE",
                        f"Sequence entry {index} requires a string id.",
                        category="schema shape",
                        source_file=source_file,
                    )
                    continue
                entries.append(
                    SequenceEntry(
                        scene_id=raw_entry["id"],
                        duration=_number(raw_entry.get("duration")),
                        provenance=Provenance(
                            source_path=source_file,
                            source_identifier=raw_entry["id"],
                            source_format=SourceFormat.OSTIM_SA,
                            location=f"scenes[{index}]",
                            extras={"jsonUnknown": _unknown_fields(raw_entry, {"id", "duration"})},
                        ),
                    )
                )
            sequences.append(
                Sequence(
                    id=stable_internal_id(SourceFormat.OSTIM_SA, source_file, sequence_id),
                    source_id=sequence_id,
                    entries=entries,
                    tags=_string_list(data.get("tags")),
                    provenance=Provenance(
                        source_path=source_file,
                        source_identifier=sequence_id,
                        source_format=SourceFormat.OSTIM_SA,
                        extras={"jsonUnknown": _unknown_fields(data, {"scenes", "tags"})},
                    ),
                )
            )

        display_name = context.display_name or (modpacks[0] if modpacks else root.name)
        pack_id = context.pack_id or _safe_identifier(display_name, "ConvertedPack")
        ir = ConversionIR(
            pack=PackMetadata(pack_id, display_name),
            source=SourceDescriptor(
                SourceFormat.OSTIM_SA,
                root.name,
                detected_version=OSTIM_SA_SCHEMA_VERSION,
                detection_evidence=["Scene IDs derived from JSON filenames."],
            ),
            graph=SceneGraph(nodes),
            assets=discover_animation_assets(root),
            sequences=sequences,
            diagnostics=diagnostics,
        )
        link_events_to_assets(ir, diagnostics)
        return ir

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection:
        return validate_ir(ir, include_assets=True, enforce_ostim_scene_ids=True)

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult:
        diagnostics = context.diagnostics
        destination.mkdir(parents=True, exist_ok=True)
        pack_id = _safe_identifier(context.pack_id or ir.pack.identifier, "ConvertedPack")
        if pack_id.casefold().startswith("ostim"):
            original = pack_id
            pack_id = f"Pack_{pack_id}"[:180]
            diagnostics.warning(
                "OSTIM_RESERVED_PACK_PREFIX_NORMALIZED",
                f"Pack ID {original!r} used the reserved OStim prefix and was renamed to {pack_id!r}.",
                category="schema shape",
            )
        display_name = context.display_name or ir.pack.display_name
        if ir.source.format != SourceFormat.OSTIM_SA:
            for scene in ir.graph.nodes:
                has_unmapped_xml = bool(scene.provenance.unknown_elements) or bool(
                    scene.provenance.extras.get("xmlUnknownAttributes")
                )
                if not has_unmapped_xml:
                    continue
                diagnostics.warning(
                    "SOURCE_XML_METADATA_UNMAPPED",
                    "Source XML contains preserved elements or attributes with no current OStim scene-field mapping.",
                    category="unsupported or lost data",
                    source_file=scene.provenance.source_path,
                    object_id=scene.source_id,
                )
                if not any(
                    loss.code == "SOURCE_XML_METADATA_UNMAPPED" and loss.source_object == scene.source_id
                    for loss in ir.losses
                ):
                    ir.losses.append(
                        ConversionLoss(
                            LossKind.UNSUPPORTED,
                            "SOURCE_XML_METADATA_UNMAPPED",
                            "Preserved source XML metadata has no current OStim scene-field mapping.",
                            scene.source_id,
                        )
                    )
                ir.quality = ConversionQuality.LOSSY

        used_ids: set[str] = set()
        scene_map: dict[str, str] = {}
        for scene in sorted(ir.graph.nodes, key=lambda item: item.source_id.casefold()):
            base = _safe_identifier(scene.source_id, "Scene")
            restored_ostim_id = (
                scene.provenance.extras.get("restoredFromManifest")
                and scene.provenance.extras.get("manifestSourceFormat") == SourceFormat.OSTIM_SA.value
            )
            if (ir.source.format != SourceFormat.OSTIM_SA and not restored_ostim_id) or base.casefold().startswith(
                "ostim"
            ):
                base = _safe_identifier(f"{pack_id}_{base}", "Scene")
            target_id = _unique_identifier(base, used_ids, scene.id)
            scene_map[scene.source_id] = target_id
            scene.target_id = target_id

        used_events: set[str] = set()
        event_map: dict[str, str] = {}
        for event in sorted(ir.events, key=lambda item: item.name.casefold()):
            base = _safe_identifier(event.name, "Animation")
            target = _unique_identifier(base, used_events, event.id)
            event_map[event.name] = target

        written: list[Path] = []
        scene_root = destination / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / pack_id
        scene_root.mkdir(parents=True, exist_ok=True)
        for scene in sorted(ir.graph.nodes, key=lambda item: scene_map[item.source_id].casefold()):
            target_id = scene_map[scene.source_id]
            data: dict[str, Any] = {
                "name": scene.name or target_id,
                "modpack": display_name,
                "length": scene.length,
            }
            if scene.is_transition:
                data["destination"] = scene_map.get(scene.transition_destination or "", scene.transition_destination)
                if scene.transition_origin:
                    data["origin"] = scene_map.get(scene.transition_origin, scene.transition_origin)
                _put_optional(data, "priority", scene.transition_priority)
                _put_optional(data, "description", scene.transition_description)
                _put_optional(data, "icon", scene.transition_icon)
                _put_optional(data, "border", scene.transition_border)
                _put_optional(data, "noWarnings", scene.transition_no_warnings, default=False)
            elif scene.navigations:
                data["navigations"] = [_serialize_navigation(edge, scene_map) for edge in scene.navigations]
            data["speeds"] = []
            for speed in scene.speeds:
                speed_data: dict[str, Any] = {"animation": event_map.get(speed.animation, speed.animation)}
                _put_optional(speed_data, "playbackSpeed", speed.playback_speed, default=1.0)
                _put_optional(speed_data, "displaySpeed", speed.display_speed)
                unknown = speed.provenance.extras.get("jsonUnknown")
                if context.preserve_unknown_fields and isinstance(unknown, dict):
                    for key in sorted(unknown):
                        if key not in speed_data:
                            speed_data[key] = unknown[key]
                data["speeds"].append(speed_data)
            _put_optional(data, "defaultSpeed", scene.default_speed, default=0)
            _put_optional(data, "noRandomSelection", scene.no_random_selection, default=False)
            _put_optional(data, "fadeOnEntry", scene.fade_on_entry, default=False)
            _put_optional(data, "furniture", scene.furniture.type, default="none")
            if scene.offset is not None and (scene.offset.was_provided or not scene.offset.is_neutral):
                data["offset"] = _serialize_transform(scene.offset)
            _put_optional(data, "scaleOffsetWithFurniture", scene.scale_offset_with_furniture, default=False)
            if scene.tags:
                data["tags"] = unique_preserving_order(scene.tags)
            if scene.auto_transitions:
                data["autoTransitions"] = {
                    key: scene_map.get(value, value) for key, value in sorted(scene.auto_transitions.items())
                }
            data["actors"] = [_serialize_actor(actor, index, scene_map) for index, actor in enumerate(scene.actors)]
            if scene.actions:
                data["actions"] = [_serialize_action(action) for action in scene.actions]

            if context.preserve_unknown_fields and ir.source.format == SourceFormat.OSTIM_SA:
                unknown = scene.provenance.extras.get("jsonUnknown")
                if isinstance(unknown, dict):
                    for key in sorted(unknown):
                        if key not in data and key not in {"id", "pack", "poses", "clips", "modPack"}:
                            data[key] = unknown[key]
            output = scene_root / f"{target_id}.json"
            output.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            written.append(output)

        if ir.sequences:
            sequence_root = destination / "Data" / "SKSE" / "Plugins" / "OStim" / "sequences"
            sequence_root.mkdir(parents=True, exist_ok=True)
            used_sequences: set[str] = set()
            for sequence in sorted(ir.sequences, key=lambda item: item.source_id.casefold()):
                sequence_id = _unique_identifier(
                    _safe_identifier(sequence.source_id, "Sequence"), used_sequences, sequence.id
                )
                sequence.target_id = sequence_id
                data = {
                    "scenes": [
                        self._serialize_sequence_entry(entry, scene_map, context.preserve_unknown_fields)
                        for entry in sequence.entries
                    ]
                }
                if sequence.tags:
                    data["tags"] = unique_preserving_order(sequence.tags)
                if context.preserve_unknown_fields:
                    unknown = sequence.provenance.extras.get("jsonUnknown")
                    if isinstance(unknown, dict):
                        for key in sorted(unknown):
                            if key not in data:
                                data[key] = unknown[key]
                output = sequence_root / f"{sequence_id}.json"
                output.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                written.append(output)

        return EmitResult(
            destination,
            written,
            diagnostics,
            scene_map,
            event_map,
            install_ready=False,
            output_pack_id=pack_id,
        )

    @staticmethod
    def _serialize_sequence_entry(
        entry: SequenceEntry, reference_map: dict[str, str], preserve_unknown_fields: bool
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"id": reference_map.get(entry.scene_id, entry.scene_id)}
        if entry.duration is not None:
            data["duration"] = entry.duration
        if preserve_unknown_fields:
            unknown = entry.provenance.extras.get("jsonUnknown")
            if isinstance(unknown, dict):
                for key in sorted(unknown):
                    if key not in data:
                        data[key] = unknown[key]
        return data

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            format=self.format,
            import_supported=True,
            export_supported=True,
            round_trip_supported=True,
            supported_actor_counts="Any count accepted by the pinned loader; mappings are validated per event.",
            navigation_support="Full current scene and transition navigation fields.",
            furniture_support="Current furniture type identifiers; custom type definitions are preserved separately.",
            annotation_support="Scene action peak flags; binary HKX annotations are not edited.",
            expected_losses=(
                "Unknown fields are preserved for same-format round trips but may be unsupported cross-format.",
                "Runtime user alignment.json state is not treated as pack metadata.",
            ),
            required_behavior_generator="pandora or nemesis",
        )


def schema_summary() -> dict[str, Any]:
    return {
        "format": SourceFormat.OSTIM_SA.value,
        "schemaCommit": OSTIM_SA_SCHEMA_COMMIT,
        "sceneIdSource": "JSON filename without .json; subdirectories do not namespace IDs",
        "sceneFields": sorted(SCENE_KEYS),
        "speedFields": sorted(SPEED_KEYS),
        "actorFields": sorted(ACTOR_KEYS),
        "actionFields": sorted(ACTION_KEYS),
        "navigationFields": sorted(NAVIGATION_KEYS),
        "offsetFields": ["x", "y", "z", "r"],
        "sequenceFields": {"sequence": ["scenes", "tags"], "entry": ["id", "duration"]},
    }
