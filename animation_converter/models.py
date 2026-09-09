"""Typed, framework-neutral conversion model."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from .diagnostics import DiagnosticCollection


class SourceFormat(str, Enum):
    AUTO = "auto"
    OSA_OSEX = "osa-osex"
    OSTIM_SA = "ostim-sa"
    OSTIM_LEGACY = "ostim-legacy"
    SLAL = "slal"
    FLOWERGIRLS = "flowergirls"


class ConversionMode(str, Enum):
    NORMAL = "normal"
    STRICT = "strict"
    BEST_EFFORT = "best-effort"
    SALVAGE = "salvage"


class ConversionQuality(str, Enum):
    EXACT = "exact"
    LOSSY = "structurally-valid-lossy"
    SALVAGE = "asset-only-salvage"
    UNSUPPORTED = "unsupported"


class LossKind(str, Enum):
    GENERATED = "generated"
    DISCARDED = "discarded"
    UNSUPPORTED = "unsupported"
    INFERRED = "inferred"


@dataclass
class Provenance:
    source_path: str | None = None
    source_identifier: str | None = None
    source_format: SourceFormat | None = None
    location: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    unknown_elements: list[str] = field(default_factory=list)
    normalizations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.source_path:
            data["sourcePath"] = self.source_path
        if self.source_identifier:
            data["sourceIdentifier"] = self.source_identifier
        if self.source_format:
            data["sourceFormat"] = self.source_format.value
        if self.location:
            data["location"] = self.location
        if self.extras:
            data["extras"] = to_plain(self.extras)
        if self.unknown_elements:
            data["unknownElements"] = list(self.unknown_elements)
        if self.normalizations:
            data["normalizations"] = list(self.normalizations)
        if self.warnings:
            data["warnings"] = list(self.warnings)
        return data


def stable_internal_id(source_format: SourceFormat, source_path: str, source_id: str) -> str:
    key = f"{source_format.value}\0{source_path.replace(chr(92), '/')}\0{source_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass
class PackMetadata:
    identifier: str
    display_name: str
    author: str | None = None
    version: str | None = None
    credits: list[str] = field(default_factory=list)
    license_files: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class SourceDescriptor:
    format: SourceFormat
    root_name: str
    detected_version: str | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    detection_evidence: list[str] = field(default_factory=list)


@dataclass
class Transform:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    provided_fields: set[str] = field(default_factory=set)
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def is_neutral(self) -> bool:
        return self.x == self.y == self.z == self.r == 0.0

    @property
    def was_provided(self) -> bool:
        return bool(self.provided_fields)

    def to_ostim_dict(self) -> dict[str, float]:
        values = {"x": self.x, "y": self.y, "z": self.z, "r": self.r}
        return {key: value for key, value in values.items() if key in self.provided_fields or value != 0.0}


@dataclass
class AnimationAsset:
    id: str
    source_path: str
    target_path: str | None = None
    sha256: str | None = None
    size: int | None = None
    event_name: str | None = None
    actor_index: int | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class AnimationEvent:
    id: str
    name: str
    actor_assets: dict[int, str] = field(default_factory=dict)
    annotations: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class SpeedVariant:
    animation: str
    playback_speed: float | None = None
    display_speed: float | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class ActorSlot:
    index: int
    type: str = "npc"
    intended_sex: str | None = None
    sos_bend: int | None = None
    tng_bend: int | None = None
    scale: float | None = None
    scale_height: float | None = None
    animation_index: int | None = None
    tags: list[str] = field(default_factory=list)
    feet_on_ground: bool | None = None
    no_strip: bool | None = None
    offset: Transform | None = None
    requirements: list[str] = field(default_factory=list)
    auto_transitions: dict[str, str] = field(default_factory=dict)
    underlying_expression: str | None = None
    expression_action: int | None = None
    expression_override: str | None = None
    look_up: int | None = None
    look_down: int | None = None
    look_left: int | None = None
    look_right: int | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def effective_animation_index(self) -> int:
        return self.index if self.animation_index is None else self.animation_index


@dataclass
class SceneAction:
    type: str
    actor: int
    target: int | None = None
    performer: int | None = None
    muted: bool | None = None
    do_peaks: bool | None = None
    peaks_annotated: bool | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class NavigationEdge:
    destination: str | None = None
    origin: str | None = None
    priority: int | None = None
    description: str | None = None
    icon: str | None = None
    border: str | None = None
    no_warnings: bool | None = None
    generated: bool = False
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class FurnitureRequirement:
    type: str = "none"
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class Annotation:
    kind: str
    value: str
    actor: int | None = None
    time: float | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class SceneNode:
    id: str
    source_id: str
    name: str
    modpack: str
    length: float
    speeds: list[SpeedVariant]
    actors: list[ActorSlot]
    relative_directory: str = ""
    target_id: str | None = None
    navigations: list[NavigationEdge] = field(default_factory=list)
    default_speed: int = 0
    no_random_selection: bool | None = None
    fade_on_entry: bool | None = None
    furniture: FurnitureRequirement = field(default_factory=FurnitureRequirement)
    offset: Transform | None = None
    scale_offset_with_furniture: bool | None = None
    tags: list[str] = field(default_factory=list)
    auto_transitions: dict[str, str] = field(default_factory=dict)
    actions: list[SceneAction] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    transition_destination: str | None = None
    transition_origin: str | None = None
    transition_priority: int | None = None
    transition_description: str | None = None
    transition_icon: str | None = None
    transition_border: str | None = None
    transition_no_warnings: bool | None = None
    salvaged: bool = False
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def scene_id(self) -> str:
        return self.target_id or self.source_id

    @property
    def is_transition(self) -> bool:
        return self.transition_destination is not None


@dataclass
class TransitionNode(SceneNode):
    """Marker type for a scene with OStim transition semantics."""


@dataclass
class SceneGraph:
    nodes: list[SceneNode] = field(default_factory=list)

    def by_scene_id(self) -> dict[str, SceneNode]:
        return {node.scene_id: node for node in self.nodes}


@dataclass
class SequenceEntry:
    scene_id: str
    duration: float | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class Sequence:
    id: str
    source_id: str
    entries: list[SequenceEntry]
    tags: list[str] = field(default_factory=list)
    target_id: str | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class BehaviorRegistration:
    event_name: str
    actor_index: int
    asset_id: str
    writer: str | None = None
    verified: bool = False
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class ConversionLoss:
    kind: LossKind
    code: str
    message: str
    source_object: str | None = None
    target_object: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "code": self.code,
            "message": self.message,
        }
        if self.source_object:
            result["sourceObject"] = self.source_object
        if self.target_object:
            result["targetObject"] = self.target_object
        return result


@dataclass
class ConversionManifest:
    converter_version: str
    source_framework: SourceFormat
    target_framework: SourceFormat
    target_schema_version: str
    source_file_hashes: dict[str, str] = field(default_factory=dict)
    source_pack_identifier: str = ""
    output_pack_identifier: str = ""
    scene_id_map: dict[str, dict[str, str]] = field(default_factory=dict)
    animation_event_map: dict[str, str] = field(default_factory=dict)
    asset_path_map: dict[str, str] = field(default_factory=dict)
    preserved_provenance: dict[str, Any] = field(default_factory=dict)
    generated_values: list[dict[str, Any]] = field(default_factory=list)
    discarded_values: list[dict[str, Any]] = field(default_factory=list)
    unsupported_values: list[dict[str, Any]] = field(default_factory=list)
    inferred_values: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conversion_timestamp: str = "1980-01-01T00:00:00Z"
    deterministic_build: bool = True
    deterministic_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "converterVersion": self.converter_version,
            "sourceFramework": self.source_framework.value,
            "targetFramework": self.target_framework.value,
            "targetSchemaVersion": self.target_schema_version,
            "sourceFileHashes": dict(sorted(self.source_file_hashes.items())),
            "sourcePackIdentifier": self.source_pack_identifier,
            "outputPackIdentifier": self.output_pack_identifier,
            "sceneIdMappings": dict(sorted(self.scene_id_map.items())),
            "animationEventMappings": dict(sorted(self.animation_event_map.items())),
            "assetPathMappings": dict(sorted(self.asset_path_map.items())),
            "preservedSourceProvenance": self.preserved_provenance,
            "generatedValues": self.generated_values,
            "discardedValues": self.discarded_values,
            "unsupportedValues": self.unsupported_values,
            "inferredValues": self.inferred_values,
            "warnings": self.warnings,
            "conversionTimestamp": self.conversion_timestamp,
            "deterministicBuild": self.deterministic_build,
            "deterministicBuildMetadata": self.deterministic_metadata,
        }


@dataclass
class ConversionIR:
    pack: PackMetadata
    source: SourceDescriptor
    graph: SceneGraph = field(default_factory=SceneGraph)
    events: list[AnimationEvent] = field(default_factory=list)
    assets: list[AnimationAsset] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    behavior_registrations: list[BehaviorRegistration] = field(default_factory=list)
    losses: list[ConversionLoss] = field(default_factory=list)
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)
    manifest: ConversionManifest | None = None
    quality: ConversionQuality = ConversionQuality.EXACT

    def scene_ids(self) -> list[str]:
        return [node.scene_id for node in self.graph.nodes]

    def semantic_dict(self) -> dict[str, Any]:
        """Return normalized semantics, excluding paths and provenance noise."""
        scenes: list[dict[str, Any]] = []
        for scene in sorted(self.graph.nodes, key=lambda item: item.scene_id.casefold()):
            scenes.append(
                {
                    "id": scene.scene_id,
                    "name": scene.name,
                    "length": scene.length,
                    "speeds": [(speed.animation, speed.playback_speed, speed.display_speed) for speed in scene.speeds],
                    "defaultSpeed": scene.default_speed,
                    "actors": [
                        {
                            "type": actor.type,
                            "sex": actor.intended_sex,
                            "animationIndex": actor.effective_animation_index,
                            "tags": sorted(actor.tags),
                            "offset": actor.offset.to_ostim_dict() if actor.offset else None,
                        }
                        for actor in scene.actors
                    ],
                    "navigations": sorted(
                        [
                            (
                                edge.origin,
                                edge.destination,
                                edge.priority,
                                edge.description,
                                edge.icon,
                                edge.border,
                            )
                            for edge in scene.navigations
                        ],
                        key=lambda item: json.dumps(item, separators=(",", ":")),
                    ),
                    "transition": (scene.transition_origin, scene.transition_destination),
                    "furniture": scene.furniture.type,
                    "tags": sorted(scene.tags),
                    "actions": sorted(
                        [
                            (
                                action.type,
                                action.actor,
                                action.target,
                                action.performer,
                                action.muted,
                                action.do_peaks,
                                action.peaks_annotated,
                            )
                            for action in scene.actions
                        ],
                        key=lambda item: json.dumps(item, separators=(",", ":")),
                    ),
                }
            )
        return {
            "scenes": scenes,
            "sequences": [
                {
                    "id": sequence.target_id or sequence.source_id,
                    "entries": [(entry.scene_id, entry.duration) for entry in sequence.entries],
                    "tags": sorted(sequence.tags),
                }
                for sequence in sorted(self.sequences, key=lambda item: item.source_id.casefold())
            ],
        }


def to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    if isinstance(value, Provenance):
        return value.to_dict()
    if isinstance(value, set):
        return sorted(to_plain(item) for item in value)
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def semantic_digest(ir: ConversionIR) -> str:
    payload = json.dumps(ir.semantic_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
