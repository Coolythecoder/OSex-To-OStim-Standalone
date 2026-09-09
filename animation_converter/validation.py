"""Cross-format semantic validation and install-readiness calculation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .diagnostics import DiagnosticCollection, Severity
from .graph import validate_graph
from .models import ConversionIR, ConversionMode, ConversionQuality, SceneNode, SourceFormat

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")


@dataclass
class InstallReadiness:
    install_ready: bool
    checks: dict[str, bool] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "installReady": self.install_ready,
            "checks": self.checks,
            "blockers": self.blockers,
        }


def _all_transforms(ir: ConversionIR):
    for scene in ir.graph.nodes:
        if scene.offset:
            yield scene.scene_id, scene.offset
        for actor in scene.actors:
            if actor.offset:
                yield scene.scene_id, actor.offset


def validate_offsets(ir: ConversionIR) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection()
    transforms = list(_all_transforms(ir))
    failed = [
        (scene_id, transform)
        for scene_id, transform in transforms
        if transform.provenance.extras.get("parseFailed")
        or transform.provenance.extras.get("inferredFromMissingSource")
    ]
    if failed and len(ir.graph.nodes) > 1 and all(transform.is_neutral for _, transform in transforms):
        diagnostics.error(
            "OFFSET_ALL_ZERO_PARSE_FAILURE",
            "Every parsed offset is zero and at least one offset failed to parse; alignment conversion is incomplete.",
            category="offsets and alignment",
            remediation="Correct the source offset mapping. Do not generate a zero-filled alignment file.",
            can_continue=False,
        )
    for scene_id, transform in transforms:
        if transform.is_neutral and transform.was_provided:
            diagnostics.info(
                "OFFSET_NEUTRAL_SOURCE_VALUE",
                "A neutral zero offset was explicitly provided by the source and is valid.",
                category="offsets and alignment",
                object_id=scene_id,
            )
    return diagnostics


def validate_scene_shape(scene: SceneNode, *, enforce_ostim_scene_id: bool | None = None) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection()
    target_filename_semantics = (
        scene.provenance.source_format == SourceFormat.OSTIM_SA
        if enforce_ostim_scene_id is None
        else enforce_ostim_scene_id
    )
    if target_filename_semantics and not SAFE_ID.fullmatch(scene.scene_id):
        diagnostics.error(
            "SCENE_ID_UNSAFE",
            f"Scene ID {scene.scene_id!r} cannot be used as a portable filename.",
            category="schema shape",
            object_id=scene.scene_id,
        )
    if (
        target_filename_semantics
        and scene.scene_id.casefold().startswith("ostim")
        and not scene.provenance.extras.get("officialOStimNamespace")
    ):
        diagnostics.error(
            "SCENE_ID_RESERVED_PREFIX",
            f"Third-party scene ID {scene.scene_id!r} uses the reserved OStim prefix.",
            category="schema shape",
            object_id=scene.scene_id,
            remediation="Prefix the scene with the animation pack's own identifier.",
        )
    if scene.length <= 0:
        diagnostics.error(
            "SCENE_LENGTH_INVALID",
            "Scene length must be greater than zero.",
            category="schema shape",
            object_id=scene.scene_id,
        )
    if not scene.speeds:
        diagnostics.error(
            "SCENE_SPEEDS_EMPTY",
            "Ordinary and transition scenes require at least one animation speed.",
            category="speed validity",
            object_id=scene.scene_id,
            can_continue=False,
        )
    elif not 0 <= scene.default_speed < len(scene.speeds):
        diagnostics.error(
            "SCENE_DEFAULT_SPEED_RANGE",
            f"defaultSpeed {scene.default_speed} is outside the speeds array.",
            category="speed validity",
            object_id=scene.scene_id,
        )
    if not scene.actors:
        diagnostics.error(
            "SCENE_ACTORS_EMPTY",
            "Scene has no actor slots.",
            category="actor index validity",
            object_id=scene.scene_id,
        )
    actor_count = len(scene.actors)
    for actor in scene.actors:
        if actor.index < 0 or actor.effective_animation_index < 0:
            diagnostics.error(
                "ACTOR_ANIMATION_INDEX_INVALID",
                f"Actor {actor.index} has invalid animationIndex {actor.effective_animation_index}.",
                category="actor index validity",
                object_id=scene.scene_id,
            )
    for action in scene.actions:
        indexes = [action.actor, action.target, action.performer]
        if any(index is not None and not 0 <= index < actor_count for index in indexes):
            diagnostics.error(
                "ACTION_ACTOR_INDEX_INVALID",
                f"Action {action.type!r} references an actor outside 0..{max(0, actor_count - 1)}.",
                category="action validity",
                object_id=scene.scene_id,
            )
    return diagnostics


def validate_event_assets(ir: ConversionIR) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection()
    event_groups: dict[str, list] = {}
    for event in ir.events:
        event_groups.setdefault(event.name.casefold(), []).append(event)
    for events in event_groups.values():
        if len(events) > 1:
            diagnostics.error(
                "EVENT_NAME_DUPLICATE",
                f"Animation event name {events[0].name!r} is ambiguous.",
                category="animation event validity",
                remediation="Use a unique event base for every speed variant.",
            )
    by_name = {name: events[0] for name, events in event_groups.items() if len(events) == 1}
    assets = {asset.id: asset for asset in ir.assets}
    for scene in ir.graph.nodes:
        for speed in scene.speeds:
            event = by_name.get(speed.animation.casefold())
            if event is None:
                diagnostics.error(
                    "EVENT_MAPPING_MISSING",
                    f"Speed {speed.animation!r} has no animation-event mapping.",
                    category="animation event validity",
                    object_id=scene.scene_id,
                    can_continue=False,
                )
                continue
            for actor in scene.actors:
                index = actor.effective_animation_index
                asset_id = event.actor_assets.get(index)
                if not asset_id:
                    diagnostics.error(
                        "EVENT_ACTOR_ASSET_MISSING",
                        f"Event {event.name!r} has no HKX mapping for animation index {index}.",
                        category="asset existence",
                        object_id=scene.scene_id,
                        can_continue=False,
                    )
                elif asset_id not in assets:
                    diagnostics.error(
                        "EVENT_ASSET_REFERENCE_DANGLING",
                        f"Event {event.name!r} references unknown asset {asset_id!r}.",
                        category="asset existence",
                        object_id=scene.scene_id,
                        can_continue=False,
                    )
    return diagnostics


def validate_sequences(ir: ConversionIR) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection()
    sequence_ids: set[str] = set()
    for sequence in ir.sequences:
        sequence_id = (sequence.target_id or sequence.source_id).casefold()
        if sequence_id in sequence_ids:
            diagnostics.error(
                "SEQUENCE_ID_DUPLICATE",
                f"Duplicate sequence ID {sequence.target_id or sequence.source_id!r}.",
                category="duplicate IDs",
            )
        sequence_ids.add(sequence_id)
        if not sequence.entries:
            diagnostics.error(
                "SEQUENCE_EMPTY",
                f"Sequence {sequence.target_id or sequence.source_id!r} is empty.",
                category="schema shape",
            )
        actor_count: int | None = None
        furniture: str | None = None
        scene_by_id = {scene.scene_id.casefold(): scene for scene in ir.graph.nodes}
        for entry in sequence.entries:
            node = scene_by_id.get(entry.scene_id.casefold())
            if not node:
                diagnostics.error(
                    "SEQUENCE_SCENE_MISSING",
                    f"Sequence references missing scene {entry.scene_id!r}.",
                    category="scene graph integrity",
                )
                continue
            if actor_count is None:
                actor_count = len(node.actors)
                furniture = node.furniture.type.casefold()
            elif len(node.actors) != actor_count:
                diagnostics.error(
                    "SEQUENCE_ACTOR_COUNT_MISMATCH",
                    f"Sequence scene {entry.scene_id!r} has a different actor count.",
                    category="actor index validity",
                )
            elif furniture not in {"none", node.furniture.type.casefold()} and node.furniture.type.casefold() != "none":
                diagnostics.error(
                    "SEQUENCE_FURNITURE_MISMATCH",
                    f"Sequence scene {entry.scene_id!r} has incompatible furniture.",
                    category="furniture validity",
                )
    return diagnostics


def validate_ir(
    ir: ConversionIR,
    *,
    include_assets: bool = True,
    enforce_ostim_scene_ids: bool | None = None,
) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection(ir.diagnostics)
    if not ir.graph.nodes:
        diagnostics.error(
            "CONVERSION_ZERO_SCENES",
            "No valid scene metadata was parsed.",
            category="schema shape",
            remediation="Use a supported metadata source. HKX files alone are not scene definitions.",
            can_continue=False,
        )
    for scene in ir.graph.nodes:
        diagnostics.extend(validate_scene_shape(scene, enforce_ostim_scene_id=enforce_ostim_scene_ids))
    diagnostics.extend(validate_graph(ir))
    diagnostics.extend(validate_offsets(ir))
    diagnostics.extend(validate_sequences(ir))
    if include_assets:
        diagnostics.extend(validate_event_assets(ir))
    return diagnostics


def calculate_install_readiness(
    ir: ConversionIR,
    diagnostics: DiagnosticCollection,
    *,
    behavior_complete: bool,
    layout_valid: bool,
) -> InstallReadiness:
    checks = {
        "hasValidScene": bool(ir.graph.nodes) and not any(scene.salvaged for scene in ir.graph.nodes),
        "schemaAndGraphValid": not diagnostics.has_errors,
        "animationAssetsMapped": not any(
            item.code.startswith("EVENT_") or item.code.startswith("ASSET_")
            for item in diagnostics
            if item.severity >= Severity.ERROR
        ),
        "behaviorRegistrationComplete": behavior_complete,
        "installLayoutValid": layout_valid,
        "notSalvage": ir.quality != ConversionQuality.SALVAGE,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return InstallReadiness(not blockers, checks, blockers)


def mode_allows_output(mode: ConversionMode, diagnostics: DiagnosticCollection, ir: ConversionIR) -> bool:
    if diagnostics.has_fatal:
        return False
    if mode == ConversionMode.STRICT:
        return not diagnostics.has_warnings and not ir.losses
    if mode == ConversionMode.NORMAL:
        return not diagnostics.has_errors
    if mode == ConversionMode.BEST_EFFORT:
        return bool(ir.graph.nodes) and not diagnostics.has_fatal
    return bool(ir.graph.nodes)
