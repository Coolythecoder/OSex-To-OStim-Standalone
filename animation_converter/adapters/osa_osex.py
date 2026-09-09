"""Explicit OSA, OSex, OSex+, and OpenSex XML adapter."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from ..assets import discover_animation_assets, link_events_to_assets, split_actor_suffix
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

SCENE_TAGS = {"scene", "scenedata", "osexscene", "sexscene"}
MODULE_TAGS = {"module", "osex", "osex+", "opensex", "osa"}
STAGE_TAGS = {"stage", "pose", "position"}
ANIMATION_TAGS = {"anim", "animation", "clip"}
KNOWN_SCENE_CHILDREN = {
    "info",
    "anim",
    "speed",
    "nav",
    "metadata",
    "actors",
    "actions",
    "stage",
    "pose",
    "position",
    "tags",
    "navigation",
}
SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.+!-]+")


def local_name(element: ET.Element) -> str:
    return str(element.tag).split("}")[-1].casefold()


def _children(element: ET.Element, *names: str) -> list[ET.Element]:
    wanted = {name.casefold() for name in names}
    return [child for child in list(element) if local_name(child) in wanted]


def _descendants(element: ET.Element, *names: str) -> list[ET.Element]:
    wanted = {name.casefold() for name in names}
    return [child for child in element.iter() if child is not element and local_name(child) in wanted]


def _first_child(element: ET.Element, *names: str) -> ET.Element | None:
    matches = _children(element, *names)
    return matches[0] if matches else None


def _attribute(element: ET.Element | None, *names: str) -> str | None:
    if element is None:
        return None
    by_folded = {key.casefold(): value for key, value in element.attrib.items()}
    for name in names:
        value = by_folded.get(name.casefold())
        if value is not None:
            return value
    return None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    folded = value.strip().casefold()
    if folded in {"1", "true", "yes", "on"}:
        return True
    if folded in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _split_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r"[,;|]", value) if item.strip()]


def _relative(path: Path, root: Path) -> str:
    try:
        return PurePosixPath(*path.relative_to(root).parts).as_posix()
    except ValueError:
        return path.name


def _is_installer_xml(path: Path) -> bool:
    folded = {part.casefold() for part in path.parts}
    return "fomod" in folded or path.name.casefold() in {"moduleconfig.xml", "info.xml"}


def _is_osa_scene_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    has_framework = any(part in {"0sa", "osa", "osa+", "0sex", "osex", "osex+", "opensex"} for part in parts)
    has_scene = any(part in {"scene", "scenes"} for part in parts[:-1])
    return has_framework and has_scene and not _is_installer_xml(path)


def discover_osa_xml_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.casefold() == ".xml" and not _is_installer_xml(root) else []
    xml_files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".xml" and not _is_installer_xml(path)
        ),
        key=lambda item: str(item).casefold(),
    )
    path_matches = [path for path in xml_files if _is_osa_scene_path(path)]
    if path_matches:
        return path_matches
    if root.name.casefold() in {"scene", "scenes"}:
        return xml_files
    explicit: list[Path] = []
    for path in xml_files:
        try:
            document = ET.parse(path)
        except (OSError, ET.ParseError):
            continue
        root_name = local_name(document.getroot())
        if root_name in SCENE_TAGS | MODULE_TAGS:
            explicit.append(path)
    return explicit


def scene_reference_base(source_id: str) -> str:
    if "|" in source_id:
        head, tail = source_id.rsplit("|", 1)
        return f"{head}|{tail.split('+', 1)[0]}"
    return source_id.split("+", 1)[0]


def resolve_relative_reference(reference: str, current_source_id: str) -> str:
    value = reference.strip()
    if not value or not current_source_id:
        return value
    if value.startswith("^"):
        return scene_reference_base(current_source_id) + value[1:]
    if value.startswith("+"):
        return scene_reference_base(current_source_id) + value
    return value


def _source_line(element: ET.Element) -> str | None:
    line = getattr(element, "sourceline", None)
    return f"line {line}" if line else None


def _unknown_attributes(element: ET.Element, known: set[str]) -> dict[str, str]:
    folded = {item.casefold() for item in known}
    return {key: value for key, value in element.attrib.items() if key.casefold() not in folded}


def _unknown_elements(scene: ET.Element) -> list[str]:
    result: list[str] = []
    for child in list(scene):
        if local_name(child) not in KNOWN_SCENE_CHILDREN:
            result.append(ET.tostring(child, encoding="unicode", short_empty_elements=True))
    return result


def _animation_event(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip().strip('"').strip("'").replace("\\", "/")
    if "/" in cleaned or cleaned.casefold().endswith(".hkx"):
        stem = PurePosixPath(cleaned).stem
        return split_actor_suffix(stem)[0]
    return cleaned


def _transform_from_attributes(
    element: ET.Element,
    diagnostics: DiagnosticCollection,
    source_file: str,
    source_id: str,
    location: str,
) -> Transform | None:
    aliases = {"x": ("x", "posX"), "y": ("y", "posY"), "z": ("z", "posZ"), "r": ("r", "rotation", "rz", "rotZ")}
    transform = Transform(
        provenance=Provenance(
            source_path=source_file,
            source_identifier=source_id,
            source_format=SourceFormat.OSA_OSEX,
            location=location,
        )
    )
    found = False
    for field_name, names in aliases.items():
        raw = _attribute(element, *names)
        if raw is None:
            continue
        found = True
        parsed = _parse_float(raw)
        if parsed is None:
            transform.provenance.extras["parseFailed"] = True
            diagnostics.error(
                "OSA_OFFSET_VALUE_INVALID",
                f"{location} has a non-numeric {field_name} offset.",
                category="offsets and alignment",
                source_file=source_file,
                object_id=source_id,
            )
            continue
        setattr(transform, field_name, parsed)
        transform.provided_fields.add(field_name)
    rx = _parse_float(_attribute(element, "rx", "rotX"))
    ry = _parse_float(_attribute(element, "ry", "rotY"))
    if rx not in {None, 0.0} or ry not in {None, 0.0}:
        transform.provenance.extras["unsupportedEulerRotation"] = {"rx": rx, "ry": ry}
        diagnostics.warning(
            "OSA_OFFSET_EULER_UNSUPPORTED",
            "Source rx/ry rotation cannot be represented by OStim's single r rotation; values were preserved in provenance.",
            category="unsupported or lost data",
            source_file=source_file,
            object_id=source_id,
        )
    return transform if found or rx is not None or ry is not None else None


def _parse_actor_elements(
    scene_element: ET.Element,
    actor_count: int,
    diagnostics: DiagnosticCollection,
    source_file: str,
    source_id: str,
) -> list[ActorSlot]:
    actors = [ActorSlot(index=index) for index in range(max(0, actor_count))]
    actor_containers = _children(scene_element, "actors")
    actor_elements = _children(scene_element, "actor", "slot", "role") + [
        actor for container in actor_containers for actor in _children(container, "actor", "slot", "role")
    ]
    for actor_element in actor_elements:
        index = _parse_int(_attribute(actor_element, "index", "actorIndex", "position", "slot", "a"))
        if index is None or index < 0:
            diagnostics.error(
                "OSA_ACTOR_INDEX_INVALID",
                "Actor metadata has no valid index.",
                category="actor index validity",
                source_file=source_file,
                object_id=source_id,
            )
            continue
        while len(actors) <= index:
            actors.append(ActorSlot(index=len(actors)))
        actor = actors[index]
        actor.type = _attribute(actor_element, "type") or actor.type
        actor.intended_sex = _attribute(actor_element, "intendedSex", "sex")
        actor.sos_bend = _parse_int(_attribute(actor_element, "sosBend", "penisAngle"))
        actor.scale = _parse_float(_attribute(actor_element, "scale"))
        actor.scale_height = _parse_float(_attribute(actor_element, "scaleHeight"))
        actor.animation_index = _parse_int(_attribute(actor_element, "animationIndex", "animation"))
        actor.tags = _split_values(_attribute(actor_element, "tags"))
        actor.feet_on_ground = _parse_bool(_attribute(actor_element, "feetOnGround"))
        actor.no_strip = _parse_bool(_attribute(actor_element, "noStrip"))
        actor.offset = _transform_from_attributes(
            actor_element, diagnostics, source_file, source_id, f"actors[{index}].offset"
        )
        actor.requirements = _split_values(_attribute(actor_element, "requirements"))
        actor.provenance = Provenance(
            source_path=source_file,
            source_identifier=f"{source_id}:actor:{index}",
            source_format=SourceFormat.OSA_OSEX,
            location=_source_line(actor_element),
            extras={
                "xmlUnknownAttributes": _unknown_attributes(
                    actor_element,
                    {
                        "index",
                        "actorIndex",
                        "position",
                        "slot",
                        "a",
                        "type",
                        "intendedSex",
                        "sex",
                        "sosBend",
                        "penisAngle",
                        "scale",
                        "scaleHeight",
                        "animationIndex",
                        "animation",
                        "tags",
                        "feetOnGround",
                        "noStrip",
                        "requirements",
                        "x",
                        "y",
                        "z",
                        "r",
                        "posX",
                        "posY",
                        "posZ",
                        "rotation",
                        "rx",
                        "ry",
                        "rz",
                        "rotX",
                        "rotY",
                        "rotZ",
                    },
                )
            },
        )
        for transition in _children(actor_element, "autotransition"):
            transition_type = _attribute(transition, "type", "event")
            destination = _attribute(transition, "destination", "dest", "to", "go")
            if transition_type and destination:
                actor.auto_transitions[transition_type.casefold()] = resolve_relative_reference(destination, source_id)
    return actors


def _parse_actions(scene_element: ET.Element, source_file: str, source_id: str) -> list[SceneAction]:
    actions: list[SceneAction] = []
    for container in _children(scene_element, "actions"):
        for index, action_element in enumerate(_children(container, "action")):
            action_type = _attribute(action_element, "type", "name")
            actor = _parse_int(_attribute(action_element, "actor", "actorIndex", "position"))
            if not action_type or actor is None:
                continue
            actions.append(
                SceneAction(
                    type=action_type.casefold(),
                    actor=actor,
                    target=_parse_int(_attribute(action_element, "target", "targetActor")),
                    performer=_parse_int(_attribute(action_element, "performer", "performerActor")),
                    muted=_parse_bool(_attribute(action_element, "muted")),
                    do_peaks=_parse_bool(_attribute(action_element, "doPeaks")),
                    peaks_annotated=_parse_bool(_attribute(action_element, "peaksAnnotated")),
                    provenance=Provenance(
                        source_path=source_file,
                        source_identifier=f"{source_id}:action:{index}",
                        source_format=SourceFormat.OSA_OSEX,
                        location=_source_line(action_element),
                        extras={
                            "xmlUnknownAttributes": _unknown_attributes(
                                action_element,
                                {
                                    "type",
                                    "name",
                                    "actor",
                                    "actorIndex",
                                    "position",
                                    "target",
                                    "targetActor",
                                    "performer",
                                    "performerActor",
                                    "muted",
                                    "doPeaks",
                                    "peaksAnnotated",
                                },
                            )
                        },
                    ),
                )
            )
    return actions


def _parse_navigations(scene_element: ET.Element, source_file: str, source_id: str) -> list[NavigationEdge]:
    navigations: list[NavigationEdge] = []
    options = _descendants(scene_element, "option", "navigation", "link")
    for index, option in enumerate(options):
        destination = _attribute(option, "go", "destination", "dest", "to")
        origin = _attribute(option, "origin", "from")
        if destination:
            destination = resolve_relative_reference(destination, source_id)
        if origin:
            origin = resolve_relative_reference(origin, source_id)
        if not destination and not origin:
            continue
        navigations.append(
            NavigationEdge(
                destination=destination,
                origin=origin,
                priority=_parse_int(_attribute(option, "priority")),
                description=_attribute(option, "description", "text", "name"),
                icon=_attribute(option, "icon"),
                border=_attribute(option, "border"),
                no_warnings=_parse_bool(_attribute(option, "noWarnings")),
                provenance=Provenance(
                    source_path=source_file,
                    source_identifier=f"{source_id}:navigation:{index}",
                    source_format=SourceFormat.OSA_OSEX,
                    location=_source_line(option),
                    extras={
                        "xmlUnknownAttributes": _unknown_attributes(
                            option,
                            {
                                "go",
                                "destination",
                                "dest",
                                "to",
                                "origin",
                                "from",
                                "priority",
                                "description",
                                "text",
                                "name",
                                "icon",
                                "border",
                                "noWarnings",
                            },
                        )
                    },
                ),
            )
        )
    return navigations


def _speed_variants(scene_element: ET.Element, source_file: str, source_id: str) -> list[SpeedVariant]:
    speeds: list[SpeedVariant] = []
    speed_containers = _children(scene_element, "speed", "speeds")
    for container in speed_containers:
        variants = _children(container, "sp", "speed", "variant")
        for index, variant in enumerate(variants):
            animation_element = _first_child(variant, "anim", "animation", "clip")
            if animation_element is None:
                animation_element = variant
            event = _animation_event(
                _attribute(animation_element, "event", "id", "name", "animation", "file", "path", "hkx")
            )
            if not event:
                continue
            speeds.append(
                SpeedVariant(
                    animation=event,
                    playback_speed=_parse_float(_attribute(animation_element, "playbackSpeed", "playback_speed")),
                    display_speed=_parse_float(_attribute(variant, "qnt", "displaySpeed", "display_speed")),
                    provenance=Provenance(
                        source_path=source_file,
                        source_identifier=event,
                        source_format=SourceFormat.OSA_OSEX,
                        location=_source_line(variant) or f"speed[{index}]",
                    ),
                )
            )
    if speeds:
        return speeds
    direct_animations = _children(scene_element, "anim", "animation", "clip")
    direct_seen: set[str] = set()
    for index, animation_element in enumerate(direct_animations):
        transition_type = (_attribute(animation_element, "t", "type", "transition") or "").casefold()
        if transition_type in {"t", "transition", "true", "1", "yes"}:
            continue
        event = _animation_event(
            _attribute(animation_element, "event", "id", "name", "animation", "file", "path", "hkx")
        )
        event, _ = split_actor_suffix(event)
        if event and event.casefold() not in direct_seen:
            direct_seen.add(event.casefold())
            speeds.append(
                SpeedVariant(
                    animation=event,
                    playback_speed=_parse_float(_attribute(animation_element, "playbackSpeed", "playback_speed")),
                    display_speed=_parse_float(_attribute(animation_element, "displaySpeed", "display_speed")),
                    provenance=Provenance(
                        source_path=source_file,
                        source_identifier=event,
                        source_format=SourceFormat.OSA_OSEX,
                        location=_source_line(animation_element) or f"anim[{index}]",
                    ),
                )
            )
    if speeds:
        return speeds

    # Namespaced module/stage dialects commonly nest one animation beneath
    # each actor. A shared suffix-free event name is one OStim speed.
    descendant_animations = _descendants(scene_element, "anim", "animation", "clip")
    event_bases: list[str] = []
    for animation_element in descendant_animations:
        event = _animation_event(
            _attribute(animation_element, "event", "id", "name", "animation", "file", "path", "hkx")
        )
        if event:
            base, _ = split_actor_suffix(event)
            if base.casefold() not in {item.casefold() for item in event_bases}:
                event_bases.append(base)
    return [
        SpeedVariant(
            animation=event,
            provenance=Provenance(
                source_path=source_file,
                source_identifier=event,
                source_format=SourceFormat.OSA_OSEX,
                location="nested actor animation",
            ),
        )
        for event in event_bases
    ]


def _bind_explicit_animation_paths(
    scene_element: ET.Element,
    scene: SceneNode,
    assets_by_path: dict[str, list],
) -> None:
    animation_elements = _descendants(scene_element, "anim", "animation", "clip")
    for animation in animation_elements:
        file_value = _attribute(animation, "file", "path", "hkx")
        if not file_value:
            continue
        normalized = file_value.replace("\\", "/").casefold()
        basename = PurePosixPath(normalized).name
        candidates = assets_by_path.get(normalized, []) or assets_by_path.get(basename, [])
        actor_index = _parse_int(_attribute(animation, "actorIndex", "actor", "index", "slot"))
        event = _animation_event(_attribute(animation, "event", "id", "name", "animation") or file_value)
        if actor_index is None:
            _, actor_index = split_actor_suffix(PurePosixPath(file_value.replace("\\", "/")).stem)
        if actor_index is None:
            actor_index = 0
        for asset in candidates:
            asset.event_name = event
            asset.actor_index = actor_index

    for actor_element in _descendants(scene_element, "actor", "slot", "role"):
        parent_index = _parse_int(_attribute(actor_element, "index", "actorIndex", "position", "slot", "a"))
        if parent_index is None:
            continue
        for animation in _descendants(actor_element, "anim", "animation", "clip"):
            file_value = _attribute(animation, "file", "path", "hkx")
            if not file_value:
                continue
            normalized = file_value.replace("\\", "/").casefold()
            candidates = assets_by_path.get(normalized, []) or assets_by_path.get(PurePosixPath(normalized).name, [])
            event = _animation_event(_attribute(animation, "event", "id", "name", "animation") or file_value)
            event, suffix_index = split_actor_suffix(event)
            actor_index = parent_index if suffix_index is None else suffix_index
            for asset in candidates:
                asset.event_name = event
                asset.actor_index = actor_index


def _metadata(scene_element: ET.Element) -> tuple[list[str], str, bool | None]:
    metadata = _first_child(scene_element, "metadata")
    tags = _split_values(_attribute(metadata, "tags")) if metadata is not None else []
    tag_container = _first_child(scene_element, "tags")
    if tag_container is not None:
        for tag in _children(tag_container, "tag"):
            value = _attribute(tag, "name", "value") or (tag.text or "").strip()
            if value:
                tags.append(value)
    furniture = _attribute(metadata, "furniture", "furnitureType") or _attribute(scene_element, "furniture") or "none"
    no_random = _parse_bool(_attribute(metadata, "noRandomSelection"))
    return unique_preserving_order(tags), furniture, no_random


def _parse_single_scene(
    element: ET.Element,
    source_file: str,
    fallback_id: str,
    display_pack: str,
    diagnostics: DiagnosticCollection,
    *,
    source_id_override: str | None = None,
    name_override: str | None = None,
) -> SceneNode | None:
    source_id = source_id_override or _attribute(element, "id", "scene_id", "sceneId") or fallback_id
    info = _first_child(element, "info")
    name = name_override or _attribute(info, "name", "title") or _attribute(element, "name", "title") or source_id
    actor_count = _parse_int(_attribute(element, "actors", "actorCount", "actor_count"))
    actor_count = 2 if actor_count is None else max(0, actor_count)
    actors = _parse_actor_elements(element, actor_count, diagnostics, source_file, source_id)
    speeds = _speed_variants(element, source_file, source_id)
    direct_animation = _first_child(element, "anim", "animation", "clip")
    length = _parse_float(_attribute(element, "length", "duration"))
    if length is None:
        length = _parse_float(_attribute(direct_animation, "l", "length", "duration"))
    length = 2.0 if length is None else length
    transition_type = (_attribute(direct_animation, "t", "type", "transition") or "").casefold()
    destination = _attribute(direct_animation, "destination", "dest", "to", "go")
    if destination is None:
        destination = _attribute(element, "destination", "dest", "to")
    is_transition = transition_type in {"t", "transition", "true", "1", "yes"} or bool(destination)
    if destination:
        destination = resolve_relative_reference(destination, source_id)
    if is_transition and not speeds:
        event = _animation_event(_attribute(direct_animation, "id", "event", "name", "file", "path", "hkx"))
        if event:
            speeds = [SpeedVariant(event)]
    tags, furniture, no_random = _metadata(element)
    navigations = _parse_navigations(element, source_file, source_id)
    offset_element = _first_child(element, "offset")
    if offset_element is None:
        offset_element = element
    scene_offset = _transform_from_attributes(
        offset_element,
        diagnostics,
        source_file,
        source_id,
        "scene.offset",
    )
    unknown_elements = _unknown_elements(element)
    if unknown_elements:
        diagnostics.warning(
            "OSA_XML_ELEMENTS_PRESERVED",
            f"Preserved {len(unknown_elements)} XML element(s) without an explicit IR mapping.",
            category="unsupported or lost data",
            source_file=source_file,
            object_id=source_id,
        )
    provenance = Provenance(
        source_path=source_file,
        source_identifier=source_id,
        source_format=SourceFormat.OSA_OSEX,
        location=_source_line(element),
        extras={
            "xmlUnknownAttributes": _unknown_attributes(
                element,
                {
                    "id",
                    "scene_id",
                    "sceneId",
                    "name",
                    "title",
                    "actors",
                    "actorCount",
                    "actor_count",
                    "length",
                    "duration",
                    "destination",
                    "dest",
                    "to",
                    "furniture",
                    "style",
                },
            )
        },
        unknown_elements=unknown_elements,
    )
    node_type = TransitionNode if is_transition else SceneNode
    node = node_type(
        id=stable_internal_id(SourceFormat.OSA_OSEX, source_file, source_id),
        source_id=source_id,
        name=name,
        modpack=display_pack,
        length=length,
        speeds=speeds,
        actors=actors,
        navigations=navigations,
        default_speed=next(
            (
                index
                for index, speed_element in enumerate(
                    [
                        item
                        for container in _children(element, "speed", "speeds")
                        for item in _children(container, "sp", "speed", "variant")
                    ]
                )
                if _attribute(speed_element, "mtx") == "^idle"
            ),
            0,
        ),
        no_random_selection=no_random,
        furniture=FurnitureRequirement(furniture),
        offset=scene_offset,
        tags=tags,
        actions=_parse_actions(element, source_file, source_id),
        transition_destination=destination,
        transition_origin=(resolve_relative_reference(_attribute(element, "origin", "from") or "", source_id) or None),
        transition_priority=_parse_int(_attribute(element, "priority")),
        transition_description=_attribute(element, "description", "text"),
        transition_icon=_attribute(element, "icon"),
        transition_border=_attribute(element, "border"),
        transition_no_warnings=_parse_bool(_attribute(element, "noWarnings")),
        provenance=provenance,
    )
    if not node.speeds and not node.is_transition:
        diagnostics.error(
            "OSA_SCENE_NO_ANIMATION_EVENTS",
            "Scene metadata contains no recognized animation events.",
            category="XML syntax",
            source_file=source_file,
            object_id=source_id,
            can_continue=False,
        )
        return None
    return node


def _parse_scene_or_stages(
    element: ET.Element,
    source_file: str,
    fallback_id: str,
    display_pack: str,
    diagnostics: DiagnosticCollection,
) -> list[SceneNode]:
    stages = _children(element, *STAGE_TAGS)
    if not stages:
        node = _parse_single_scene(element, source_file, fallback_id, display_pack, diagnostics)
        return [node] if node else []
    parent_id = _attribute(element, "id", "scene_id", "sceneId") or fallback_id
    parent_name = _attribute(element, "name", "title") or parent_id
    nodes: list[SceneNode] = []
    for index, stage in enumerate(stages):
        stage_id = _attribute(stage, "id", "stage_id", "name") or f"stage{index + 1}"
        source_id = parent_id if index == 0 else f"{parent_id}|{stage_id}"
        stage_name = _attribute(stage, "name", "title") or f"{parent_name} {stage_id}"
        synthetic = ET.Element("scene", dict(element.attrib))
        synthetic.attrib["id"] = source_id
        synthetic.attrib["name"] = stage_name
        synthetic.attrib["actors"] = _attribute(element, "actors", "actorCount", "actor_count") or "2"
        stage_length = _attribute(stage, "length", "duration", "l")
        if stage_length is not None:
            synthetic.attrib["length"] = stage_length
        for child in list(element):
            if child is not stage and local_name(child) in {"info", "metadata", "actors", "actions", "nav", "tags"}:
                synthetic.append(child)
        for child in list(stage):
            synthetic.append(child)
        node = _parse_single_scene(
            synthetic,
            source_file,
            source_id,
            display_pack,
            diagnostics,
            source_id_override=source_id,
            name_override=stage_name,
        )
        if node:
            node.provenance.extras["parentSceneId"] = parent_id
            node.provenance.extras["stageId"] = stage_id
            node.provenance.extras["parentXmlUnknownAttributes"] = _unknown_attributes(
                element,
                {
                    "id",
                    "scene_id",
                    "sceneId",
                    "name",
                    "title",
                    "actors",
                    "actorCount",
                    "actor_count",
                    "length",
                    "duration",
                    "destination",
                    "dest",
                    "to",
                    "furniture",
                    "style",
                },
            )
            parent_unknown_elements = _unknown_elements(element)
            node.provenance.unknown_elements.extend(parent_unknown_elements)
            if parent_unknown_elements:
                diagnostics.warning(
                    "OSA_XML_ELEMENTS_PRESERVED",
                    f"Preserved {len(parent_unknown_elements)} parent XML element(s) without an explicit IR mapping.",
                    category="unsupported or lost data",
                    source_file=source_file,
                    object_id=source_id,
                )
            nodes.append(node)
    return nodes


def _safe_id(value: str, fallback: str) -> str:
    result = SAFE_IDENTIFIER.sub("_", value.strip()).strip("._-+")
    return (result or fallback)[:180]


def _xml_text(element: ET.Element) -> str:
    ET.indent(element, space="  ")
    return ET.tostring(element, encoding="unicode", xml_declaration=False, short_empty_elements=True) + "\n"


class OsaOsexAdapter:
    format = SourceFormat.OSA_OSEX

    def detect(self, path_or_tree: Path) -> DetectionResult:
        files = discover_osa_xml_files(path_or_tree)
        root = path_or_tree if path_or_tree.is_dir() else path_or_tree.parent
        inspected: list[str] = []
        scene_documents = 0
        path_signatures = 0
        malformed = 0
        for path in files[:64]:
            inspected.append(_relative(path, root))
            path_signatures += int(_is_osa_scene_path(path))
            try:
                document = ET.parse(path)
            except (OSError, ET.ParseError):
                malformed += 1
                continue
            document_root = document.getroot()
            if local_name(document_root) in SCENE_TAGS:
                scene_documents += 1
            elif local_name(document_root) in MODULE_TAGS and _descendants(document_root, *SCENE_TAGS):
                scene_documents += 1
        confidence = 0.0
        evidence: list[str] = []
        if scene_documents:
            confidence = 0.98 if path_signatures else 0.82
            evidence.append(f"{scene_documents} XML file(s) contain explicit OSA/OSex scene structures.")
        if path_signatures:
            evidence.append(f"{path_signatures} file(s) are under an OSA/OSex scene directory.")
        conflicts = [f"{malformed} candidate XML file(s) are malformed."] if malformed else []
        return DetectionResult(self.format, confidence, evidence, conflicts, inspected)

    def parse(self, source: Path, context: ParseContext) -> ConversionIR:
        root = source if source.is_dir() else source.parent
        diagnostics = context.diagnostics
        files = discover_osa_xml_files(source)
        pack_display = context.display_name or context.pack_id or root.name
        nodes: list[SceneNode] = []
        assets = discover_animation_assets(root)
        assets_by_path: dict[str, list] = {}
        for asset in assets:
            normalized = asset.source_path.replace("\\", "/").casefold()
            assets_by_path.setdefault(normalized, []).append(asset)
            assets_by_path.setdefault(PurePosixPath(normalized).name, []).append(asset)
        for path in files:
            source_file = _relative(path, root)
            try:
                document = ET.parse(path)
            except ET.ParseError as exc:
                diagnostics.error(
                    "OSA_XML_MALFORMED",
                    f"Malformed XML at line {exc.position[0]}, column {exc.position[1]}: {exc}",
                    category="XML syntax",
                    source_file=source_file,
                    can_continue=context.mode.value in {"best-effort", "salvage"},
                )
                continue
            except (OSError, UnicodeError) as exc:
                diagnostics.error(
                    "OSA_XML_READ_FAILED",
                    f"Could not decode XML: {exc}",
                    category="XML syntax",
                    source_file=source_file,
                    can_continue=False,
                )
                continue
            document_root = document.getroot()
            if local_name(document_root) in SCENE_TAGS:
                scene_elements = [document_root]
            elif local_name(document_root) in MODULE_TAGS:
                scene_elements = _descendants(document_root, *SCENE_TAGS)
            else:
                diagnostics.warning(
                    "OSA_XML_ROOT_UNSUPPORTED",
                    f"Skipped XML root <{local_name(document_root)}>; it is not a known scene or module structure.",
                    category="format detection",
                    source_file=source_file,
                )
                continue
            for scene_index, scene_element in enumerate(scene_elements):
                fallback_id = path.stem if len(scene_elements) == 1 else f"{path.stem}_{scene_index + 1}"
                parsed = _parse_scene_or_stages(scene_element, source_file, fallback_id, pack_display, diagnostics)
                for node in parsed:
                    _bind_explicit_animation_paths(scene_element, node, assets_by_path)
                nodes.extend(parsed)

        pack_id = context.pack_id or _safe_id(pack_display, "ConvertedPack")
        ir = ConversionIR(
            pack=PackMetadata(pack_id, pack_display),
            source=SourceDescriptor(
                SourceFormat.OSA_OSEX,
                root.name,
                detection_evidence=["Parsed only explicit scene/module/stage handlers."],
            ),
            graph=SceneGraph(nodes),
            assets=assets,
            diagnostics=diagnostics,
        )
        link_events_to_assets(ir, diagnostics)
        if any(
            transform.provenance.extras.get("unsupportedEulerRotation")
            for scene in nodes
            for transform in [scene.offset, *(actor.offset for actor in scene.actors)]
            if transform is not None
        ):
            ir.losses.append(
                ConversionLoss(
                    LossKind.UNSUPPORTED,
                    "OSA_EULER_ROTATION",
                    "OSA rx/ry rotations are preserved in provenance but cannot be emitted as OStim offsets.",
                )
            )
            ir.quality = ConversionQuality.LOSSY
        return ir

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection:
        return validate_ir(ir, include_assets=True, enforce_ostim_scene_ids=False)

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult:
        diagnostics = context.diagnostics
        destination.mkdir(parents=True, exist_ok=True)
        pack_id = _safe_id(context.pack_id or ir.pack.identifier, "ConvertedPack")
        display_name = context.display_name or ir.pack.display_name
        scene_root = destination / "Data" / "meshes" / "0SA" / "mod" / pack_id / "scene"
        scene_root.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        scene_map: dict[str, str] = {}
        for scene in sorted(ir.graph.nodes, key=lambda item: item.source_id.casefold()):
            # XML IDs may contain OSA's pipe and relative-link punctuation.
            # Keep the framework ID separate from the Windows-safe filename.
            base = re.sub(r"[\x00-\x1f\x7f]", "_", scene.source_id).strip() or "Scene"
            candidate = base
            if candidate.casefold() in used:
                suffix = hashlib.sha256(scene.id.encode()).hexdigest()[:8]
                candidate = f"{base[:220]}_{suffix}"
            used.add(candidate.casefold())
            scene_map[scene.source_id] = candidate
            scene.target_id = candidate

        event_map = {event.name: _safe_id(event.name, "Animation") for event in ir.events}
        written: list[Path] = []
        used_filenames: set[str] = set()
        if ir.sequences:
            message = "OSA/OSex export has no validated equivalent for OStim sequence JSON."
            diagnostics.warning(
                "OSA_EXPORT_SEQUENCES_UNSUPPORTED",
                message,
                category="unsupported or lost data",
            )
            if not any(loss.code == "OSA_EXPORT_SEQUENCES_UNSUPPORTED" for loss in ir.losses):
                ir.losses.append(ConversionLoss(LossKind.UNSUPPORTED, "OSA_EXPORT_SEQUENCES_UNSUPPORTED", message))
            ir.quality = ConversionQuality.LOSSY
        for scene in sorted(ir.graph.nodes, key=lambda item: scene_map[item.source_id].casefold()):
            target_id = scene_map[scene.source_id]
            preserve_osa_provenance = scene.provenance.source_format == SourceFormat.OSA_OSEX
            scene_element = ET.Element(
                "scene",
                {
                    "id": target_id,
                    "actors": str(len(scene.actors)),
                    "style": "OScene",
                },
            )
            if scene.offset:
                for key, value in scene.offset.to_ostim_dict().items():
                    scene_element.attrib[key] = format(value, ".15g")
            if context.preserve_unknown_fields and preserve_osa_provenance:
                for extras_key in ("xmlUnknownAttributes", "parentXmlUnknownAttributes"):
                    unknown_attributes = scene.provenance.extras.get(extras_key)
                    if isinstance(unknown_attributes, dict):
                        for key, value in unknown_attributes.items():
                            scene_element.attrib.setdefault(str(key), str(value))
            ET.SubElement(scene_element, "info", {"name": scene.name, "module": display_name})
            if scene.is_transition:
                attributes = {
                    "id": event_map.get(scene.speeds[0].animation, scene.speeds[0].animation)
                    if scene.speeds
                    else target_id,
                    "t": "T",
                    "l": format(scene.length, ".15g"),
                    "destination": scene_map.get(
                        scene.transition_destination or "", scene.transition_destination or ""
                    ),
                }
                if scene.transition_origin:
                    attributes["origin"] = scene_map.get(scene.transition_origin, scene.transition_origin)
                ET.SubElement(scene_element, "anim", attributes)
            else:
                first_event = (
                    event_map.get(scene.speeds[0].animation, scene.speeds[0].animation) if scene.speeds else target_id
                )
                first_animation_attributes = {
                    "id": first_event,
                    "t": "L",
                    "l": format(scene.length, ".15g"),
                }
                if scene.speeds:
                    if scene.speeds[0].playback_speed is not None:
                        first_animation_attributes["playbackSpeed"] = format(scene.speeds[0].playback_speed, ".15g")
                    if scene.speeds[0].display_speed is not None:
                        first_animation_attributes["displaySpeed"] = format(scene.speeds[0].display_speed, ".15g")
                ET.SubElement(
                    scene_element,
                    "anim",
                    first_animation_attributes,
                )
                if len(scene.speeds) > 1:
                    speed_element = ET.SubElement(scene_element, "speed")
                    for index, speed in enumerate(scene.speeds):
                        speed_attributes = {"mtx": "^idle" if index == scene.default_speed else "^thrustsPerSecond"}
                        if speed.display_speed is not None:
                            speed_attributes["qnt"] = format(speed.display_speed, ".15g")
                        variant = ET.SubElement(speed_element, "sp", speed_attributes)
                        animation_attributes = {
                            "id": event_map.get(speed.animation, speed.animation),
                            "t": "L",
                            "l": format(scene.length, ".15g"),
                        }
                        if speed.playback_speed is not None:
                            animation_attributes["playbackSpeed"] = format(speed.playback_speed, ".15g")
                        ET.SubElement(variant, "anim", animation_attributes)

            if scene.navigations:
                nav = ET.SubElement(scene_element, "nav")
                tab = ET.SubElement(nav, "tab", {"actor": "0"})
                page = ET.SubElement(tab, "page")
                for edge in scene.navigations:
                    attributes: dict[str, str] = {}
                    if edge.destination:
                        attributes["go"] = scene_map.get(edge.destination, edge.destination)
                    if edge.origin:
                        attributes["origin"] = scene_map.get(edge.origin, edge.origin)
                    if edge.description:
                        attributes["text"] = edge.description
                    if edge.icon:
                        attributes["icon"] = edge.icon
                    if edge.priority is not None:
                        attributes["priority"] = str(edge.priority)
                    if edge.border:
                        attributes["border"] = edge.border
                    if edge.no_warnings is not None:
                        attributes["noWarnings"] = str(edge.no_warnings).lower()
                    ET.SubElement(page, "option", attributes)

            metadata_attributes: dict[str, str] = {}
            if scene.tags:
                metadata_attributes["tags"] = ",".join(unique_preserving_order(scene.tags))
            if scene.furniture.type.casefold() != "none":
                metadata_attributes["furniture"] = scene.furniture.type
            if scene.no_random_selection is not None:
                metadata_attributes["noRandomSelection"] = str(scene.no_random_selection).lower()
            if metadata_attributes:
                ET.SubElement(scene_element, "metadata", metadata_attributes)

            actors_element = ET.SubElement(scene_element, "actors")
            for actor in scene.actors:
                attributes = {"index": str(actor.index)}
                if actor.type != "npc":
                    attributes["type"] = actor.type
                if actor.intended_sex:
                    attributes["intendedSex"] = actor.intended_sex
                if actor.sos_bend is not None:
                    attributes["sosBend"] = str(actor.sos_bend)
                if actor.scale is not None:
                    attributes["scale"] = format(actor.scale, ".15g")
                if actor.scale_height is not None:
                    attributes["scaleHeight"] = format(actor.scale_height, ".15g")
                if actor.animation_index is not None:
                    attributes["animationIndex"] = str(actor.animation_index)
                if actor.tags:
                    attributes["tags"] = ",".join(unique_preserving_order(actor.tags))
                if actor.feet_on_ground is not None:
                    attributes["feetOnGround"] = str(actor.feet_on_ground).lower()
                if actor.no_strip is not None:
                    attributes["noStrip"] = str(actor.no_strip).lower()
                if actor.requirements:
                    attributes["requirements"] = ",".join(actor.requirements)
                if actor.offset:
                    for key, value in actor.offset.to_ostim_dict().items():
                        attributes[key] = format(value, ".15g")
                if context.preserve_unknown_fields and actor.provenance.source_format == SourceFormat.OSA_OSEX:
                    unknown_attributes = actor.provenance.extras.get("xmlUnknownAttributes")
                    if isinstance(unknown_attributes, dict):
                        for key, value in unknown_attributes.items():
                            attributes.setdefault(str(key), str(value))
                actor_element = ET.SubElement(actors_element, "actor", attributes)
                for transition_type, target in sorted(actor.auto_transitions.items()):
                    ET.SubElement(
                        actor_element,
                        "autotransition",
                        {"type": transition_type, "destination": scene_map.get(target, target)},
                    )

            if scene.actions:
                actions_element = ET.SubElement(scene_element, "actions")
                for action in scene.actions:
                    attributes = {"type": action.type, "actor": str(action.actor)}
                    if action.target is not None:
                        attributes["target"] = str(action.target)
                    if action.performer is not None:
                        attributes["performer"] = str(action.performer)
                    if action.muted is not None:
                        attributes["muted"] = str(action.muted).lower()
                    if action.do_peaks is not None:
                        attributes["doPeaks"] = str(action.do_peaks).lower()
                    if action.peaks_annotated is not None:
                        attributes["peaksAnnotated"] = str(action.peaks_annotated).lower()
                    ET.SubElement(actions_element, "action", attributes)

            unsupported: list[str] = []
            if scene.fade_on_entry:
                unsupported.append("fadeOnEntry")
            if scene.scale_offset_with_furniture:
                unsupported.append("scaleOffsetWithFurniture")
            if scene.auto_transitions:
                unsupported.append("autoTransitions")
            if scene.annotations:
                unsupported.append("annotations")
            if not preserve_osa_provenance and scene.provenance.extras.get("jsonUnknown"):
                unsupported.append("unknown OStim JSON fields")
            for actor in scene.actors:
                if any(
                    value is not None
                    for value in (
                        actor.underlying_expression,
                        actor.expression_action,
                        actor.expression_override,
                        actor.look_up,
                        actor.look_down,
                        actor.look_left,
                        actor.look_right,
                    )
                ):
                    unsupported.append(f"actor[{actor.index}].expression")
            if unsupported:
                message = f"OSA/OSex export cannot represent: {', '.join(unsupported)}."
                diagnostics.warning(
                    "OSA_EXPORT_FIELDS_UNSUPPORTED",
                    message,
                    category="unsupported or lost data",
                    object_id=target_id,
                )
                ir.losses.append(
                    ConversionLoss(
                        LossKind.UNSUPPORTED, "OSA_EXPORT_FIELDS_UNSUPPORTED", message, scene.source_id, target_id
                    )
                )
                ir.quality = ConversionQuality.LOSSY

            if context.preserve_unknown_fields and preserve_osa_provenance:
                for raw_element in scene.provenance.unknown_elements:
                    try:
                        scene_element.append(ET.fromstring(raw_element))
                    except ET.ParseError:
                        diagnostics.warning(
                            "OSA_UNKNOWN_XML_REEMIT_FAILED",
                            "A preserved XML element could not be re-emitted and remains available in provenance.",
                            category="unsupported or lost data",
                            object_id=target_id,
                        )

            filename_base = _safe_id(target_id, "Scene")
            filename = filename_base
            if filename.casefold() in used_filenames:
                suffix = hashlib.sha256(scene.id.encode()).hexdigest()[:8]
                filename = f"{filename_base[:171]}_{suffix}"
            used_filenames.add(filename.casefold())
            output = scene_root / f"{filename}.xml"
            output.write_text(_xml_text(scene_element), encoding="utf-8", newline="\n")
            written.append(output)

        event_by_asset: dict[str, tuple[str, int]] = {}
        for event in ir.events:
            for actor_index, asset_id in event.actor_assets.items():
                event_by_asset[asset_id] = (event_map.get(event.name, event.name), actor_index)
        for asset in ir.assets:
            event = event_by_asset.get(asset.id)
            filename = f"{event[0]}_{event[1]}.hkx" if event else PurePosixPath(asset.source_path).name
            asset.target_path = (PurePosixPath("Data/meshes/0SA/mod") / pack_id / "animations" / filename).as_posix()
        return EmitResult(
            destination,
            written,
            diagnostics,
            scene_map,
            event_map,
            install_ready=False,
            output_pack_id=pack_id,
        )

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            format=self.format,
            import_supported=True,
            export_supported=True,
            round_trip_supported=True,
            supported_actor_counts="Explicit actor counts and indexed actor metadata.",
            navigation_support="Explicit option links, relative ^/+ references, and transition origin/destination.",
            furniture_support="Named furniture requirement when exposed by metadata.",
            annotation_support="Unknown XML and unsupported annotations retained in provenance.",
            expected_losses=(
                "OStim fadeOnEntry and expression controls have no validated OSA/OSex equivalent.",
                "OSA rx/ry Euler rotation cannot map to OStim's single rotational r field.",
                "Framework menus, scripts, quests, and plugins are outside animation-pack conversion.",
            ),
            required_behavior_generator="pandora or nemesis after OStim export",
        )
