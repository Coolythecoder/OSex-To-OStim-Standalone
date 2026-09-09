"""SexLab Animation Loader (SLAL) metadata importer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..assets import discover_animation_assets
from ..diagnostics import DiagnosticCollection
from ..models import (
    ActorSlot,
    AnimationAsset,
    AnimationEvent,
    ConversionIR,
    ConversionLoss,
    ConversionQuality,
    FurnitureRequirement,
    LossKind,
    PackMetadata,
    Provenance,
    SceneAction,
    SceneGraph,
    SceneNode,
    SourceDescriptor,
    SourceFormat,
    SpeedVariant,
    stable_internal_id,
)
from ..validation import validate_ir
from .base import CapabilityDescriptor, DetectionResult, EmitContext, EmitResult, ParseContext


def _relative(path: Path, root: Path) -> str:
    try:
        return PurePosixPath(*path.relative_to(root).parts).as_posix()
    except ValueError:
        return path.name


def _is_slal_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return any(parts[index : index + 2] == ["slanims", "json"] for index in range(len(parts) - 1))


def _looks_like_slal(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("animations"), (list, dict)):
        return False
    animations = list(value["animations"].values()) if isinstance(value["animations"], dict) else value["animations"]
    return any(isinstance(item, dict) and isinstance(item.get("actors"), list) for item in animations[:5])


def discover_slal_json_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.casefold() == ".json" else []
    files = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.casefold() == ".json"),
        key=lambda item: str(item).casefold(),
    )
    marked = [path for path in files if _is_slal_path(path)]
    if marked:
        return marked
    result: list[Path] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _looks_like_slal(data):
            result.append(path)
    return result


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _safe_event(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return result or "SLALAnimation"


def _stage_event_aliases(event_name: str) -> list[str]:
    """Return conservative aliases used by real SLAL JSON/HKX pairs.

    Some packs retain an author token in JSON stage IDs while the HKX stem
    omits it. Actor/stage order can also appear as either ``A#_S#`` or
    ``S#_A#``. Only names with an explicit actor/stage suffix are normalized.
    """
    aliases: list[str] = []

    def add(value: str) -> None:
        if value and value.casefold() not in {item.casefold() for item in aliases}:
            aliases.append(value)

    add(event_name)
    match = re.match(
        r"^(?P<prefix>.+)_(?P<first>[AS])(?P<first_num>\d+)_(?P<second>[AS])(?P<second_num>\d+)$",
        event_name,
        re.IGNORECASE,
    )
    if not match or match.group("first").casefold() == match.group("second").casefold():
        return aliases

    first_is_actor = match.group("first").casefold() == "a"
    actor_number = match.group("first_num") if first_is_actor else match.group("second_num")
    stage_number = match.group("second_num") if first_is_actor else match.group("first_num")
    prefixes = [match.group("prefix")]
    if "_" in prefixes[0]:
        prefixes.append(prefixes[0].split("_", 1)[1])
    for prefix in prefixes:
        add(f"{prefix}_A{int(actor_number)}_S{int(stage_number)}")
        add(f"{prefix}_S{int(stage_number)}_A{int(actor_number)}")
    return aliases


def _common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    count = 0
    for left_part, right_part in zip(left, right, strict=False):
        if left_part.casefold() != right_part.casefold():
            break
        count += 1
    return count


def _stage_asset_candidates(
    event_name: str,
    assets_by_stem: dict[str, list[AnimationAsset]],
    source_file_parts: tuple[str, ...],
    actor_data: dict[str, Any],
) -> list[AnimationAsset]:
    candidates: list[AnimationAsset] = []
    for alias in _stage_event_aliases(event_name):
        candidates = assets_by_stem.get(PurePosixPath(alias.replace("\\", "/")).stem.casefold(), [])
        if candidates:
            break
    if len(candidates) <= 1:
        return candidates

    locality_scores = {
        asset.id: _common_prefix_length(
            source_file_parts,
            tuple(PurePosixPath(asset.source_path.replace("\\", "/")).parts),
        )
        for asset in candidates
    }
    best_score = max(locality_scores.values(), default=0)
    local = [asset for asset in candidates if locality_scores[asset.id] == best_score]
    candidates = local if best_score else candidates
    if len(candidates) <= 1:
        return candidates

    expected_creature = _actor_data_is_creature(actor_data)
    matching_actor_type = [
        asset
        for asset in candidates
        if (_asset_actor_root(asset) != "character") == expected_creature
    ]
    if matching_actor_type:
        candidates = matching_actor_type
    if len(candidates) <= 1:
        return candidates

    race = actor_data.get("race")
    if expected_creature and isinstance(race, str) and race.strip():
        race_token = _compact_token(race).removesuffix("s")
        matching_race = [
            asset
            for asset in candidates
            if race_token
            and (
                race_token in _compact_token(_asset_actor_root(asset))
                or _compact_token(_asset_actor_root(asset)).removesuffix("s") in race_token
            )
        ]
        candidates = matching_race or candidates
        if len(candidates) <= 1:
            return candidates

    depths = {
        asset.id: len(PurePosixPath(asset.source_path.replace("\\", "/")).parts)
        for asset in candidates
    }
    shortest_depth = min(depths.values())
    shortest = [asset for asset in candidates if depths[asset.id] == shortest_depth]
    return shortest if len(shortest) == 1 else candidates


def _asset_actor_root(asset: AnimationAsset) -> str:
    parts = PurePosixPath(asset.source_path.replace("\\", "/")).parts
    lowered = [part.casefold() for part in parts]
    try:
        actors_index = lowered.index("actors")
        animations_index = lowered.index("animations", actors_index + 1)
    except ValueError:
        return "character"
    root = parts[actors_index + 1 : animations_index]
    return PurePosixPath(*root).as_posix().casefold() if root else "character"


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _sex(actor_type: Any) -> str | None:
    if not isinstance(actor_type, str):
        return None
    folded = actor_type.casefold()
    if folded in {"f", "female"} or folded.endswith("female"):
        return "female"
    if folded in {"m", "male"} or folded.endswith("male"):
        return "male"
    return None


def _actor_data_is_creature(actor_data: dict[str, Any]) -> bool:
    actor_type = str(actor_data.get("type") or "").strip().casefold()
    if "creature" in actor_type:
        return True
    if actor_type and actor_type not in {"m", "male", "f", "female", "npc", "human"}:
        return True
    race = actor_data.get("race")
    return isinstance(race, str) and bool(race.strip()) and race.strip().casefold() not in {"human", "humans"}


class SlalAdapter:
    format = SourceFormat.SLAL

    def detect(self, path_or_tree: Path) -> DetectionResult:
        root = path_or_tree if path_or_tree.is_dir() else path_or_tree.parent
        files = discover_slal_json_files(path_or_tree)
        inspected: list[str] = []
        matches = 0
        marked = 0
        for path in files[:64]:
            inspected.append(_relative(path, root))
            marked += int(_is_slal_path(path))
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            matches += int(_looks_like_slal(data))
        evidence: list[str] = []
        if matches:
            evidence.append(f"{matches} JSON file(s) contain SLAL animations with actor stage arrays.")
        if marked:
            evidence.append(f"{marked} file(s) are under SLAnims/json.")
        return DetectionResult(
            self.format, 0.97 if marked and matches else (0.82 if matches else 0.0), evidence, [], inspected
        )

    def parse(self, source: Path, context: ParseContext) -> ConversionIR:
        root = source if source.is_dir() else source.parent
        diagnostics = context.diagnostics
        assets = discover_animation_assets(root)
        assets_by_stem: dict[str, list[AnimationAsset]] = {}
        for asset in assets:
            assets_by_stem.setdefault(PurePosixPath(asset.source_path).stem.casefold(), []).append(asset)
        nodes: list[SceneNode] = []
        events: list[AnimationEvent] = []
        metadata_files = discover_slal_json_files(source)
        pack_name = context.display_name or context.pack_id or root.name
        use_metadata_pack_name = not context.display_name and not context.pack_id and len(metadata_files) == 1
        for path in metadata_files:
            source_file = _relative(path, root)
            source_file_parts = tuple(PurePosixPath(source_file).parts)
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except UnicodeDecodeError as exc:
                diagnostics.error(
                    "SLAL_JSON_ENCODING",
                    f"SLAL JSON is not UTF-8: {exc}",
                    category="JSON syntax",
                    source_file=source_file,
                )
                continue
            except (OSError, json.JSONDecodeError) as exc:
                diagnostics.error(
                    "SLAL_JSON_MALFORMED",
                    f"Could not parse SLAL JSON: {exc}",
                    category="JSON syntax",
                    source_file=source_file,
                )
                continue
            if not _looks_like_slal(data):
                continue
            if use_metadata_pack_name and isinstance(data.get("name"), str):
                pack_name = data["name"]
            raw_animations = data.get("animations")
            animations = list(raw_animations.values()) if isinstance(raw_animations, dict) else raw_animations
            for animation_index, animation in enumerate(animations):
                if not isinstance(animation, dict):
                    diagnostics.warning(
                        "SLAL_ANIMATION_SHAPE",
                        f"Skipped non-object animation entry {animation_index}.",
                        category="schema shape",
                        source_file=source_file,
                    )
                    continue
                source_id = str(animation.get("id") or animation.get("name") or f"{path.stem}_{animation_index + 1}")
                raw_actors = animation.get("actors")
                if not isinstance(raw_actors, list) or not raw_actors:
                    diagnostics.error(
                        "SLAL_ACTORS_MISSING",
                        "SLAL animation has no actor stage metadata.",
                        category="actor index validity",
                        source_file=source_file,
                        object_id=source_id,
                    )
                    continue
                actors: list[ActorSlot] = []
                stage_count = 0
                for index, actor_data in enumerate(raw_actors):
                    if not isinstance(actor_data, dict):
                        actors.append(ActorSlot(index))
                        continue
                    stages = actor_data.get("stages")
                    stage_count = max(stage_count, len(stages) if isinstance(stages, list) else 0)
                    actor_type = str(actor_data.get("type") or "npc")
                    creature = _actor_data_is_creature(actor_data)
                    race = actor_data.get("race") if isinstance(actor_data.get("race"), str) else None
                    actor_tags = ["creature", actor_type] if creature else []
                    if creature and race:
                        actor_tags.append(f"creature:{race}")
                    actors.append(
                        ActorSlot(
                            index=index,
                            type="creature" if creature else "npc",
                            intended_sex=_sex(actor_type),
                            tags=actor_tags,
                            provenance=Provenance(
                                source_path=source_file,
                                source_identifier=f"{source_id}:actor:{index}",
                                source_format=SourceFormat.SLAL,
                                extras={
                                    "jsonUnknown": {
                                        key: value for key, value in actor_data.items() if key not in {"type", "stages"}
                                    }
                                },
                            ),
                        )
                    )
                speeds: list[SpeedVariant] = []
                for stage_index in range(stage_count):
                    event_name = f"{_safe_event(source_id)}_S{stage_index + 1}"
                    event = AnimationEvent(
                        hashlib.sha256(f"slal-event\0{source_file}\0{source_id}\0{stage_index}".encode()).hexdigest()[
                            :24
                        ],
                        event_name,
                        provenance=Provenance(
                            source_path=source_file,
                            source_identifier=f"{source_id}:stage:{stage_index + 1}",
                            source_format=SourceFormat.SLAL,
                        ),
                    )
                    complete = True
                    for actor_index, actor_data in enumerate(raw_actors):
                        stages = actor_data.get("stages") if isinstance(actor_data, dict) else None
                        if (
                            not isinstance(stages, list)
                            or stage_index >= len(stages)
                            or not isinstance(stages[stage_index], dict)
                        ):
                            complete = False
                            break
                        original_event = stages[stage_index].get("id")
                        if not isinstance(original_event, str) or not original_event:
                            complete = False
                            break
                        candidates = _stage_asset_candidates(
                            original_event,
                            assets_by_stem,
                            source_file_parts,
                            actor_data,
                        )
                        if len(candidates) == 1:
                            asset = candidates[0]
                            asset.event_name = event_name
                            asset.actor_index = actor_index
                            event.actor_assets[actor_index] = asset.id
                        elif len(candidates) > 1:
                            diagnostics.error(
                                "SLAL_STAGE_ASSET_AMBIGUOUS",
                                f"Stage event {original_event!r} matches multiple HKX files.",
                                category="animation event validity",
                                source_file=source_file,
                                object_id=source_id,
                            )
                    if complete:
                        speeds.append(SpeedVariant(event_name, display_speed=float(stage_index)))
                        events.append(event)
                    else:
                        diagnostics.error(
                            "SLAL_STAGE_INCOMPLETE",
                            f"Stage {stage_index + 1} does not define an event for every actor.",
                            category="animation event validity",
                            source_file=source_file,
                            object_id=source_id,
                        )
                if not speeds:
                    continue
                actions: list[SceneAction] = []
                raw_actions = animation.get("actions")
                if isinstance(raw_actions, list):
                    for action in raw_actions:
                        if (
                            isinstance(action, dict)
                            and isinstance(action.get("type"), str)
                            and isinstance(action.get("actor"), int)
                        ):
                            actions.append(
                                SceneAction(
                                    action["type"],
                                    action["actor"],
                                    action.get("target") if isinstance(action.get("target"), int) else None,
                                    action.get("performer") if isinstance(action.get("performer"), int) else None,
                                )
                            )
                furniture = next(
                    (
                        animation[key]
                        for key in ("furniture", "furnitureType", "furniture_type")
                        if isinstance(animation.get(key), str)
                    ),
                    "none",
                )
                nodes.append(
                    SceneNode(
                        id=stable_internal_id(SourceFormat.SLAL, source_file, source_id),
                        source_id=source_id,
                        name=str(animation.get("name") or source_id),
                        modpack=pack_name,
                        length=float(animation.get("duration", 6.0))
                        if isinstance(animation.get("duration", 6.0), (int, float))
                        else 6.0,
                        speeds=speeds,
                        actors=actors,
                        furniture=FurnitureRequirement(furniture),
                        tags=_tags(animation.get("tags")),
                        actions=actions,
                        provenance=Provenance(
                            source_path=source_file,
                            source_identifier=source_id,
                            source_format=SourceFormat.SLAL,
                            extras={
                                "jsonUnknown": {
                                    key: value
                                    for key, value in animation.items()
                                    if key
                                    not in {
                                        "id",
                                        "name",
                                        "actors",
                                        "tags",
                                        "duration",
                                        "actions",
                                        "furniture",
                                        "furnitureType",
                                        "furniture_type",
                                    }
                                }
                            },
                        ),
                    )
                )
        return ConversionIR(
            pack=PackMetadata(context.pack_id or _safe_event(pack_name), pack_name),
            source=SourceDescriptor(SourceFormat.SLAL, root.name),
            graph=SceneGraph(nodes),
            events=events,
            assets=assets,
            losses=[
                ConversionLoss(
                    LossKind.INFERRED,
                    "SLAL_STAGE_TO_SPEED",
                    "SLAL stage order is represented as OStim speed variants; SLAL registry behavior is not preserved.",
                )
            ],
            diagnostics=diagnostics,
            quality=ConversionQuality.LOSSY,
        )

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection:
        return validate_ir(ir, include_assets=True, enforce_ostim_scene_ids=False)

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult:
        raise NotImplementedError("SLAL reverse export is experimental and not implemented in 2.0.0.")

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            format=self.format,
            import_supported=True,
            export_supported=False,
            round_trip_supported=False,
            supported_actor_counts="Actor arrays with complete per-actor stage IDs.",
            navigation_support="SLAL does not provide an OStim scene graph; no links are invented.",
            furniture_support="Named furniture metadata when present.",
            annotation_support="Raw unknown stage fields retained in provenance; HKX binaries are not edited.",
            expected_losses=(
                "SLAL registry, sound, anim-object, and framework-specific semantics may not map to OStim.",
            ),
            required_behavior_generator="pandora or nemesis",
        )
