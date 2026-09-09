"""Flower Girls and comparable FNIS animation-list importer."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from ..assets import discover_animation_assets
from ..diagnostics import DiagnosticCollection
from ..models import (
    ActorSlot,
    AnimationEvent,
    ConversionIR,
    ConversionLoss,
    ConversionQuality,
    LossKind,
    PackMetadata,
    Provenance,
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


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "unsupported FNIS list encoding")


def parse_fnis_entries(text: str) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";", "//")):
            continue
        parts = stripped.split()
        hkx_index = next(
            (index for index, part in enumerate(parts) if part.strip('"').rstrip(";").casefold().endswith(".hkx")),
            None,
        )
        if hkx_index is None or hkx_index == 0:
            continue
        event = parts[hkx_index - 1].strip('"').rstrip(";")
        hkx = parts[hkx_index].strip('"').rstrip(";")
        if event and not event.startswith("-"):
            entries.append((event, hkx, line_number))
    return entries


def discover_flowergirls_lists(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.casefold() == ".txt" else []
    files = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt"),
        key=lambda item: str(item).casefold(),
    )
    named = [
        path
        for path in files
        if "flowergirl" in path.name.casefold() or ("fnis" in path.name.casefold() and "list" in path.name.casefold())
    ]
    return [path for path in named if parse_fnis_entries(_read_text(path))]


def _event_parts(event_name: str) -> tuple[str, str, int, int]:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_name).strip("._-") or "FlowerGirlsScene"
    patterns = [
        re.compile(r"^(?P<base>.+?)[_-](?:a|actor)(?P<actor>\d+)[_-](?:s|stage)(?P<stage>\d+)$", re.I),
        re.compile(r"^(?P<base>.+?)[_-](?:s|stage)(?P<stage>\d+)[_-](?:a|actor)(?P<actor>\d+)$", re.I),
    ]
    for pattern in patterns:
        match = pattern.match(normalized)
        if match:
            actor = max(0, int(match.group("actor")) - 1)
            stage = max(1, int(match.group("stage")))
            base = match.group("base")
            return base, f"{base}_S{stage}", actor, stage
    suffix = re.match(r"^(?P<base>.+)_(?P<actor>\d+)$", normalized)
    if suffix:
        return suffix.group("base"), suffix.group("base"), int(suffix.group("actor")), 1
    return normalized, normalized, 0, 1


class FlowerGirlsAdapter:
    format = SourceFormat.FLOWERGIRLS

    def detect(self, path_or_tree: Path) -> DetectionResult:
        root = path_or_tree if path_or_tree.is_dir() else path_or_tree.parent
        inspected: list[str] = []
        entries = 0
        named = 0
        for path in discover_flowergirls_lists(path_or_tree)[:32]:
            inspected.append(_relative(path, root))
            named += int("flowergirl" in path.name.casefold())
            entries += len(parse_fnis_entries(_read_text(path)))
        evidence: list[str] = []
        if entries:
            evidence.append(f"Found {entries} FNIS event-to-HKX rows.")
        if named:
            evidence.append(f"{named} list filename(s) identify Flower Girls.")
        confidence = 0.95 if named and entries else (0.65 if entries else 0.0)
        return DetectionResult(self.format, confidence, evidence, [], inspected)

    def parse(self, source: Path, context: ParseContext) -> ConversionIR:
        root = source if source.is_dir() else source.parent
        diagnostics = context.diagnostics
        assets = discover_animation_assets(root)
        assets_by_name: dict[str, list] = {}
        for asset in assets:
            assets_by_name.setdefault(PurePosixPath(asset.source_path).name.casefold(), []).append(asset)
        grouped: dict[str, dict[str, object]] = {}
        for path in discover_flowergirls_lists(source):
            source_file = _relative(path, root)
            try:
                entries = parse_fnis_entries(_read_text(path))
            except (OSError, UnicodeDecodeError) as exc:
                diagnostics.error(
                    "FLOWERGIRLS_LIST_ENCODING",
                    f"Could not decode FNIS list: {exc}",
                    category="format detection",
                    source_file=source_file,
                )
                continue
            for event_name, hkx_path, line_number in entries:
                scene_id, speed_name, actor_index, stage_index = _event_parts(event_name)
                group = grouped.setdefault(
                    scene_id,
                    {"speeds": {}, "actors": 0, "source": source_file, "events": {}},
                )
                group["speeds"][speed_name] = min(stage_index, group["speeds"].get(speed_name, stage_index))  # type: ignore[index,union-attr]
                group["actors"] = max(int(group["actors"]), actor_index + 1)
                event_key = (speed_name, actor_index)
                group["events"][event_key] = (event_name, hkx_path, line_number)  # type: ignore[index]

        nodes: list[SceneNode] = []
        events: list[AnimationEvent] = []
        for scene_id, group in sorted(grouped.items(), key=lambda item: item[0].casefold()):
            speed_order = sorted(group["speeds"].items(), key=lambda item: (item[1], item[0].casefold()))  # type: ignore[union-attr]
            speeds = [SpeedVariant(name, display_speed=float(index)) for index, (name, _) in enumerate(speed_order)]
            actors = [ActorSlot(index) for index in range(max(1, int(group["actors"])))]
            for speed_name, _ in speed_order:
                event = AnimationEvent(
                    hashlib.sha256(f"flower-event\0{scene_id}\0{speed_name}".encode()).hexdigest()[:24],
                    speed_name,
                    provenance=Provenance(
                        source_path=str(group["source"]),
                        source_identifier=speed_name,
                        source_format=SourceFormat.FLOWERGIRLS,
                    ),
                )
                for actor_index in range(len(actors)):
                    row = group["events"].get((speed_name, actor_index))  # type: ignore[union-attr]
                    if not row:
                        continue
                    original_event, hkx_path, line_number = row
                    candidates = assets_by_name.get(PurePosixPath(hkx_path.replace("\\", "/")).name.casefold(), [])
                    if len(candidates) == 1:
                        asset = candidates[0]
                        asset.event_name = speed_name
                        asset.actor_index = actor_index
                        event.actor_assets[actor_index] = asset.id
                    elif not candidates:
                        diagnostics.error(
                            "FLOWERGIRLS_HKX_MISSING",
                            f"FNIS event {original_event!r} references missing HKX {hkx_path!r}.",
                            category="asset existence",
                            source_file=str(group["source"]),
                            object_id=scene_id,
                            details={"line": line_number},
                        )
                events.append(event)
            nodes.append(
                SceneNode(
                    id=stable_internal_id(SourceFormat.FLOWERGIRLS, str(group["source"]), scene_id),
                    source_id=scene_id,
                    name=scene_id.replace("_", " "),
                    modpack=context.display_name or context.pack_id or root.name,
                    length=6.0,
                    speeds=speeds,
                    actors=actors,
                    tags=["flowergirls", "converted"],
                    provenance=Provenance(
                        source_path=str(group["source"]),
                        source_identifier=scene_id,
                        source_format=SourceFormat.FLOWERGIRLS,
                    ),
                )
            )
        display_name = context.display_name or context.pack_id or root.name
        return ConversionIR(
            pack=PackMetadata(context.pack_id or display_name, display_name),
            source=SourceDescriptor(SourceFormat.FLOWERGIRLS, root.name),
            graph=SceneGraph(nodes),
            events=events,
            assets=assets,
            losses=[
                ConversionLoss(
                    LossKind.UNSUPPORTED,
                    "FLOWERGIRLS_FRAMEWORK_FEATURES",
                    "Flower Girls quests, dialogue, scripts, spells, and menus are not animation metadata and were not converted.",
                )
            ],
            diagnostics=diagnostics,
            quality=ConversionQuality.LOSSY,
        )

    def validate(self, ir: ConversionIR, context: ParseContext | EmitContext) -> DiagnosticCollection:
        return validate_ir(ir, include_assets=True, enforce_ostim_scene_ids=False)

    def emit(self, ir: ConversionIR, destination: Path, context: EmitContext) -> EmitResult:
        raise NotImplementedError("Flower Girls reverse export is experimental and not implemented in 2.0.0.")

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            format=self.format,
            import_supported=True,
            export_supported=False,
            round_trip_supported=False,
            supported_actor_counts="Derived from explicit actor indices in FNIS event names.",
            navigation_support="FNIS lists contain no scene graph; no links are invented.",
            furniture_support="Not inferred from HKX or event names.",
            annotation_support="FNIS flags retained only as source evidence in this release.",
            expected_losses=("Quests, scripts, dialogue, menus, spells, and framework state are outside scope.",),
            required_behavior_generator="pandora or nemesis",
        )
