"""Animation-asset discovery, mapping, copying, and hash verification."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path, PurePosixPath

from .diagnostics import DiagnosticCollection
from .models import AnimationAsset, AnimationEvent, ConversionIR, Provenance

DOCUMENT_NAMES = re.compile(
    r"(?:^|[-_. ])(?:readme|license|licence|credits?|permissions?|copying)(?:[-_. ]|$)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    try:
        return PurePosixPath(*path.resolve().relative_to(root.resolve()).parts).as_posix()
    except ValueError:
        return path.name


def data_relative_path(relative_path: str) -> PurePosixPath:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    parts = list(path.parts)
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        return PurePosixPath(*parts[lowered.index("data") + 1 :])
    if "meshes" in lowered:
        return PurePosixPath(*parts[lowered.index("meshes") :])
    return path


def animation_actor_root(relative_path: str) -> PurePosixPath | None:
    relative = data_relative_path(relative_path)
    parts = relative.parts
    lowered = [part.casefold() for part in parts]
    try:
        actors_index = lowered.index("actors")
        animations_index = lowered.index("animations", actors_index + 1)
    except ValueError:
        return None
    root = parts[actors_index + 1 : animations_index]
    return PurePosixPath(*root) if root else None


def split_actor_suffix(stem: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<event>.+)_(?P<actor>[0-9]+)$", stem)
    if not match:
        return stem, None
    return match.group("event"), int(match.group("actor"))


def discover_animation_assets(root: Path) -> list[AnimationAsset]:
    paths = [root] if root.is_file() and root.suffix.casefold() == ".hkx" else root.rglob("*")
    assets: list[AnimationAsset] = []
    for path in sorted(
        (item for item in paths if item.is_file() and item.suffix.casefold() == ".hkx"),
        key=lambda item: str(item).casefold(),
    ):
        relative = relative_posix(path, root if root.is_dir() else root.parent)
        relative_parts = {part.casefold() for part in PurePosixPath(relative).parts}
        if "behaviors" in relative_parts:
            continue
        event_name, actor_index = split_actor_suffix(path.stem)
        asset_id = hashlib.sha256(relative.casefold().encode("utf-8")).hexdigest()[:24]
        assets.append(
            AnimationAsset(
                id=asset_id,
                source_path=relative,
                sha256=sha256_file(path),
                size=path.stat().st_size,
                event_name=event_name,
                actor_index=actor_index,
                provenance=Provenance(
                    source_path=relative,
                    source_identifier=path.stem,
                ),
            )
        )
    return assets


def discover_source_documents(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    allowed = {".txt", ".md", ".rst", ".rtf", ".pdf"}
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in allowed and DOCUMENT_NAMES.search(path.name)
        ),
        key=lambda item: str(item).casefold(),
    )


def link_events_to_assets(ir: ConversionIR, diagnostics: DiagnosticCollection | None = None) -> None:
    diagnostics = diagnostics or ir.diagnostics
    by_stem: dict[str, list[AnimationAsset]] = {}
    by_explicit: dict[tuple[str, int], list[AnimationAsset]] = {}
    for asset in ir.assets:
        stem = PurePosixPath(asset.source_path).stem
        by_stem.setdefault(stem.casefold(), []).append(asset)
        if asset.event_name is not None and asset.actor_index is not None:
            by_explicit.setdefault((asset.event_name.casefold(), asset.actor_index), []).append(asset)

    events: dict[str, AnimationEvent] = {event.name.casefold(): event for event in ir.events}
    for scene in ir.graph.nodes:
        for speed in scene.speeds:
            key = speed.animation.casefold()
            event = events.get(key)
            if event is None:
                event = AnimationEvent(
                    id=hashlib.sha256(f"event\0{key}".encode()).hexdigest()[:24],
                    name=speed.animation,
                    provenance=Provenance(
                        source_identifier=speed.animation,
                        source_format=ir.source.format,
                    ),
                )
                events[key] = event
            for actor in scene.actors:
                actor_index = actor.effective_animation_index
                if actor_index in event.actor_assets:
                    continue
                candidates = by_explicit.get((key, actor_index), [])
                if not candidates:
                    candidates = by_stem.get(f"{speed.animation}_{actor_index}".casefold(), [])
                if len(candidates) == 1:
                    asset = candidates[0]
                    event.actor_assets[actor_index] = asset.id
                    asset.event_name = speed.animation
                    asset.actor_index = actor_index
                elif len(candidates) > 1:
                    diagnostics.error(
                        "ASSET_EVENT_AMBIGUOUS",
                        f"Animation event {speed.animation!r} actor {actor_index} matches multiple HKX files.",
                        category="animation event validity",
                        object_id=scene.scene_id,
                        remediation="Rename colliding HKX files or provide an explicit event-to-asset mapping.",
                    )
    ir.events = sorted(events.values(), key=lambda item: item.name.casefold())


def assign_ostim_target_paths(ir: ConversionIR, pack_id: str, event_map: dict[str, str]) -> None:
    event_by_asset: dict[str, tuple[str, int]] = {}
    for event in ir.events:
        target_event = event_map.get(event.name, event.name)
        for actor_index, asset_id in event.actor_assets.items():
            event_by_asset[asset_id] = (target_event, actor_index)

    for asset in ir.assets:
        mapping = event_by_asset.get(asset.id)
        if mapping:
            event_name, actor_index = mapping
            filename = f"{event_name}_{actor_index}.hkx"
            actor_root = animation_actor_root(asset.source_path) or PurePosixPath("character")
            asset.target_path = (
                PurePosixPath("Data/meshes/actors") / actor_root / "animations" / pack_id / filename
            ).as_posix()
            continue
        relative = data_relative_path(asset.source_path)
        if relative.parts and relative.parts[0].casefold() == "meshes":
            asset.target_path = (PurePosixPath("Data") / relative).as_posix()
        else:
            asset.target_path = (
                PurePosixPath("Data/meshes/actors/character/animations")
                / pack_id
                / PurePosixPath(asset.source_path).name
            ).as_posix()


def copy_assets_verified(
    ir: ConversionIR,
    source_root: Path,
    destination_root: Path,
    diagnostics: DiagnosticCollection,
) -> list[Path]:
    written: list[Path] = []
    collisions: dict[str, str] = {}
    for asset in sorted(ir.assets, key=lambda item: (item.target_path or "").casefold()):
        if not asset.target_path:
            diagnostics.error(
                "ASSET_TARGET_MISSING",
                f"No destination path was assigned to {asset.source_path}.",
                category="asset existence",
                source_file=asset.source_path,
                can_continue=False,
            )
            continue
        target_rel = PurePosixPath(asset.target_path.replace("\\", "/"))
        if target_rel.is_absolute() or ".." in target_rel.parts:
            diagnostics.fatal(
                "ASSET_TARGET_UNSAFE",
                f"Unsafe asset destination path: {asset.target_path}",
                category="install directory layout",
                source_file=asset.source_path,
            )
            continue
        collision_key = target_rel.as_posix().casefold()
        previous = collisions.get(collision_key)
        if previous and previous != asset.id:
            diagnostics.error(
                "ASSET_TARGET_COLLISION",
                f"Multiple assets map to {target_rel.as_posix()}.",
                category="asset existence",
                remediation="Use unique animation events or filenames.",
                can_continue=False,
            )
            continue
        collisions[collision_key] = asset.id

        source_path = source_root.joinpath(*PurePosixPath(asset.source_path).parts)
        if not source_path.is_file():
            diagnostics.error(
                "ASSET_SOURCE_MISSING",
                f"Referenced HKX asset does not exist: {asset.source_path}",
                category="asset existence",
                source_file=asset.source_path,
                can_continue=False,
            )
            continue
        destination = destination_root.joinpath(*target_rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            diagnostics.error(
                "ASSET_OVERWRITE",
                f"Asset copy would overwrite an existing file: {target_rel.as_posix()}",
                category="asset existence",
                can_continue=False,
            )
            continue
        shutil.copyfile(source_path, destination)
        source_hash = sha256_file(source_path)
        destination_hash = sha256_file(destination)
        if source_hash != destination_hash:
            destination.unlink(missing_ok=True)
            diagnostics.fatal(
                "ASSET_HASH_MISMATCH",
                f"HKX hash changed while copying {asset.source_path}.",
                category="asset existence",
                source_file=asset.source_path,
            )
            continue
        asset.sha256 = source_hash
        asset.size = source_path.stat().st_size
        written.append(destination)
    return written


def copy_source_documents(root: Path, destination: Path, diagnostics: DiagnosticCollection) -> list[Path]:
    documents = discover_source_documents(root)
    if not documents:
        diagnostics.warning(
            "REDISTRIBUTION_PERMISSION_UNKNOWN",
            "No README, license, credit, or permission document was found in the source package.",
            category="licensing and redistribution notices",
            remediation="Treat the conversion as local-use only unless the original author grants redistribution permission.",
        )
        return []
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used: set[str] = set()
    for document in documents:
        name = document.name
        candidate = name
        index = 2
        while candidate.casefold() in used:
            candidate = f"{document.stem}_{index}{document.suffix}"
            index += 1
        used.add(candidate.casefold())
        target = destination / candidate
        shutil.copyfile(document, target)
        written.append(target)
    return written


def source_file_hashes(root: Path, formats: set[str] | None = None) -> dict[str, str]:
    if root.is_file():
        return {root.name: sha256_file(root)}
    result: dict[str, str] = {}
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and (formats is None or item.suffix.casefold() in formats)),
        key=lambda item: str(item).casefold(),
    ):
        result[relative_posix(path, root)] = sha256_file(path)
    return result
