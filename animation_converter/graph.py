"""Scene-graph reference mapping and integrity checks."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .diagnostics import DiagnosticCollection
from .models import ConversionIR, NavigationEdge, SceneNode


def map_reference(reference: str | None, mapping: dict[str, str]) -> str | None:
    if reference is None:
        return None
    exact = mapping.get(reference)
    if exact:
        return exact
    folded = {key.casefold(): value for key, value in mapping.items()}
    return folded.get(reference.casefold(), reference)


def remap_scene_references(ir: ConversionIR, mapping: dict[str, str]) -> None:
    for scene in ir.graph.nodes:
        scene.target_id = mapping.get(scene.source_id, scene.target_id or scene.source_id)
        scene.transition_destination = map_reference(scene.transition_destination, mapping)
        scene.transition_origin = map_reference(scene.transition_origin, mapping)
        for edge in scene.navigations:
            edge.destination = map_reference(edge.destination, mapping)
            edge.origin = map_reference(edge.origin, mapping)
        scene.auto_transitions = {
            key: map_reference(value, mapping) or value for key, value in scene.auto_transitions.items()
        }
        for actor in scene.actors:
            actor.auto_transitions = {
                key: map_reference(value, mapping) or value for key, value in actor.auto_transitions.items()
            }
    for sequence in ir.sequences:
        for entry in sequence.entries:
            entry.scene_id = map_reference(entry.scene_id, mapping) or entry.scene_id


def _edge_targets(scene: SceneNode) -> Iterable[tuple[str, NavigationEdge | None]]:
    if scene.transition_destination:
        yield scene.transition_destination, None
    for edge in scene.navigations:
        if edge.destination:
            yield edge.destination, edge
    for target in scene.auto_transitions.values():
        yield target, None
    for actor in scene.actors:
        for target in actor.auto_transitions.values():
            yield target, None


def validate_graph(ir: ConversionIR) -> DiagnosticCollection:
    diagnostics = DiagnosticCollection()
    nodes = ir.graph.nodes
    by_folded: dict[str, SceneNode] = {}
    duplicate_ids: set[str] = set()
    for node in nodes:
        key = node.scene_id.casefold()
        if key in by_folded:
            duplicate_ids.add(node.scene_id)
            diagnostics.error(
                "GRAPH_DUPLICATE_SCENE_ID",
                f"Scene ID {node.scene_id!r} collides with {by_folded[key].scene_id!r}.",
                category="duplicate IDs",
                object_id=node.scene_id,
                remediation="Give every scene JSON file a globally unique filename.",
                can_continue=False,
            )
        else:
            by_folded[key] = node

    existing = set(by_folded)
    adjacency: dict[str, set[str]] = defaultdict(set)
    inbound: dict[str, int] = defaultdict(int)

    for node in nodes:
        source_key = node.scene_id.casefold()
        if node.is_transition:
            if not node.transition_destination:
                diagnostics.error(
                    "GRAPH_TRANSITION_DESTINATION_MISSING",
                    "Transition scene has no destination.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                    can_continue=False,
                )
            if node.navigations:
                diagnostics.warning(
                    "GRAPH_TRANSITION_NAVIGATIONS_IGNORED",
                    "OStim ignores navigations on a scene that has a destination.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                )
        elif node.transition_origin:
            diagnostics.error(
                "GRAPH_ORIGIN_WITHOUT_TRANSITION",
                "A scene defines transition origin without defining destination.",
                category="scene graph integrity",
                object_id=node.scene_id,
            )

        for edge_index, edge in enumerate(node.navigations):
            if not edge.destination and not edge.origin:
                diagnostics.error(
                    "GRAPH_EMPTY_NAVIGATION",
                    f"Navigation {edge_index} has neither origin nor destination.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                )
            if edge.destination and edge.origin:
                diagnostics.error(
                    "GRAPH_AMBIGUOUS_NAVIGATION",
                    f"Navigation {edge_index} defines both origin and destination.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                    remediation="Use destination for in-pack links, or origin for an external entry link.",
                )
            if edge.destination and edge.destination.casefold() == source_key:
                diagnostics.warning(
                    "GRAPH_SELF_LINK",
                    f"Navigation {edge_index} links {node.scene_id!r} to itself.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                )

        for target, edge in _edge_targets(node):
            target_key = target.casefold()
            if target_key not in existing:
                no_warnings = bool(edge and edge.no_warnings)
                severity_method = diagnostics.warning if no_warnings else diagnostics.error
                severity_method(
                    "GRAPH_DANGLING_DESTINATION",
                    f"Scene {node.scene_id!r} references missing destination {target!r}.",
                    category="scene graph integrity",
                    object_id=node.scene_id,
                    remediation="Add the target scene or remove the navigation.",
                )
                continue
            adjacency[source_key].add(target_key)
            inbound[target_key] += 1
            target_node = by_folded[target_key]
            if len(node.actors) != len(target_node.actors):
                diagnostics.error(
                    "GRAPH_ACTOR_COUNT_MISMATCH",
                    f"Navigation {node.scene_id!r} -> {target_node.scene_id!r} changes actor count "
                    f"from {len(node.actors)} to {len(target_node.actors)}.",
                    category="actor index validity",
                    object_id=node.scene_id,
                )
            source_furniture = node.furniture.type.casefold()
            target_furniture = target_node.furniture.type.casefold()
            if source_furniture != target_furniture and "none" not in {source_furniture, target_furniture}:
                diagnostics.error(
                    "GRAPH_FURNITURE_MISMATCH",
                    f"Navigation {node.scene_id!r} -> {target_node.scene_id!r} connects incompatible furniture "
                    f"types {node.furniture.type!r} and {target_node.furniture.type!r}.",
                    category="furniture validity",
                    object_id=node.scene_id,
                )

        if node.transition_origin:
            origin_key = node.transition_origin.casefold()
            if origin_key not in existing:
                if not node.transition_no_warnings:
                    diagnostics.error(
                        "GRAPH_DANGLING_ORIGIN",
                        f"Transition {node.scene_id!r} references missing origin {node.transition_origin!r}.",
                        category="scene graph integrity",
                        object_id=node.scene_id,
                    )
            else:
                adjacency[origin_key].add(source_key)
                inbound[source_key] += 1
        for edge in node.navigations:
            if edge.origin:
                origin_key = edge.origin.casefold()
                if origin_key not in existing:
                    if not edge.no_warnings:
                        diagnostics.warning(
                            "GRAPH_EXTERNAL_ORIGIN",
                            f"Navigation expects optional external origin {edge.origin!r}.",
                            category="scene graph integrity",
                            object_id=node.scene_id,
                        )
                else:
                    adjacency[origin_key].add(source_key)
                    inbound[source_key] += 1

    if not nodes or duplicate_ids:
        return diagnostics

    ordinary = [node for node in nodes if not node.is_transition]
    entry_keys = {
        node.scene_id.casefold()
        for node in ordinary
        if inbound[node.scene_id.casefold()] == 0
        or any(edge.origin and edge.origin.casefold() not in existing for edge in node.navigations)
    }
    if ordinary and not entry_keys:
        diagnostics.error(
            "GRAPH_NO_ENTRY_POINT",
            "The scene graph has no ordinary scene that can act as an entry point.",
            category="scene graph integrity",
            remediation="Add a valid origin navigation or retain a root scene with no incoming link.",
            can_continue=False,
        )
        return diagnostics

    undirected: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    remaining = set(existing)
    while remaining:
        seed = next(iter(remaining))
        component: set[str] = set()
        component_queue: deque[str] = deque([seed])
        while component_queue:
            current = component_queue.popleft()
            if current in component:
                continue
            component.add(current)
            component_queue.extend(undirected.get(current, set()) - component)
        remaining -= component
        if component.isdisjoint(entry_keys):
            labels = sorted(by_folded[item].scene_id for item in component)
            diagnostics.error(
                "GRAPH_COMPONENT_NO_ENTRY",
                f"Scene component has no valid entry point: {', '.join(labels[:5])}"
                + (" ..." if len(labels) > 5 else ""),
                category="scene graph integrity",
                remediation="Add a source-backed origin or entry navigation for this component.",
            )

    reachable: set[str] = set(entry_keys)
    queue: deque[str] = deque(entry_keys)
    while queue:
        source = queue.popleft()
        for target in adjacency.get(source, set()):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    for node in nodes:
        if node.scene_id.casefold() not in reachable:
            diagnostics.warning(
                "GRAPH_UNREACHABLE_SCENE",
                f"Scene {node.scene_id!r} is not reachable from any detected entry point.",
                category="scene graph integrity",
                object_id=node.scene_id,
                remediation="Add a source-backed navigation; do not invent links solely to suppress this warning.",
            )
    return diagnostics
