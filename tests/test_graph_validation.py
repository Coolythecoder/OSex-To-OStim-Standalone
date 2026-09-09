from __future__ import annotations

from animation_converter.graph import validate_graph
from animation_converter.models import (
    ActorSlot,
    ConversionIR,
    FurnitureRequirement,
    NavigationEdge,
    PackMetadata,
    Provenance,
    SceneGraph,
    SceneNode,
    SourceDescriptor,
    SourceFormat,
    SpeedVariant,
)


def scene(scene_id: str, *, actors: int = 2, furniture: str = "none") -> SceneNode:
    return SceneNode(
        id=f"ir-{scene_id}",
        source_id=scene_id,
        name=scene_id,
        modpack="Pack",
        length=2,
        speeds=[SpeedVariant(f"{scene_id}Event")],
        actors=[ActorSlot(index) for index in range(actors)],
        furniture=FurnitureRequirement(furniture),
        provenance=Provenance(source_format=SourceFormat.OSTIM_SA),
    )


def ir_with(*nodes: SceneNode) -> ConversionIR:
    return ConversionIR(
        PackMetadata("Pack", "Pack"),
        SourceDescriptor(SourceFormat.OSTIM_SA, "fixture"),
        graph=SceneGraph(list(nodes)),
    )


def codes(ir: ConversionIR) -> set[str]:
    return {item.code for item in validate_graph(ir)}


def test_duplicate_scene_ids_are_case_insensitive() -> None:
    assert "GRAPH_DUPLICATE_SCENE_ID" in codes(ir_with(scene("Same"), scene("same")))


def test_dangling_destination_and_origin_are_reported() -> None:
    node = scene("Start")
    node.navigations.append(NavigationEdge(destination="Missing"))
    node.transition_origin = "Absent"
    result = codes(ir_with(node))
    assert "GRAPH_DANGLING_DESTINATION" in result
    assert "GRAPH_ORIGIN_WITHOUT_TRANSITION" in result


def test_malformed_and_self_navigation_are_reported() -> None:
    node = scene("Start")
    node.navigations.extend(
        [NavigationEdge(), NavigationEdge(destination="Start"), NavigationEdge(destination="Start", origin="Start")]
    )
    result = codes(ir_with(node))
    assert "GRAPH_EMPTY_NAVIGATION" in result
    assert "GRAPH_SELF_LINK" in result
    assert "GRAPH_AMBIGUOUS_NAVIGATION" in result


def test_actor_count_and_furniture_mismatches_are_blocking() -> None:
    start = scene("Start", actors=2, furniture="bed")
    target = scene("Target", actors=3, furniture="chair")
    start.navigations.append(NavigationEdge(destination="Target"))
    result = codes(ir_with(start, target))
    assert "GRAPH_ACTOR_COUNT_MISMATCH" in result
    assert "GRAPH_FURNITURE_MISMATCH" in result


def test_unreachable_component_and_no_entry_are_reported() -> None:
    first = scene("First")
    second = scene("Second")
    first.navigations.append(NavigationEdge(destination="First"))
    second.navigations.append(NavigationEdge(destination="Second"))
    result = codes(ir_with(first, second))
    assert "GRAPH_NO_ENTRY_POINT" in result


def test_resolvable_menu_graph_has_entry_and_no_dangling_links() -> None:
    start = scene("Start")
    target = scene("Target")
    start.navigations.extend(
        [NavigationEdge(origin="ExternalMenu", no_warnings=True), NavigationEdge(destination="Target")]
    )
    target.navigations.append(NavigationEdge(destination="Start"))
    result = codes(ir_with(start, target))
    assert "GRAPH_DANGLING_DESTINATION" not in result
    assert "GRAPH_NO_ENTRY_POINT" not in result
    assert "GRAPH_UNREACHABLE_SCENE" not in result


def test_disconnected_component_without_entry_is_reported() -> None:
    entry = scene("Entry")
    entry.navigations.append(NavigationEdge(origin="ExternalMenu", no_warnings=True))
    loop_a = scene("LoopA")
    loop_b = scene("LoopB")
    loop_a.navigations.append(NavigationEdge(destination="LoopB"))
    loop_b.navigations.append(NavigationEdge(destination="LoopA"))
    assert "GRAPH_COMPONENT_NO_ENTRY" in codes(ir_with(entry, loop_a, loop_b))
