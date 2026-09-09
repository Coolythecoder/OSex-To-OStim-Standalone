from __future__ import annotations

from pathlib import Path

from animation_converter.adapters.base import ParseContext
from animation_converter.adapters.osa_osex import OsaOsexAdapter, resolve_relative_reference
from animation_converter.models import ConversionMode


def test_namespaced_multistage_xml_preserves_metadata(fixture_root: Path) -> None:
    ir = OsaOsexAdapter().parse(fixture_root / "osa_namespaced", ParseContext())

    assert [scene.source_id for scene in ir.graph.nodes] == ["DemoScene", "DemoScene|Fast"]
    assert [scene.speeds[0].animation for scene in ir.graph.nodes] == ["DemoSlow", "DemoFast"]
    assert [scene.length for scene in ir.graph.nodes] == [3.5, 2.0]
    assert len(ir.graph.nodes[0].actors) == 2
    assert ir.graph.nodes[0].actors[0].offset.y == 1.5
    assert ir.graph.nodes[0].actors[0].offset.r == 5.0
    assert ir.graph.nodes[0].furniture.type == "bed"
    assert ir.graph.nodes[0].actions[0].type == "kissing"
    assert "romantic" in ir.graph.nodes[0].tags
    assert any("mystery" in value for value in ir.graph.nodes[0].provenance.unknown_elements)
    assert ir.graph.nodes[0].actors[0].provenance.extras["xmlUnknownAttributes"]["mystery"] == "kept"
    assert len(ir.events) == 2
    assert all(len(event.actor_assets) == 2 for event in ir.events)


def test_multiple_xml_files_and_relative_navigation_are_resolved(tmp_path: Path) -> None:
    scene_dir = tmp_path / "Data" / "meshes" / "0SA" / "mod" / "Demo" / "scene"
    scene_dir.mkdir(parents=True)
    (scene_dir / "base.xml").write_text(
        '<scene id="Demo|Pose" actors="1"><anim id="PoseEvent" t="L" l="2" />'
        '<nav><tab><page><option go="^+Next" text="Next" /></page></tab></nav></scene>',
        encoding="utf-8",
    )
    (scene_dir / "next.xml").write_text(
        '<scene id="Demo|Pose+Next" actors="1"><anim id="NextEvent" t="L" l="2" /></scene>',
        encoding="utf-8",
    )
    animation_dir = tmp_path / "Data" / "meshes" / "actors" / "character" / "animations" / "Demo"
    animation_dir.mkdir(parents=True)
    (animation_dir / "PoseEvent_0.hkx").write_bytes(b"pose")
    (animation_dir / "NextEvent_0.hkx").write_bytes(b"next")

    ir = OsaOsexAdapter().parse(tmp_path, ParseContext())
    assert len(ir.graph.nodes) == 2
    assert ir.graph.nodes[0].navigations[0].destination == "Demo|Pose+Next"
    assert resolve_relative_reference("+Next", "Demo|Pose+Old") == "Demo|Pose+Next"


def test_malformed_xml_reports_source_path(tmp_path: Path) -> None:
    scene_dir = tmp_path / "meshes" / "0SA" / "mod" / "Demo" / "scene"
    scene_dir.mkdir(parents=True)
    broken = scene_dir / "broken.xml"
    broken.write_text("<scene><anim></scene>", encoding="utf-8")

    ir = OsaOsexAdapter().parse(tmp_path, ParseContext(mode=ConversionMode.BEST_EFFORT))
    diagnostic = next(item for item in ir.diagnostics if item.code == "OSA_XML_MALFORMED")
    assert diagnostic.source_file.endswith("broken.xml")
    assert not ir.graph.nodes


def test_hkx_files_without_scene_metadata_do_not_become_scenes(tmp_path: Path) -> None:
    animation_dir = tmp_path / "meshes" / "actors" / "character" / "animations" / "Pack"
    animation_dir.mkdir(parents=True)
    (animation_dir / "OnlyAsset_0.hkx").write_bytes(b"asset")

    ir = OsaOsexAdapter().parse(tmp_path, ParseContext())
    assert ir.graph.nodes == []
    assert len(ir.assets) == 1


def test_undecodable_xml_has_clear_error(tmp_path: Path) -> None:
    scene_dir = tmp_path / "meshes" / "0SA" / "mod" / "Demo" / "scene"
    scene_dir.mkdir(parents=True)
    (scene_dir / "bad.xml").write_bytes(b"\xff\xfe\x00not xml")
    ir = OsaOsexAdapter().parse(tmp_path, ParseContext())
    assert any(item.code in {"OSA_XML_MALFORMED", "OSA_XML_READ_FAILED"} for item in ir.diagnostics)
