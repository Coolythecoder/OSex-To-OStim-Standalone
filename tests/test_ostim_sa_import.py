from __future__ import annotations

import json
from pathlib import Path

from animation_converter.adapters.base import ParseContext
from animation_converter.adapters.ostim_sa import OStimSAAdapter


def test_scene_id_comes_from_filename_and_unknown_fields_survive(fixture_root: Path) -> None:
    ir = OStimSAAdapter().parse(fixture_root / "ostim_sa", ParseContext())
    scene = next(scene for scene in ir.graph.nodes if scene.source_id == "ExampleIdle")
    assert scene.source_id == "ExampleIdle"
    assert scene.provenance.extras["jsonUnknown"]["fixtureExtension"] == {"preserve": True}
    assert scene.speeds[0].animation == "ExampleIdleEvent"
    assert scene.actors[0].offset.is_neutral
    assert scene.actors[0].offset.was_provided
    assert len(ir.sequences) == 1
    assert ir.sequences[0].entries[1].scene_id == "ExampleNext"


def test_duplicate_filenames_across_subfolders_are_rejected(tmp_path: Path) -> None:
    scene_root = tmp_path / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes"
    payload = {"name": "Duplicate", "modpack": "Pack", "length": 2, "speeds": [{"animation": "E"}], "actors": [{}]}
    for folder in ("one", "two"):
        target = scene_root / folder
        target.mkdir(parents=True)
        (target / "Same.json").write_text(json.dumps(payload), encoding="utf-8")

    ir = OStimSAAdapter().parse(tmp_path, ParseContext())
    assert any(item.code == "OSTIM_DUPLICATE_FILENAME" for item in ir.diagnostics)


def test_modpack_alias_is_normalized_with_warning(tmp_path: Path) -> None:
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    payload = {
        "name": "Alias",
        "modPack": "Historical",
        "length": 2,
        "speeds": [{"animation": "AliasEvent"}],
        "actors": [{}],
    }
    (scene_root / "Alias.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "AliasEvent_0.hkx").write_bytes(b"hkx")

    ir = OStimSAAdapter().parse(scene_root, ParseContext())
    assert ir.graph.nodes[0].modpack == "Historical"
    assert any(item.code == "OSTIM_MODPACK_ALIAS" for item in ir.diagnostics)


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    payload = (
        b'\xef\xbb\xbf{"name":"BOM","modpack":"Pack","length":2,"speeds":[{"animation":"BomEvent"}],"actors":[{}]}'
    )
    (scene_root / "BomScene.json").write_bytes(payload)
    ir = OStimSAAdapter().parse(scene_root, ParseContext())
    assert ir.graph.nodes[0].source_id == "BomScene"


def test_malformed_and_non_utf8_json_report_errors(tmp_path: Path) -> None:
    scene_root = tmp_path / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / "Pack"
    scene_root.mkdir(parents=True)
    (scene_root / "Malformed.json").write_text("{", encoding="utf-8")
    (scene_root / "Encoding.json").write_bytes(b"\xff\xfeinvalid")
    ir = OStimSAAdapter().parse(tmp_path, ParseContext())
    codes = {item.code for item in ir.diagnostics}
    assert "JSON_MALFORMED" in codes
    assert "JSON_ENCODING_UNSUPPORTED" in codes


def test_transition_fields_are_parsed(tmp_path: Path) -> None:
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    payload = {
        "name": "Transition",
        "modpack": "Pack",
        "length": 1.5,
        "destination": "End",
        "origin": "Start",
        "priority": 12,
        "description": "Move",
        "icon": "icon",
        "border": "abcdef",
        "noWarnings": True,
        "speeds": [{"animation": "TransitionEvent"}],
        "actors": [{}, {}],
    }
    (scene_root / "Transition.json").write_text(json.dumps(payload), encoding="utf-8")
    ir = OStimSAAdapter().parse(scene_root, ParseContext())
    scene = ir.graph.nodes[0]
    assert scene.is_transition
    assert scene.transition_destination == "End"
    assert scene.transition_origin == "Start"
    assert scene.transition_priority == 12
