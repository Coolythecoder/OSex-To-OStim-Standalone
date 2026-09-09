from __future__ import annotations

import json
from pathlib import Path

from animation_converter.adapters.base import EmitContext, ParseContext
from animation_converter.adapters.ostim_sa import OStimSAAdapter


def test_export_uses_canonical_loader_shape(fixture_root: Path, tmp_path: Path) -> None:
    adapter = OStimSAAdapter()
    ir = adapter.parse(fixture_root / "ostim_sa", ParseContext())
    result = adapter.emit(ir, tmp_path, EmitContext(pack_id="ExamplePack", display_name="Example Pack"))
    output = next(path for path in result.written_files if path.name == "ExampleIdle.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert list(data)[:3] == ["name", "modpack", "length"]
    assert data["modpack"] == "Example Pack"
    assert data["speeds"][0]["animation"] == "ExampleIdleEvent"
    assert set(data).isdisjoint({"id", "pack", "poses", "clips", "modPack"})
    assert data["actors"][0]["offset"] == {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0}
    assert data["fixtureExtension"] == {"preserve": True}
    assert data["speeds"][0]["fixtureSpeedExtension"] == 7
    assert data["navigations"][0]["fixtureNavExtension"] == "preserve"

    sequence = json.loads(
        next(path for path in result.written_files if path.name == "ExampleSequence.json").read_text(encoding="utf-8")
    )
    assert sequence["fixtureSequenceExtension"] is True
    assert sequence["scenes"][0]["fixtureEntryExtension"] == "preserve"


def test_export_is_deterministic(fixture_root: Path, tmp_path: Path) -> None:
    adapter = OStimSAAdapter()
    first_ir = adapter.parse(fixture_root / "ostim_sa", ParseContext())
    second_ir = adapter.parse(fixture_root / "ostim_sa", ParseContext())
    first = adapter.emit(first_ir, tmp_path / "one", EmitContext(pack_id="ExamplePack"))
    second = adapter.emit(second_ir, tmp_path / "two", EmitContext(pack_id="ExamplePack"))
    first_bytes = {path.relative_to(first.destination).as_posix(): path.read_bytes() for path in first.written_files}
    second_bytes = {path.relative_to(second.destination).as_posix(): path.read_bytes() for path in second.written_files}
    assert first_bytes == second_bytes


def test_default_speed_and_animation_indices_validate(fixture_root: Path) -> None:
    adapter = OStimSAAdapter()
    ir = adapter.parse(fixture_root / "ostim_sa", ParseContext())
    diagnostics = adapter.validate(ir, ParseContext())
    assert not any(item.code == "SCENE_DEFAULT_SPEED_RANGE" for item in diagnostics)
    assert not any(item.code == "ACTOR_ANIMATION_INDEX_INVALID" for item in diagnostics)
    assert not any(item.code == "EVENT_ACTOR_ASSET_MISSING" for item in diagnostics)


def test_reserved_ostim_prefix_is_not_emitted_for_third_party(tmp_path: Path, fixture_root: Path) -> None:
    adapter = OStimSAAdapter()
    ir = adapter.parse(fixture_root / "ostim_sa", ParseContext())
    ir.graph.nodes[0].source_id = "OStimReserved"
    result = adapter.emit(ir, tmp_path, EmitContext(pack_id="ThirdParty"))
    assert result.scene_id_map["OStimReserved"].startswith("ThirdParty_")
