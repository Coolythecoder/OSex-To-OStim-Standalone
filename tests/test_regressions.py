from __future__ import annotations

import json
from pathlib import Path

from animation_converter.adapters.base import ParseContext
from animation_converter.adapters.ostim_sa import OStimSAAdapter
from animation_converter.models import (
    ActorSlot,
    ConversionIR,
    ConversionMode,
    PackMetadata,
    Provenance,
    SceneGraph,
    SceneNode,
    SourceDescriptor,
    SourceFormat,
    SpeedVariant,
    Transform,
)
from animation_converter.packaging import PackageMode
from animation_converter.service import ConversionRequest, ConverterService
from animation_converter.validation import validate_ir, validate_offsets


def minimal_ir(transform: Transform | None = None, scene_count: int = 2) -> ConversionIR:
    nodes = [
        SceneNode(
            id=f"ir-{index}",
            source_id=f"Scene{index}",
            name=f"Scene {index}",
            modpack="Pack",
            length=2,
            speeds=[SpeedVariant(f"Event{index}")],
            actors=[ActorSlot(0, offset=transform if index == 0 else None)],
            provenance=Provenance(source_format=SourceFormat.OSA_OSEX),
        )
        for index in range(scene_count)
    ]
    return ConversionIR(
        PackMetadata("Pack", "Pack"),
        SourceDescriptor(SourceFormat.OSA_OSEX, "fixture"),
        graph=SceneGraph(nodes),
    )


def test_conversion_cannot_succeed_with_zero_scenes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    request = ConversionRequest(
        source,
        SourceFormat.OSTIM_SA,
        tmp_path / "output",
        SourceFormat.OSA_OSEX,
        ConversionMode.NORMAL,
        PackageMode.DIRECTORY,
    )
    result = ConverterService().convert(request)
    assert not result.output_written
    assert any(item.code == "CONVERSION_ZERO_SCENES" for item in result.diagnostics)
    assert (tmp_path / "output-conversion-report.json").exists()
    assert (tmp_path / "output-conversion-report.txt").exists()
    assert (tmp_path / "output-conversion-manifest.json").exists()


def test_normal_mode_never_falls_back_to_one_scene_per_hkx(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Loose_0.hkx").write_bytes(b"hkx")
    result = ConverterService().convert(
        ConversionRequest(
            source,
            SourceFormat.OSTIM_SA,
            tmp_path / "normal",
            SourceFormat.OSA_OSEX,
            ConversionMode.NORMAL,
            PackageMode.DIRECTORY,
        )
    )
    assert not result.output_written
    assert result.ir.graph.nodes == []


def test_one_hkx_per_scene_is_only_explicit_salvage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Loose_0.hkx").write_bytes(b"hkx")
    result = ConverterService().convert(
        ConversionRequest(
            source,
            SourceFormat.OSTIM_SA,
            tmp_path / "salvage",
            SourceFormat.OSA_OSEX,
            ConversionMode.SALVAGE,
            PackageMode.DIRECTORY,
            pack_id="SalvagePack",
        )
    )
    assert result.output_written
    assert len(result.ir.graph.nodes) == 1
    assert result.ir.graph.nodes[0].salvaged
    assert not result.readiness.install_ready
    assert "notSalvage" in result.readiness.blockers


def test_all_zero_offsets_caused_by_parse_failure_are_blocking() -> None:
    transform = Transform(provenance=Provenance(extras={"parseFailed": True, "inferredFromMissingSource": True}))
    diagnostics = validate_offsets(minimal_ir(transform))
    assert any(item.code == "OFFSET_ALL_ZERO_PARSE_FAILURE" for item in diagnostics)


def test_explicit_neutral_zero_offset_is_valid() -> None:
    transform = Transform(provided_fields={"x", "y", "z", "r"})
    diagnostics = validate_offsets(minimal_ir(transform, scene_count=1))
    assert not diagnostics.has_errors
    assert any(item.code == "OFFSET_NEUTRAL_SOURCE_VALUE" for item in diagnostics)


def test_old_id_pack_poses_clips_shape_is_rejected_as_current_ostim(tmp_path: Path) -> None:
    scene_root = tmp_path / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / "Pack"
    scene_root.mkdir(parents=True)
    old = {
        "id": "OldScene",
        "pack": "OldPack",
        "poses": [{"clips": [{"file": "meshes/actors/character/animations/Old_0.hkx"}]}],
    }
    (scene_root / "OldScene.json").write_text(json.dumps(old), encoding="utf-8")
    ir = OStimSAAdapter().parse(tmp_path, ParseContext())
    assert not ir.graph.nodes
    assert any(item.code == "OSTIM_LEGACY_INTERMEDIATE_SHAPE" for item in ir.diagnostics)


def test_issue_one_menu_graph_regression_is_resolvable(fixture_root: Path) -> None:
    ir = OStimSAAdapter().parse(fixture_root / "ostim_sa", ParseContext())
    diagnostics = validate_ir(ir)
    graph_codes = {item.code for item in diagnostics}
    assert "GRAPH_DANGLING_DESTINATION" not in graph_codes
    assert "GRAPH_NO_ENTRY_POINT" not in graph_codes
    assert not diagnostics.has_errors
