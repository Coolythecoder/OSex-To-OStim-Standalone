from __future__ import annotations

import json
import shutil
from pathlib import Path

from animation_converter.adapters.base import ParseContext
from animation_converter.adapters.osa_osex import OsaOsexAdapter
from animation_converter.adapters.ostim_sa import OStimSAAdapter
from animation_converter.assets import sha256_file
from animation_converter.models import ConversionMode, SourceFormat
from animation_converter.packaging import PackageMode
from animation_converter.service import ConversionRequest, ConverterService


def scene_semantics(ir) -> list[dict]:
    return [
        {
            "name": scene.name,
            "length": scene.length,
            "speeds": [speed.animation for speed in scene.speeds],
            "actors": len(scene.actors),
            "tags": sorted(scene.tags),
            "actions": [(action.type, action.actor, action.target) for action in scene.actions],
        }
        for scene in sorted(ir.graph.nodes, key=lambda item: item.name.casefold())
    ]


def test_ostim_same_format_semantic_roundtrip_and_hashes(fixture_root: Path, tmp_path: Path) -> None:
    service = ConverterService()
    source = fixture_root / "ostim_sa"
    output = tmp_path / "roundtrip"
    result = service.convert(
        ConversionRequest(
            source,
            SourceFormat.OSTIM_SA,
            output,
            SourceFormat.OSTIM_SA,
            ConversionMode.NORMAL,
            PackageMode.DIRECTORY,
            "none",
            "ExamplePack",
            "Example Pack",
        )
    )
    assert result.output_written
    original = OStimSAAdapter().parse(source, ParseContext())
    reparsed = OStimSAAdapter().parse(output, ParseContext())
    assert scene_semantics(original) == scene_semantics(reparsed)
    assert {scene.source_id for scene in reparsed.graph.nodes} == {"ExampleIdle", "ExampleNext"}
    for asset in result.ir.assets:
        source_path = source.joinpath(*PurePath(asset.provenance.source_path or asset.source_path).parts)
        target_path = output.joinpath(*PurePath(asset.target_path).parts)
        assert sha256_file(source_path) == sha256_file(target_path)


def PurePath(value: str):
    from pathlib import PurePosixPath

    return PurePosixPath(value.replace("\\", "/"))


def test_osa_same_format_roundtrip_preserves_representable_subset(fixture_root: Path, tmp_path: Path) -> None:
    service = ConverterService()
    source = fixture_root / "osa_namespaced"
    output = tmp_path / "osa"
    result = service.convert(
        ConversionRequest(
            source,
            SourceFormat.OSA_OSEX,
            output,
            SourceFormat.OSA_OSEX,
            ConversionMode.BEST_EFFORT,
            PackageMode.DIRECTORY,
            "none",
            "Demo",
        )
    )
    assert result.output_written
    original = OsaOsexAdapter().parse(source, ParseContext())
    reparsed = OsaOsexAdapter().parse(output, ParseContext())
    assert [scene.name for scene in original.graph.nodes] == [scene.name for scene in reparsed.graph.nodes]
    assert [len(scene.actors) for scene in original.graph.nodes] == [
        len(scene.actors) for scene in reparsed.graph.nodes
    ]
    assert [scene.actions[0].type for scene in reparsed.graph.nodes] == ["kissing", "kissing"]
    emitted_xml = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.xml"))
    assert "mystery" in emitted_xml
    assert 'mystery="kept"' in emitted_xml


def test_osa_to_ostim_to_osa_uses_manifest_to_restore_ids(fixture_root: Path, tmp_path: Path) -> None:
    service = ConverterService()
    first_output = tmp_path / "ostim"
    first = service.convert(
        ConversionRequest(
            fixture_root / "osa_namespaced",
            SourceFormat.OSTIM_SA,
            first_output,
            SourceFormat.OSA_OSEX,
            ConversionMode.BEST_EFFORT,
            PackageMode.DIRECTORY,
            "none",
            "DemoBridge",
        )
    )
    assert first.output_written
    assert any(item.code == "SOURCE_XML_METADATA_UNMAPPED" for item in first.diagnostics)
    assert any(loss.code == "SOURCE_XML_METADATA_UNMAPPED" for loss in first.ir.losses)
    manifest = json.loads((first_output / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["sourceFramework"] == "osa-osex"
    assert manifest["sceneIdMappings"]["DemoScene"]["targetId"].startswith("DemoBridge_")

    second_output = tmp_path / "osa"
    second = service.convert(
        ConversionRequest(
            first_output,
            SourceFormat.OSA_OSEX,
            second_output,
            SourceFormat.OSTIM_SA,
            ConversionMode.BEST_EFFORT,
            PackageMode.DIRECTORY,
            "none",
            "DemoBack",
            roundtrip_sidecar=True,
        )
    )
    assert second.output_written
    assert {scene.source_id for scene in second.ir.graph.nodes} == {"DemoScene", "DemoScene|Fast"}
    assert any(item.code == "MANIFEST_SOURCE_IDS_RESTORED" for item in second.diagnostics)
    restored_xml = "\n".join(path.read_text(encoding="utf-8") for path in second_output.rglob("*.xml"))
    assert "mystery" in restored_xml
    assert 'mystery="kept"' in restored_xml


def test_ostim_to_osa_reports_unrepresentable_fields(fixture_root: Path, tmp_path: Path) -> None:
    ir = OStimSAAdapter().parse(fixture_root / "ostim_sa", ParseContext())
    ir.graph.nodes[0].fade_on_entry = True
    result = OsaOsexAdapter().emit(
        ir,
        tmp_path,
        context=__import__("animation_converter.adapters.base", fromlist=["EmitContext"]).EmitContext(
            pack_id="Reverse"
        ),
    )
    assert result.written_files
    assert any(item.code == "OSA_EXPORT_FIELDS_UNSUPPORTED" for item in result.diagnostics)
    assert any(loss.code == "OSA_EXPORT_FIELDS_UNSUPPORTED" for loss in ir.losses)


def test_supported_ostim_to_osa_to_ostim_subset_is_semantically_equal(fixture_root: Path, tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(fixture_root / "ostim_sa", source)
    shutil.rmtree(source / "Data" / "SKSE" / "Plugins" / "OStim" / "sequences")
    request = ConversionRequest(
        source,
        SourceFormat.OSA_OSEX,
        tmp_path / "back-to-ostim",
        SourceFormat.OSTIM_SA,
        ConversionMode.BEST_EFFORT,
        PackageMode.DIRECTORY,
        "none",
        "RoundTripPack",
    )
    result = ConverterService().roundtrip(request, through=SourceFormat.OSA_OSEX)
    assert result.second is not None and result.second.output_written
    assert result.semantically_equal
    assert result.source_digest == result.roundtrip_digest


def test_osa_to_ostim_to_osa_subset_is_semantically_equal(fixture_root: Path, tmp_path: Path) -> None:
    request = ConversionRequest(
        fixture_root / "osa_namespaced",
        SourceFormat.OSTIM_SA,
        tmp_path / "back-to-osa",
        SourceFormat.OSA_OSEX,
        ConversionMode.BEST_EFFORT,
        PackageMode.DIRECTORY,
        "none",
        "RoundTripPack",
    )
    result = ConverterService().roundtrip(request, through=SourceFormat.OSTIM_SA)
    assert result.second is not None and result.second.output_written
    assert result.semantically_equal
    assert not any(item.code == "SCENE_ID_UNSAFE" for item in result.second.diagnostics)
