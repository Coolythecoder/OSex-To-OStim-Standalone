from __future__ import annotations

import json
from pathlib import Path

from animation_converter.models import ConversionMode, SourceFormat
from animation_converter.packaging import PackageMode
from animation_converter.service import ConversionRequest, ConverterService


def test_behavior_manifest_uses_target_event_and_effective_pack_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scene_root = source / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / "Source"
    animation_root = source / "Data" / "meshes" / "actors" / "character" / "animations" / "Source"
    scene_root.mkdir(parents=True)
    animation_root.mkdir(parents=True)
    (scene_root / "Scene.json").write_text(
        json.dumps(
            {
                "name": "Scene",
                "modpack": "Source",
                "length": 2,
                "speeds": [{"animation": "Event With Space"}],
                "actors": [{}],
            }
        ),
        encoding="utf-8",
    )
    (animation_root / "Event With Space_0.hkx").write_bytes(b"hkx bytes")
    (source / "LICENSE.txt").write_text("Synthetic fixture permission.", encoding="utf-8")

    output = tmp_path / "output"
    result = ConverterService().convert(
        ConversionRequest(
            source,
            SourceFormat.OSTIM_SA,
            output,
            SourceFormat.OSTIM_SA,
            ConversionMode.NORMAL,
            PackageMode.DIRECTORY,
            "pandora",
            "OStimThirdParty",
        )
    )
    assert result.output_written
    assert result.emit_result.output_pack_id == "Pack_OStimThirdParty"
    behavior_path = (
        output
        / "Data"
        / "SKSE"
        / "Plugins"
        / "OStim"
        / "converter_metadata"
        / "Pack_OStimThirdParty"
        / "behavior-registration.json"
    )
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    assert behavior["registrations"][0]["animationEvent"] == "Event_With_Space"
    assert "Pack_OStimThirdParty" in behavior["registrations"][0]["hkxPath"]
    manifest = json.loads((output / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputPackIdentifier"] == "Pack_OStimThirdParty"
    assert "LICENSE.txt" in manifest["sourceFileHashes"]
    assert not result.readiness.install_ready
