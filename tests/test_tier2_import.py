from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from animation_converter.adapters.base import ParseContext
from animation_converter.adapters.flowergirls import FlowerGirlsAdapter
from animation_converter.adapters.slal import SlalAdapter
from animation_converter.assets import assign_ostim_target_paths, discover_animation_assets
from animation_converter.service import ConverterService


def test_slal_import_maps_complete_actor_stages(fixture_root: Path) -> None:
    ir = SlalAdapter().parse(fixture_root / "slal", ParseContext())
    assert len(ir.graph.nodes) == 1
    scene = ir.graph.nodes[0]
    assert len(scene.actors) == 2
    assert [speed.animation for speed in scene.speeds] == ["FixtureCouple_S1", "FixtureCouple_S2"]
    assert [actor.intended_sex for actor in scene.actors] == ["male", "female"]
    assert len(ir.events) == 2
    assert all(len(event.actor_assets) == 2 for event in ir.events)
    assert ir.losses


def test_slal_import_matches_author_prefixed_stage_ids_to_local_hkx_branch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    local_json = source / "PackSE" / "SLAnims" / "json" / "Pack.json"
    local_hkx = source / "PackSE" / "meshes" / "actors" / "character" / "animations" / "Pack"
    duplicate_hkx = source / "PackLE" / "meshes" / "actors" / "character" / "animations" / "Pack"
    local_json.parent.mkdir(parents=True)
    local_hkx.mkdir(parents=True)
    duplicate_hkx.mkdir(parents=True)
    local_json.write_text(
        json.dumps(
            {
                "name": "Prefixed Pack",
                "animations": [
                    {
                        "id": "B_Pose",
                        "actors": [
                            {"type": "Female", "stages": [{"id": "B_Pose_A1_S1"}]},
                            {"type": "Male", "stages": [{"id": "B_Pose_A2_S1"}]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for root in (local_hkx, duplicate_hkx):
        (root / "Pose_A1_S1.hkx").write_bytes(b"actor one")
        (root / "Pose_A2_S1.hkx").write_bytes(b"actor two")

    ir = SlalAdapter().parse(source, ParseContext())

    assert len(ir.graph.nodes) == 1
    assert len(ir.events) == 1
    mapped = {asset_id for asset_id in ir.events[0].actor_assets.values()}
    mapped_paths = {asset.source_path for asset in ir.assets if asset.id in mapped}
    assert mapped_paths == {
        "PackSE/meshes/actors/character/animations/Pack/Pose_A1_S1.hkx",
        "PackSE/meshes/actors/character/animations/Pack/Pose_A2_S1.hkx",
    }


def test_slal_import_disambiguates_human_and_creature_actor_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    json_path = source / "CreaturePack" / "SLAnims" / "json" / "CreaturePack.json"
    human_hkx = source / "CreaturePack" / "meshes" / "actors" / "character" / "animations" / "CreaturePack"
    creature_hkx = source / "CreaturePack" / "meshes" / "actors" / "horse" / "animations" / "CreaturePack"
    legacy_creature_hkx = creature_hkx / "HorseOld"
    json_path.parent.mkdir(parents=True)
    human_hkx.mkdir(parents=True)
    creature_hkx.mkdir(parents=True)
    legacy_creature_hkx.mkdir(parents=True)
    json_path.write_text(
        json.dumps(
            {
                "name": "Creature Pack",
                "animations": [
                    {
                        "id": "B_HorsePose",
                        "actors": [
                            {"type": "Female", "stages": [{"id": "B_HorsePose_A1_S1"}]},
                            {
                                "type": "CreatureMale",
                                "race": "Horses",
                                "stages": [{"id": "B_HorsePose_A2_S1"}],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for root in (human_hkx, creature_hkx):
        (root / "HorsePose_A1_S1.hkx").write_bytes(b"actor one")
        (root / "HorsePose_A2_S1.hkx").write_bytes(b"actor two")
    (legacy_creature_hkx / "HorsePose_A2_S1.hkx").write_bytes(b"legacy actor two")

    ir = SlalAdapter().parse(source, ParseContext())

    assert len(ir.events) == 1
    assert ir.diagnostics.counts()["ERROR"] == 0
    scene = ir.graph.nodes[0]
    assert [(actor.type, actor.intended_sex) for actor in scene.actors] == [
        ("npc", "female"),
        ("creature", "male"),
    ]
    mapped = {
        actor_index: next(asset for asset in ir.assets if asset.id == asset_id)
        for actor_index, asset_id in ir.events[0].actor_assets.items()
    }
    assert mapped[0].source_path.endswith("actors/character/animations/CreaturePack/HorsePose_A1_S1.hkx")
    assert mapped[1].source_path.endswith("actors/horse/animations/CreaturePack/HorsePose_A2_S1.hkx")

    assign_ostim_target_paths(ir, "CreaturePack", {ir.events[0].name: ir.events[0].name})
    assert mapped[0].target_path == "Data/meshes/actors/character/animations/CreaturePack/B_HorsePose_S1_0.hkx"
    assert mapped[1].target_path == "Data/meshes/actors/horse/animations/CreaturePack/B_HorsePose_S1_1.hkx"


def test_slal_multi_json_pack_uses_source_name_instead_of_last_subpack(tmp_path: Path) -> None:
    source = tmp_path / "Billyy Complete Bundle"
    json_root = source / "SLAnims" / "json"
    animation_root = source / "meshes" / "actors" / "character" / "animations" / "Billyy"
    json_root.mkdir(parents=True)
    animation_root.mkdir(parents=True)
    for index, subpack in enumerate(("Billyy Human", "Billyy Orgy"), start=1):
        event = f"BillyyScene{index}_A1_S1"
        (json_root / f"part{index}.json").write_text(
            json.dumps(
                {
                    "name": subpack,
                    "animations": [
                        {
                            "id": f"BillyyScene{index}",
                            "actors": [{"type": "Female", "stages": [{"id": event}]}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (animation_root / f"BillyyScene{index}_A1_S1.hkx").write_bytes(b"animation")

    ir = SlalAdapter().parse(source, ParseContext())

    assert ir.pack.display_name == source.name
    assert ir.pack.identifier == "Billyy_Complete_Bundle"
    assert len(ir.graph.nodes) == 2


def test_service_uses_original_archive_name_for_multi_json_slal_pack(tmp_path: Path) -> None:
    archive_path = tmp_path / "Billyy Full Archive v9.9.zip"
    with ZipFile(archive_path, "w") as archive:
        for index, subpack in enumerate(("Billyy Human", "Billyy Orgy"), start=1):
            event = f"BillyyScene{index}_A1_S1"
            archive.writestr(
                f"SLAnims/json/part{index}.json",
                json.dumps(
                    {
                        "name": subpack,
                        "animations": [
                            {
                                "id": f"BillyyScene{index}",
                                "actors": [{"type": "Female", "stages": [{"id": event}]}],
                            }
                        ],
                    }
                ),
            )
            archive.writestr(f"meshes/actors/character/animations/Billyy/{event}.hkx", b"animation")

    result = ConverterService().inspect(archive_path)

    assert result.ir is not None
    assert result.ir.pack.display_name == archive_path.stem
    assert result.ir.pack.identifier == "Billyy_Full_Archive_v9.9"


def test_animation_discovery_excludes_behavior_graph_hkx(tmp_path: Path) -> None:
    animation = tmp_path / "meshes" / "actors" / "character" / "animations" / "Pack" / "Scene_0.hkx"
    behavior = tmp_path / "meshes" / "actors" / "character" / "Behaviors" / "FNIS_Pack_Behavior.hkx"
    animation.parent.mkdir(parents=True)
    behavior.parent.mkdir(parents=True)
    animation.write_bytes(b"animation")
    behavior.write_bytes(b"behavior graph")

    assets = discover_animation_assets(tmp_path)

    assert [asset.source_path for asset in assets] == [
        "meshes/actors/character/animations/Pack/Scene_0.hkx"
    ]


def test_flowergirls_import_uses_only_fnis_references(fixture_root: Path) -> None:
    ir = FlowerGirlsAdapter().parse(fixture_root / "flowergirls", ParseContext())
    assert len(ir.graph.nodes) == 1
    scene = ir.graph.nodes[0]
    assert scene.source_id == "FlowerKiss"
    assert len(scene.actors) == 2
    assert [speed.animation for speed in scene.speeds] == ["FlowerKiss_S1", "FlowerKiss_S2"]
    assert all(len(event.actor_assets) == 2 for event in ir.events)
    assert "flowergirls" in scene.tags


def test_tier2_reverse_export_is_explicitly_unsupported(fixture_root: Path, tmp_path: Path) -> None:
    for adapter, fixture in ((SlalAdapter(), "slal"), (FlowerGirlsAdapter(), "flowergirls")):
        ir = adapter.parse(fixture_root / fixture, ParseContext())
        assert not adapter.capabilities().export_supported
        try:
            adapter.emit(
                ir, tmp_path, __import__("animation_converter.adapters.base", fromlist=["EmitContext"]).EmitContext()
            )
        except NotImplementedError:
            pass
        else:
            raise AssertionError("Tier 2 reverse export must not silently succeed")
