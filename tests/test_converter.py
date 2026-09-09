import ast
import importlib.util
import copy
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PurePosixPath
from unittest import mock
from zipfile import ZipFile as _ZipFile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Osex-to-Ostim-Standalone.py"
CONVERT_SCRIPT = ROOT / "convert.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)

convert_spec = importlib.util.spec_from_file_location("convert_cli", CONVERT_SCRIPT)
convert_cli = importlib.util.module_from_spec(convert_spec)
assert convert_spec.loader is not None
sys.modules[convert_spec.name] = convert_cli
convert_spec.loader.exec_module(convert_cli)

from tools import release_check


class ZipFile(_ZipFile):
    def getinfo(self, name):
        try:
            return super().getinfo(name)
        except KeyError:
            if isinstance(name, str) and name.startswith("Data/"):
                return super().getinfo(name[5:])
            raise

    def namelist(self):
        names = super().namelist()
        aliases = [f"Data/{name}" for name in names if not name.startswith("Data/")]
        return names + aliases


def mod_entry(path: str) -> str:
    return path[5:] if path.startswith("Data/") else path


def hkx_packfile_bytes(pointer_size: int, file_version: int = 8, payload: bytes = b"") -> bytes:
    header = bytearray(32)
    header[:8] = converter.HKX_PACKFILE_MAGIC
    header[12:16] = int(file_version).to_bytes(4, byteorder="little", signed=False)
    header[16] = pointer_size
    header[17] = 1
    return bytes(header) + payload


def character_att_list_paths(names: set[str], _pack_folder: str) -> list[str]:
    prefix = "Data/meshes/actors/character/animations/"
    return sorted(
        {
            name
            for name in names
            if name.startswith(prefix)
            and PurePosixPath(name).name.startswith("ATT_")
            and name.endswith("_animlist.txt")
        }
    )


def single_character_att_list_path(names: set[str], pack_folder: str) -> str:
    paths = character_att_list_paths(names, pack_folder)
    if len(paths) != 1:
        raise AssertionError(f"Expected one character ATT list for {pack_folder}, found {paths}")
    return paths[0]


def att_speed_var_from_list_path(path: str) -> str:
    name = PurePosixPath(path).name
    list_code = name.removeprefix("ATT_").removesuffix("_animlist.txt")
    return f"{converter.sanitize_name(list_code, 'ConvertedPack').upper()}_AnimationSpeed"


def read_pandora_compatible_events(
    archive: ZipFile,
    names: set[str],
    pack_folder: str,
    actor_root: str = "character",
) -> str:
    if actor_root == "character":
        path = single_character_att_list_path(names, pack_folder)
    else:
        prefix = f"Data/meshes/actors/{actor_root}/animations/{pack_folder}/FNIS_"
        paths = sorted(
            name for name in names
            if name.startswith(prefix) and name.endswith("_List.txt")
        )
        if len(paths) != 1:
            raise AssertionError(f"Expected one FNIS list for {actor_root}, found {paths}")
        path = paths[0]
    return archive.read(mod_entry(path)).decode("utf-8")


def assert_pandora_compatible_registration(
    testcase: unittest.TestCase,
    names: set[str],
    pack_folder: str,
    project: str = "DefaultMale",
) -> str:
    """Assert the Nemesis/ATT and FNIS inputs that Pandora actually assembles."""
    att_paths = character_att_list_paths(names, pack_folder)
    if att_paths:
        code = PurePosixPath(att_paths[0]).name.removeprefix("ATT_").removesuffix("_animlist.txt")
    else:
        info_prefix = "Data/Nemesis_Engine/mod/"
        codes = sorted(
            name[len(info_prefix):].split("/", 1)[0]
            for name in names
            if name.startswith(info_prefix) and name.endswith("/info.ini")
        )
        if not codes:
            raise AssertionError(f"No Pandora-compatible Nemesis module was packaged for {pack_folder}")
        code = codes[0]
    testcase.assertIn(f"Data/Nemesis_Engine/mod/{code}/info.ini", names)
    testcase.assertTrue(
        any(name.startswith(f"Data/Nemesis_Engine/mod/{code}/0_master/") for name in names),
        f"No generated 0_master patch was packaged for {pack_folder}",
    )
    if project.casefold() == "defaultmale":
        single_character_att_list_path(names, pack_folder)
        return code

    actor_roots = {
        "horseproject": "horse",
        "chaurusflyer": "dlc01/chaurusflyer",
        "dogproject": "canine",
        "wolfproject": "canine",
        "rieklingproject": "dlc02/riekling",
    }
    actor_root = actor_roots[project.casefold()]
    fnis_prefix = f"Data/meshes/actors/{actor_root}/animations/{pack_folder}/FNIS_"
    testcase.assertTrue(
        any(name.startswith(fnis_prefix) and name.endswith("_List.txt") for name in names),
        f"No Pandora-readable FNIS list was packaged for actor root {actor_root}",
    )
    return code


def write_ostim_scene_pack(
    archive: Path,
    scene_data: dict,
    events: dict[str, bytes],
    scene_name: str = "Scene",
    scene_pack: str = "NativePack",
) -> None:
    with ZipFile(archive, "w") as source:
        source.writestr(f"Data/SKSE/Plugins/OStim/scenes/{scene_pack}/{scene_name}.json", json.dumps(scene_data))
        for event, payload in events.items():
            source.writestr(f"Data/meshes/actors/character/animations/{scene_pack}/{event}.hkx", payload)


def write_pandora_ostim_zip(
    archive: Path,
    *,
    pack: str = "PandoraPack",
    prefix: str = "Data",
    include_info: bool = True,
    animationdata_text: str | None = "Pose_0\nPose_1\n",
    animationsetdata_text: str | None = (
        "meshes\\actors\\character\\animations\\PandoraPack\\Pose_0.hkx\n"
        "meshes\\actors\\character\\animations\\PandoraPack\\Pose_1.hkx\n"
    ),
    include_hkx: bool = True,
    scene_speed: str = "Pose",
) -> str:
    code = converter.safe_behavior_mod_code(pack)
    root = prefix.strip("/").rstrip("/")
    root = f"{root}/" if root else ""
    scene_data = {
        "name": "Pose Scene",
        "modpack": pack,
        "length": 3,
        "speeds": [{"animation": scene_speed}],
        "actors": [{}, {}],
        "actions": [{"type": "kissing", "actor": 0, "target": 1}],
    }
    with ZipFile(archive, "w") as source:
        source.writestr(f"{root}SKSE/Plugins/OStim/scenes/{pack}/Scene.json", json.dumps(scene_data))
        if include_hkx:
            source.writestr(f"{root}meshes/actors/character/animations/{pack}/Pose_0.hkx", b"")
            source.writestr(f"{root}meshes/actors/character/animations/{pack}/Pose_1.hkx", b"")
        if include_info:
            source.writestr(
                f"{root}Pandora_Engine/mod/{code}/info.xml",
                f'<mod code="{code}"><name>{pack}</name><author>Tester</author><site>Unit test</site></mod>\n',
            )
        if animationdata_text is not None:
            source.writestr(f"{root}Pandora_Engine/mod/{code}/animationdata/DefaultMale.txt", animationdata_text)
        if animationsetdata_text is not None:
            source.writestr(f"{root}Pandora_Engine/mod/{code}/animationsetdata/DefaultMale.txt", animationsetdata_text)
            source.writestr(f"{root}Pandora_Engine/mod/{code}/animationsetdata/DefaultMale/H2HDual.txt", animationsetdata_text)
    return code


def make_slsb_animation(animation_id: str, *, creature: bool = False, furniture: str = "") -> dict:
    actors = [
        {"type": "CreatureMale" if creature else "Female", "race": "Canine" if creature else "Human"},
        {"type": "Female" if creature else "Male", "race": "Human"},
    ]
    for actor_index, actor in enumerate(actors, start=1):
        actor["stages"] = [{"id": f"{animation_id}_A{actor_index}_S1"}]
    return {
        "id": animation_id,
        "name": animation_id.replace("_", " "),
        "tags": "creature,canine" if creature else "MF,vaginal",
        "actors": actors,
        "stages": [{"number": 1, "timer": 4.0}],
        "sound": "Squishing",
        "furniture": furniture,
    }


def write_osa_folder_source(root: Path) -> Path:
    source = root / "Extracted OSA Source"
    scene_dir = source / "Data" / "Meshes" / "0SA" / "mod" / "0Sex" / "scene" / "AA" / "Standing" / "HJ"
    hkx_dir = source / "Data" / "meshes" / "actors" / "character" / "animations" / "0Sex" / "AA" / "Standing" / "HJ"
    scene_dir.mkdir(parents=True)
    hkx_dir.mkdir(parents=True)
    (scene_dir / "Base.xml").write_text(
        """\
<scene id="AA|Standing|HJ|Base" actors="2">
  <info name="Base" />
  <anim id="BaseLoop" l="3" />
</scene>
""",
        encoding="utf-8",
    )
    (hkx_dir / "BaseLoop_0.hkx").write_bytes(b"hkx0")
    (hkx_dir / "BaseLoop_1.hkx").write_bytes(b"hkx1")
    return source


class ConverterTests(unittest.TestCase):
    def test_gui_background_jobs_do_not_read_tk_state(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        run_gui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch_gui")
        background_jobs = [
            node
            for node in ast.walk(run_gui)
            if isinstance(node, ast.FunctionDef) and (node.name == "job" or node.name.endswith("_job"))
        ]
        violations = []
        for job in background_jobs:
            for call in (node for node in ast.walk(job) if isinstance(node, ast.Call)):
                function = call.func
                if not isinstance(function, ast.Attribute) or function.attr != "get":
                    continue
                value = function.value
                if not isinstance(value, ast.Subscript) or not isinstance(value.value, ast.Name):
                    continue
                if value.value.id == "state":
                    violations.append((job.name, call.lineno))

        self.assertEqual(violations, [], f"Tk state read from worker thread at: {violations}")

    def test_gui_quick_conversion_work_stays_inside_background_job(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        run_gui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch_gui")
        quick_convert = next(
            node for node in run_gui.body if isinstance(node, ast.FunctionDef) and node.name == "quick_convert_archive"
        )
        job = next(node for node in quick_convert.body if isinstance(node, ast.FunctionDef) and node.name == "job")

        outside_job_calls = [
            call
            for statement in quick_convert.body
            if statement is not job
            for call in ast.walk(statement)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "job_log"
        ]
        called_names = {
            call.func.id
            for call in ast.walk(job)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }

        self.assertEqual(outside_job_calls, [])
        self.assertIn("convert_archive_to_ready_zip", called_names)
        self.assertIn("convert_archive_to_sexlab_zip", called_names)

    def test_gui_advanced_job_does_not_shadow_zip_out(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        run_gui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch_gui")
        on_run = next(node for node in run_gui.body if isinstance(node, ast.FunctionDef) and node.name == "on_run")
        job = next(node for node in on_run.body if isinstance(node, ast.FunctionDef) and node.name == "job")

        zip_out_stores = [
            node for node in ast.walk(job) if isinstance(node, ast.Name) and node.id == "zip_out" and isinstance(node.ctx, ast.Store)
        ]
        job_zip_out_stores = [
            node
            for node in ast.walk(job)
            if isinstance(node, ast.Name) and node.id == "job_zip_out" and isinstance(node.ctx, ast.Store)
        ]

        self.assertEqual(zip_out_stores, [])
        self.assertGreaterEqual(len(job_zip_out_stores), 2)

    def test_gui_recommended_build_respects_human_only_checkbox(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        run_gui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch_gui")
        recommended_build = next(
            node for node in run_gui.body if isinstance(node, ast.FunctionDef) and node.name == "build_recommended_package"
        )
        job = next(node for node in recommended_build.body if isinstance(node, ast.FunctionDef) and node.name == "job")

        recommendation_assignments = [
            node
            for node in ast.walk(job)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "effective_human_only_ostim" for target in node.targets)
            and isinstance(node.value, ast.Name)
            and node.value.id == "recommended_human_only"
        ]

        self.assertEqual(recommendation_assignments, [])

    def test_convert_py_builds_extracted_folder_and_requested_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_osa_folder_source(root)
            output_zip = root / "ConvertedFolder.zip"

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = convert_cli.run_cli(
                    [
                        "--input",
                        str(source),
                        "--output",
                        str(output_zip),
                        "--overwrite",
                        "--report-json",
                        "--report-md",
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertTrue(output_zip.exists())
            self.assertTrue((root / "ConvertedFolder_conversion_report.json").exists())
            self.assertTrue((root / "ConvertedFolder_conversion_report.md").exists())
            verification = converter.verify_converted_zip(output_zip, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            names = set(ZipFile(output_zip).namelist())
            self.assertIn("Data/SKSE/Plugins/OStim/scenes/Extracted_OSA_Source/Extracted_OSA_Source_AA_Standing_HJ_Base.json", names)
            assert_pandora_compatible_registration(self, names, "Extracted_OSA_Source")

    def test_convert_py_dry_run_folder_writes_diagnosis_without_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_osa_folder_source(root)
            output_zip = root / "DryRun.zip"

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = convert_cli.run_cli(
                    [
                        "--input",
                        str(source),
                        "--output",
                        str(output_zip),
                        "--dry-run",
                        "--report-json",
                        "--report-md",
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertFalse(output_zip.exists())
            self.assertTrue((root / "DryRun_source_diagnosis.json").exists())
            self.assertTrue((root / "DryRun_source_diagnosis.md").exists())

    def test_convert_py_validate_only_checks_converted_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_osa_folder_source(root)
            output_zip = root / "ValidatedFolder.zip"
            with redirect_stdout(io.StringIO()):
                build_exit = convert_cli.run_cli(
                    [
                        "--input",
                        str(source),
                        "--output",
                        str(output_zip),
                        "--overwrite",
                    ]
                )
            self.assertEqual(build_exit, 0)

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = convert_cli.run_cli(
                    [
                        "--input",
                        str(output_zip),
                        "--validate-only",
                        "--report-md",
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertTrue((root / "ValidatedFolder_deploy_verify.md").exists())

    def test_osa_source_exports_ostim_tools_project_with_alignment_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "OSA Source.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Base.xml",
                    """\
<scene id="AA|Standing|HJ|Base" actors="2">
  <info name="Base Handjob" />
  <anim id="BaseHandjob" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_1.hkx", b"")
            project_out = root / "OsaProject"
            options = converter.OStimToolsProjectOptions(alignment_template=True)

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=project_out, options=options)

            project = json.loads((project_out / converter.OSTIM_TOOLS_PROJECT_FILE).read_text(encoding="utf-8"))
            scene_json = json.loads((project_out / project["scenes"][0]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(result.report["outputTarget"], "OStim Tools JSON Project")
            self.assertEqual(project["counts"]["scenes"], 1)
            self.assertEqual(len(scene_json["actors"]), 2)
            self.assertEqual(scene_json["speeds"][0]["animation"], "BaseHandjob")
            self.assertTrue(scene_json["actions"])
            self.assertNotIn("aacWarnings", scene_json)
            self.assertNotIn("aacSource", scene_json)
            self.assertIn("ostimToolsBridgeConfig", project["paths"])
            bridge = json.loads((project_out / project["paths"]["ostimToolsBridgeConfig"]).read_text(encoding="utf-8"))
            self.assertEqual(len(bridge), 1)
            bridge_entry = next(iter(bridge.values()))
            self.assertEqual(bridge_entry["actorsKeyword"], "")
            self.assertEqual(bridge_entry["stages"][0]["fileName"], "BaseHandjob")
            self.assertTrue((project_out / "alignment" / "alignment.json").exists())
            self.assertIn("not game-installable", result.report["recommendedUserAction"])
            self.assertFalse(converter.project_json_contains_full_path(project))

    def test_slal_source_exports_ostim_tools_project_with_inferred_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SLAL Source.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Ace_Test.json",
                    json.dumps(
                        {
                            "name": "Ace_Test",
                            "animations": [
                                {
                                    "id": "Ace_TestFootjob",
                                    "name": "Ace Test Footjob",
                                    "tags": "Ace,footjob,feet,MF",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "Ace_TestFootjob_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "Ace_TestFootjob_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")
            project_out = root / "SlalProject"

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=project_out)

            project = json.loads((project_out / converter.OSTIM_TOOLS_PROJECT_FILE).read_text(encoding="utf-8"))
            scene_json = json.loads((project_out / project["scenes"][0]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(result.report["detectedSourceType"], "SexLab/SLAL animation JSON")
            self.assertEqual(scene_json["speeds"][0]["animation"], "Ace_TestFootjob_S1")
            self.assertIn("footjob", scene_json["tags"])
            self.assertEqual(scene_json["actions"][0]["type"], "footjob")
            self.assertEqual(scene_json["actions"][0]["actor"], 0)
            self.assertEqual(scene_json["actions"][0]["target"], 1)
            bridge = json.loads((project_out / project["paths"]["ostimToolsBridgeConfig"]).read_text(encoding="utf-8"))
            bridge_entry = next(iter(bridge.values()))
            self.assertEqual(bridge_entry["actorsKeyword"], "fm")
            self.assertEqual(bridge_entry["stages"][0]["meta"]["tags"], scene_json["tags"])
            self.assertEqual(bridge_entry["stages"][0]["actions"][0]["type"], "footjob")

    def test_ostim_tools_project_export_validate_and_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "NativePack.zip"
            scene_data = {
                "name": "Manual Scene",
                "modpack": "NativePack",
                "length": 4,
                "speeds": [{"animation": "ManualScene"}],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "kissing", "actor": 0, "target": 1}],
                "tags": ["romance"],
            }
            write_ostim_scene_pack(
                archive,
                scene_data,
                {"ManualScene_0": b"hkx0", "ManualScene_1": b"hkx1"},
                scene_name="ManualScene",
                scene_pack="NativePack",
            )
            project_out = root / "Project"

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=project_out)

            project = json.loads((project_out / converter.OSTIM_TOOLS_PROJECT_FILE).read_text(encoding="utf-8"))
            self.assertEqual(project["schema"], converter.OSTIM_TOOLS_PROJECT_SCHEMA)
            self.assertFalse(project["gameInstallable"])
            self.assertEqual(project["counts"]["scenes"], 1)
            self.assertEqual(project["ostimToolsCompatibility"]["upstreamVersionInspected"], converter.OSTIM_TOOLS_UPSTREAM_VERSION)
            self.assertTrue(project["paths"]["ostimToolsBridgeConfig"].endswith("ScenesConfig.json"))
            self.assertTrue((project_out / "scenes" / project["scenes"][0]["file"].split("/", 1)[1]).exists())
            self.assertTrue((project_out / "actions").exists())
            self.assertIn("not a complete in-game mod package", (project_out / converter.AAC_README_FILE).read_text(encoding="utf-8"))
            self.assertFalse(re.search(r"[A-Za-z]:\\\\", json.dumps(project)))
            self.assertIn("not game-installable", result.report["recommendedUserAction"])

            validation = converter.validate_ostim_tools_project(project_out, write_report=False)
            self.assertTrue(validation.ok, validation.report["errors"])
            self.assertEqual(validation.report["hkxAssetCount"], 2)
            self.assertTrue(validation.report["ostimToolsBridgeConfigDeclared"])
            self.assertEqual(validation.report["ostimToolsBridgeConfigEntries"], 1)
            imported = converter.import_ostim_tools_project(project_out)
            self.assertEqual(len(imported.scenes), 1)
            self.assertEqual(len(imported.hkx_assets), 2)
            self.assertEqual(imported.scenes[0].actions[0].type, "kissing")
            ready_zip = root / "Imported_OStimSA.zip"
            packaged = converter.convert_imported_ostim_tools_project_to_ready_zip(project_out, ready_zip, ostim_menu_entry=True)
            self.assertIsNotNone(packaged.verification)
            self.assertTrue(packaged.verification.ok, packaged.verification.report["errors"])

    def test_ostim_tools_cli_import_writes_scene_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ImportNative.zip"
            scene_data = {
                "name": "Import Scene",
                "modpack": "ImportNative",
                "length": 4,
                "speeds": [{"animation": "ImportScene"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            write_ostim_scene_pack(
                archive,
                scene_data,
                {"ImportScene_0": b"hkx"},
                scene_name="ImportScene",
                scene_pack="ImportNative",
            )
            project_out = root / "Project"
            scenes_out = root / "ImportedScenes"
            converter.convert_archive_to_ostim_tools_project(archive, project_out=project_out)

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = converter.run_cli(
                    [
                        "--import-ostim-tools-project",
                        str(project_out),
                        "--output-scenes",
                        str(scenes_out),
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertTrue(list((scenes_out / "ImportNative").glob("*.json")))

    def test_scene_json_folder_mode_writes_no_project_manifest_or_action_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SceneFolder.zip"
            scene_data = {
                "name": "Folder Scene",
                "modpack": "SceneFolder",
                "length": 4,
                "speeds": [{"animation": "FolderScene"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            write_ostim_scene_pack(archive, scene_data, {"FolderScene_0": b"hkx"}, scene_name="FolderScene", scene_pack="SceneFolder")
            out = root / "SceneJSON"
            options = converter.OStimToolsProjectOptions(scene_json_folder_only=True)

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=out, options=options)

            self.assertEqual(result.report["outputTarget"], "Scene JSON Folder")
            self.assertFalse((out / converter.OSTIM_TOOLS_PROJECT_FILE).exists())
            self.assertTrue(list((out / "scenes").glob("*.json")))
            self.assertFalse((out / "actions").exists())
            self.assertFalse((out / "animations").exists())
            self.assertFalse((out / "configs").exists())
            self.assertIn("editable scene JSON folder", (out / converter.AAC_README_FILE).read_text(encoding="utf-8"))

    def test_ostim_tools_export_strips_aac_only_scene_fields_for_official_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "StrictScene.zip"
            scene_data = {
                "name": "Strict Scene",
                "modpack": "StrictScene",
                "length": 4,
                "speeds": [{"animation": "StrictScene"}],
                "actors": [
                    {
                        "intendedSex": "female",
                        "animationIndex": 0,
                        "underlyingExpression": "legacy",
                        "requirements": ["aac"],
                        "creatureRace": "wolf",
                    }
                ],
                "actions": [{"type": "kissing", "actor": 0, "muted": True, "doPeaks": False}],
                "fadeOnEntry": True,
                "scaleOffsetWithFurniture": True,
            }
            write_ostim_scene_pack(archive, scene_data, {"StrictScene_0": b"hkx"}, scene_name="StrictScene", scene_pack="StrictScene")
            out = root / "Project"

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=out)

            project = json.loads((out / converter.OSTIM_TOOLS_PROJECT_FILE).read_text(encoding="utf-8"))
            exported_scene = json.loads((out / project["scenes"][0]["file"]).read_text(encoding="utf-8"))
            self.assertTrue(result.report["officialOStimToolsSceneSchema"])
            self.assertNotIn("fadeOnEntry", exported_scene)
            self.assertNotIn("scaleOffsetWithFurniture", exported_scene)
            self.assertNotIn("aacSource", exported_scene)
            self.assertNotIn("underlyingExpression", exported_scene["actors"][0])
            self.assertNotIn("requirements", exported_scene["actors"][0])
            self.assertNotIn("creatureRace", exported_scene["actors"][0])
            self.assertNotIn("muted", exported_scene["actions"][0])
            self.assertNotIn("doPeaks", exported_scene["actions"][0])

    def test_ostim_tools_validation_fails_missing_scene_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "Project"
            scenes = project / "scenes"
            scenes.mkdir(parents=True)
            scene_data = {
                "name": "Duplicate",
                "modpack": "Project",
                "length": 4,
                "speeds": [{"animation": "Duplicate"}],
                "actors": [{"animationIndex": 0}],
            }
            (scenes / "One.json").write_text(json.dumps(scene_data), encoding="utf-8")
            (scenes / "Two.json").write_text(json.dumps(scene_data), encoding="utf-8")
            manifest = {
                "schema": converter.OSTIM_TOOLS_PROJECT_SCHEMA,
                "schemaVersion": converter.OSTIM_TOOLS_PROJECT_SCHEMA_VERSION,
                "paths": {"scenes": "scenes/", "alignment": "alignment/missing.json"},
                "scenes": [
                    {"id": "SameId", "file": "scenes/One.json"},
                    {"id": "SameId", "file": "scenes/Two.json"},
                    {"id": "Missing", "file": "scenes/Missing.json"},
                ],
            }
            (project / converter.OSTIM_TOOLS_PROJECT_FILE).write_text(json.dumps(manifest), encoding="utf-8")

            validation = converter.validate_ostim_tools_project(project, write_report=False)

            self.assertFalse(validation.ok)
            self.assertTrue(any("Duplicate scene ID" in error for error in validation.report["errors"]))
            self.assertTrue(any("listed scene file does not exist" in error for error in validation.report["errors"]))
            self.assertTrue(any("Declared alignment file is missing" in error for error in validation.report["errors"]))

    def test_ostim_tools_unknown_schema_warns_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "FutureProject"
            scenes = project / "scenes"
            scenes.mkdir(parents=True)
            (scenes / "Scene.json").write_text(
                json.dumps(
                    {
                        "name": "Future Scene",
                        "modpack": "FutureProject",
                        "length": 4,
                        "speeds": [{"animation": "FutureScene"}],
                        "actors": [{"animationIndex": 0}],
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "schema": "Future.OStimTools.Project",
                "schemaVersion": 99,
                "paths": {"scenes": "scenes/"},
                "scenes": [{"id": "FutureScene", "file": "scenes/Scene.json"}],
            }
            (project / converter.OSTIM_TOOLS_PROJECT_FILE).write_text(json.dumps(manifest), encoding="utf-8")

            validation = converter.validate_ostim_tools_project(project, write_report=False)
            imported = converter.import_ostim_tools_project(project)

            self.assertTrue(validation.ok, validation.report["errors"])
            self.assertTrue(any("Unknown OStim Tools project schema" in warning for warning in validation.report["warnings"]))
            self.assertEqual(len(imported.scenes), 1)
            self.assertTrue(any("Unknown OStim Tools project schema" in warning for warning in imported.warnings))

    def test_no_attribute_error_when_ostimtools_scene_entry_is_string(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "Project"
            project.mkdir(parents=True)
            manifest = {
                "schema": converter.OSTIM_TOOLS_PROJECT_SCHEMA,
                "schemaVersion": converter.OSTIM_TOOLS_PROJECT_SCHEMA_VERSION,
                "pack": "Malformed Project",
                "scenes": ["not an object"],
            }
            (project / converter.OSTIM_TOOLS_PROJECT_FILE).write_text(json.dumps(manifest), encoding="utf-8")

            validation = converter.validate_ostim_tools_project(project, write_report=False)

            self.assertFalse(validation.ok)
            self.assertTrue(any("project manifest scene entry 1" in error for error in validation.report["errors"]))
            self.assertNotIn("AttributeError", "\n".join(validation.report["errors"]))

    def test_ostimtools_manifest_string_sections_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "Project"
            project.mkdir(parents=True)
            manifest = {
                "schema": converter.OSTIM_TOOLS_PROJECT_SCHEMA,
                "schemaVersion": converter.OSTIM_TOOLS_PROJECT_SCHEMA_VERSION,
                "pack": "String Sections",
                "scenes": "not an array",
                "paths": "not an object",
                "settings": "not an object",
                "counts": "not an object",
                "assets": "not an object",
            }
            (project / converter.OSTIM_TOOLS_PROJECT_FILE).write_text(json.dumps(manifest), encoding="utf-8")

            validation = converter.validate_ostim_tools_project(project, write_report=False)

            self.assertFalse(validation.ok)
            error_text = "\n".join(validation.report["errors"])
            self.assertIn("project manifest field 'scenes'", error_text)
            self.assertIn("project manifest field 'paths'", error_text)
            self.assertIn("project manifest field 'settings'", error_text)
            self.assertIn("project manifest field 'counts'", error_text)
            self.assertIn("project manifest field 'assets'", error_text)

    def test_ostim_tools_template_fields_are_preserved_and_cli_alias_works(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "TemplatePack.zip"
            scene_data = {
                "name": "Template Scene",
                "modpack": "TemplatePack",
                "length": 4,
                "speeds": [{"animation": "TemplateScene"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            write_ostim_scene_pack(archive, scene_data, {"TemplateScene_0": b"hkx"}, scene_name="TemplateScene", scene_pack="TemplatePack")
            template = root / "template.json"
            template.write_text(json.dumps({"templateName": "Synthetic", "required": ["futureRequiredField"]}), encoding="utf-8")
            project_out = root / "CliProject"

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = converter.run_cli(
                    [
                        "--mod-archive",
                        str(archive),
                        "--target",
                        "ostim-tools-json",
                        "--ostim-tools-project-out",
                        str(project_out),
                        "--ostim-tools-template",
                        str(template),
                        "--ostim-tools-include-actions",
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            project = json.loads((project_out / converter.OSTIM_TOOLS_PROJECT_FILE).read_text(encoding="utf-8"))
            report = json.loads((project_out / "reports" / converter.OSTIM_TOOLS_REPORT_JSON).read_text(encoding="utf-8"))
            self.assertEqual(project["templateName"], "Synthetic")
            self.assertTrue(any("futureRequiredField" in warning for warning in project["warnings"]))
            self.assertGreater(report["actionsWritten"], 0)

    def test_ostim_tools_include_actions_off_writes_no_action_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "NoActions.zip"
            scene_data = {
                "name": "No Action Export",
                "modpack": "NoActions",
                "length": 4,
                "speeds": [{"animation": "NoActionExport"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            write_ostim_scene_pack(archive, scene_data, {"NoActionExport_0": b"hkx"}, scene_name="NoActionExport", scene_pack="NoActions")
            out = root / "Project"
            options = converter.OStimToolsProjectOptions(include_actions=False)

            result = converter.convert_archive_to_ostim_tools_project(archive, project_out=out, options=options)

            self.assertEqual(result.report["actionsWritten"], 0)
            self.assertFalse((out / "actions").exists())

    def test_gui_layout_is_scrollable_and_resizable_for_small_screens(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ctk.CTkScrollableFrame", source)
        self.assertIn("root.resizable(True, True)", source)
        self.assertIn('text="Build From Advanced Options"', source)
        self.assertIn('values=["Simple Mode", "Advanced Mode"]', source)
        self.assertIn('text="Build Recommended Package"', source)
        self.assertIn('text="Verify Converted ZIP"', source)
        self.assertIn('text="Diagnose Archive"', source)
        self.assertIn('text="Human-only OStim output"', source)
        self.assertIn('text="Generate OStim Tools JSON"', source)
        self.assertIn('text="Generate Scene JSON Folder"', source)
        self.assertIn('text="Validate OStim Tools Project"', source)
        self.assertIn('text="Project Tools"', source)
        self.assertIn('text="Next Steps"', source)
        self.assertIn('text="Reports & Files"', source)
        self.assertIn("install_panel.grid(row=8", source)
        self.assertIn("simple_support_frame.grid(row=9", source)
        minsize_match = re.search(r"root\.minsize\((\d+),\s*(\d+)\)", source)
        self.assertIsNotNone(minsize_match)
        self.assertLessEqual(int(minsize_match.group(2)), 480)
        log_height_match = re.search(r"CTkTextbox\(frame,\s*height=(\d+)", source)
        self.assertIsNotNone(log_height_match)
        self.assertLessEqual(int(log_height_match.group(1)), 180)

    def test_simple_mode_settings_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_dir = Path(temp)
            with mock.patch.object(converter, "app_settings_dir", return_value=settings_dir):
                converter.save_app_settings(
                    {
                        "uiMode": "Advanced Mode",
                        "openOutputAfterBuild": False,
                        "humanOnlyOStimOutput": True,
                        "lastHumanReport": str(settings_dir / "report.txt"),
                        "lastJsonReport": str(settings_dir / "report.json"),
                        "ignoredKey": "nope",
                    }
                )
                settings = converter.load_app_settings()

            self.assertEqual(settings["uiMode"], "Advanced Mode")
            self.assertFalse(settings["openOutputAfterBuild"])
            self.assertTrue(settings["humanOnlyOStimOutput"])
            self.assertIn("lastHumanReport", settings)
            self.assertNotIn("ignoredKey", settings)

    def test_install_steps_generated_for_ostim_and_sexlab(self):
        ostim_steps = converter.install_steps_text(
            {
                "outputType": "OStim Standalone",
                "pack": "Demo Pack",
                "recommendedBehaviorTool": "Pandora",
            },
            zip_name="Demo_OStimSA.zip",
        )
        sexlab_steps = converter.install_steps_text(
            {
                "target": "SexLab/SLAL",
                "pack": "Demo Pack",
            },
            zip_name="Demo_SexLab.zip",
        )

        self.assertIn("Run Pandora", ostim_steps)
        self.assertIn("Open OStim", ostim_steps)
        self.assertIn("SexLab Animation Loader", sexlab_steps)
        self.assertIn("Register or enable", sexlab_steps)

    def test_failed_install_steps_do_not_look_installable(self):
        steps = converter.install_steps_text(
            {
                "outputType": "OStim Standalone",
                "pack": "Broken Pack",
                "status": "FAIL",
                "ok": False,
                "postBuildVerification": {"status": "FAIL"},
            },
            zip_name="FAILED_DoNotInstall_Broken_Pack_OStimSA.zip",
        )

        self.assertIn("Do not install this ZIP", steps)
        self.assertIn("Nexus-safe report bundle", steps)
        self.assertNotIn("Run Pandora", steps)
        self.assertNotIn("Open OStim", steps)

    def test_report_bundle_includes_diagnosis_placeholder_when_not_run(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle.zip"

            converter.create_nexus_safe_report_bundle(bundle, report={"pack": "Demo", "status": "PASS", "ok": True})

            with _ZipFile(bundle) as archive:
                diagnosis = json.loads(archive.read("diagnosis_summary_public.json").decode("utf-8"))

        self.assertEqual(diagnosis["status"], "NOT RUN")
        self.assertEqual(diagnosis["message"], "Diagnosis was not run.")

    def test_report_bundle_includes_supplied_diagnosis_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle.zip"

            converter.create_nexus_safe_report_bundle(
                bundle,
                report={"pack": "Demo", "status": "PASS", "ok": True},
                diagnosis={"status": "PASS WITH WARNINGS", "detectedSourceType": "SexLab/SLAL animation JSON"},
            )

            with _ZipFile(bundle) as archive:
                diagnosis = json.loads(archive.read("diagnosis_summary_public.json").decode("utf-8"))

        self.assertEqual(diagnosis["status"], "PASS WITH WARNINGS")
        self.assertEqual(diagnosis["detectedSourceType"], "SexLab/SLAL animation JSON")

    def test_recommended_target_defaults_to_ostim_unless_explicit(self):
        self.assertEqual(
            converter.recommended_target_from_diagnosis(
                {"ok": True, "recommendedOutput": "OStim Standalone or SexLab/SLAL", "detectedSourceTypeCode": "sexlabSlal"}
            ),
            "ostim",
        )
        self.assertEqual(
            converter.recommended_target_from_diagnosis({"ok": True, "recommendedOutput": "SexLab/SLAL"}),
            "sexlab",
        )
        self.assertEqual(
            converter.recommended_target_from_diagnosis(
                {"ok": True, "recommendedOutput": "OStim Standalone or SexLab P+", "detectedSourceTypeCode": "sexlabSceneBuilder"},
                sexlab_plus_enabled=True,
            ),
            "sexlabplus",
        )
        self.assertEqual(converter.recommended_target_from_diagnosis({"ok": False, "recommendedOutput": "unsupported"}), "unsupported")

    def test_recommended_workflow_text_is_beginner_readable(self):
        text = converter.recommended_workflow_panel_text(
            {
                "detectedSourceType": "SexLab/SLAL pack",
                "detectionConfidence": "High",
                "recommendedOutput": "OStim Standalone",
                "recommendedBehaviorTool": "Pandora",
                "creatureRuntimeNeeded": False,
                "furnitureScenesDetected": 2,
                "sceneOrAnimationRecordsFound": 12,
                "hkxFilesFound": 24,
                "missingReferencedHkxCount": 0,
            }
        )

        self.assertIn("Detected: SexLab/SLAL pack", text)
        self.assertIn("Recommended output: OStim Standalone", text)
        self.assertIn("Recommended next step:", text)

    def test_archive_structure_export_contains_paths_only(self):
        report = converter.archive_structure_from_paths(
            [
                "Data/SLAnims/json/Demo.json",
                "Data/SLAnims/source/Demo.txt",
                "Data/meshes/actors/character/animations/Demo/Event.hkx",
                "Data/Nemesis_Engine/mod/demo/info.ini",
            ],
            archive_name="Demo.zip",
        )

        self.assertEqual(report["archiveName"], "Demo.zip")
        self.assertIn("SLAnims/json", report["detectedKeyFolders"])
        self.assertIn("meshes/actors", report["detectedKeyFolders"])
        self.assertIn("Data/SLAnims/json/Demo.json", report["paths"])
        self.assertNotIn(str(ROOT), json.dumps(report))

    def test_nexus_safe_bundle_excludes_assets_and_local_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = converter.create_nexus_safe_report_bundle(
                root / "AAC_BugReport_Demo.zip",
                report={"zipPath": r"C:\Users\josep\Desktop\Demo_OStimSA.zip", "outputType": "OStim Standalone"},
                verification=None,
                archive_structure=converter.archive_structure_from_paths(["Data/meshes/actors/character/animations/Demo/Event.hkx"]),
                selected_settings={"debugMode": False},
                log_text=r"ZIP: C:\Users\josep\Desktop\Demo_OStimSA.zip",
                debug_mode=False,
            )

            with _ZipFile(bundle) as archive:
                names = archive.namelist()
                payload = "\n".join(archive.read(name).decode("utf-8") for name in names if name.endswith((".json", ".txt")))

            self.assertFalse(any(name.lower().endswith(".hkx") for name in names))
            self.assertNotIn(r"C:\Users\josep", payload)
            self.assertIn("Demo_OStimSA.zip", payload)

    def test_public_log_text_strips_full_paths_in_normal_mode(self):
        text = converter.public_safe_log_text(r"Report: C:\Users\josep\AppData\Local\Temp\conversion_report.txt", debug_mode=False)

        self.assertIn("conversion_report.txt", text)
        self.assertNotIn(r"C:\Users\josep", text)

    def test_source_diagnosis_report_file_redacts_unsafe_path_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "UnsafePath.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(r"C:\Users\Tester\Desktop\UnsafePath\Anim.hkx", b"")

            result = converter.diagnose_source_archive(archive, report_dir=root)
            self.assertFalse(result.report["ok"])
            self.assertTrue(result.report_path)
            report_text = result.report_path.read_text(encoding="utf-8")
            text_report = result.text_report_path.read_text(encoding="utf-8")
            self.assertNotIn(r"C:\Users\Tester", report_text)
            self.assertNotIn(r"C:\Users\Tester", text_report)
            self.assertIn("Anim.hkx", report_text)

    def test_creature_runtime_setup_check_reports_markers_and_missing_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "Runtime"

            def write(relative: str, data: bytes = b"") -> None:
                path = runtime / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            write("Data/SKSE/Plugins/OStim/scenes/index.json", b"{}")
            write("mods/OCreatures/Data/SKSE/Plugins/OStim/scenes/OCreatures/.keep")
            write("mods/Creature Framework/Data/CreatureFramework.esm")
            write("mods/MNC/Data/MoreNastyCritters.esp")
            write("tools/Pandora/Pandora Behaviour Engine.exe")
            write("Data/meshes/actors/canine/animations/Pack/Event.hkx")

            missing = converter.check_creature_runtime_setup(
                runtime,
                expected_roots=["canine", "dlc01/chaurusflyer"],
                report_dir=root / "reports",
            )
            self.assertFalse(missing.ok)
            self.assertEqual(missing.report["status"], "FAIL")
            self.assertIn("dlc01/chaurusflyer", missing.report["missingExpectedCreatureRoots"])
            self.assertTrue(any(check["id"] == "ostimCreatureExtension" and check["found"] for check in missing.report["checklist"]))
            self.assertTrue(missing.report_path)
            public_text = missing.report_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), public_text)
            self.assertIn("Event.hkx", public_text)

            write("Data/meshes/actors/dlc01/chaurusflyer/animations/Pack/FlyerEvent.hkx")
            ready = converter.check_creature_runtime_setup(
                runtime,
                expected_roots=["canine", "dlc01/chaurusflyer"],
                write_report=False,
            )
            self.assertTrue(ready.ok, ready.report["errors"])
            self.assertEqual(ready.report["status"], "PASS")
            self.assertEqual(ready.report["missingExpectedCreatureRoots"], [])

    def test_creature_runtime_setup_check_is_inconclusive_when_virtualized_markers_are_not_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "Data"
            animation = runtime / "meshes/actors/canine/animations/Pack/Event.hkx"
            animation.parent.mkdir(parents=True, exist_ok=True)
            animation.write_bytes(b"")

            result = converter.check_creature_runtime_setup(
                runtime,
                expected_roots=["canine"],
                write_report=False,
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.report["status"], "INCONCLUSIVE")
            self.assertEqual(result.report["errors"], [])
            self.assertIn("OStim Standalone", result.report["requiredMarkersNotDetected"])
            self.assertIn("does not prove", result.report["recommendedUserAction"])

    def test_creature_runtime_text_accepts_string_checklist_entries(self):
        report = {
            "status": "PASS WITH WARNINGS",
            "targetPath": "Runtime",
            "filesScanned": 0,
            "checklist": [
                "Legacy checklist note from external report.",
                {"label": "OStim scenes folder", "found": True, "required": True, "examples": ["SKSE/Plugins/OStim/scenes/index.json"]},
            ],
            "warnings": [],
            "errors": [],
        }

        text = converter.creature_runtime_check_report_to_text(report)

        self.assertIn("[INFO] Legacy checklist note from external report.", text)
        self.assertIn("[PASS] OStim scenes folder", text)

    def test_failed_verification_renames_zip_do_not_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "Broken_OStimSA.zip"
            with _ZipFile(bad_zip, "w") as archive:
                archive.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/Broken/Broken_Scene.json",
                    json.dumps(
                        {
                            "name": "Broken",
                            "modpack": "Broken",
                            "speeds": [{"animation": "MissingEvent"}],
                            "actors": [{"animationIndex": 0}],
                            "actions": [{"type": "kissing", "actor": 0}],
                        }
                    ),
                )

            verification = converter.verify_converted_zip(bad_zip)
            failed_zip = converter.mark_failed_zip_do_not_install(verification)

            self.assertFalse(verification.ok)
            self.assertFalse(bad_zip.exists())
            self.assertTrue(failed_zip.name.startswith(converter.FAILED_ZIP_PREFIX))
            self.assertTrue(verification.report_path.exists())
            self.assertIn("Do not install", "\n".join(verification.report["warnings"]))

    def test_stale_aac_manifest_warning_triggers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stale_zip = root / "Stale.zip"
            with _ZipFile(stale_zip, "w") as archive:
                archive.writestr(
                    converter.AAC_MANIFEST_FILE,
                    json.dumps(
                        {
                            "schema": converter.AAC_MANIFEST_SCHEMA,
                            "converterVersion": "2024.old",
                            "outputType": "OStim Standalone",
                        }
                    ),
                )

            verification = converter.verify_converted_zip(stale_zip, write_report=False)

            self.assertTrue(any("generated by converter 2024.old" in warning for warning in verification.report["warnings"]))

    def test_generated_zip_contains_aac_readme_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "AAC Meta.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Meta.xml",
                    """\
<scene id="AA|Standing|Meta" actors="1">
  <info name="Meta" />
  <anim id="MetaLoop" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/MetaLoop_0.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)

            with ZipFile(result.zip_path) as converted:
                self.assertIn(converter.AAC_README_FILE, converted.namelist())
                self.assertIn(converter.AAC_MANIFEST_FILE, converted.namelist())
                manifest = json.loads(converted.read(converter.AAC_MANIFEST_FILE).decode("utf-8"))
                readme = converted.read(converter.AAC_README_FILE).decode("utf-8")

            self.assertEqual(manifest["converterVersion"], converter.CONVERTER_VERSION)
            self.assertEqual(manifest["outputType"], "OStim Standalone")
            self.assertIn("Install Steps", readme)

    def test_console_progress_output_is_optional_for_windowed_exe(self):
        class BrokenStdout:
            def write(self, _value):
                raise OSError(22, "Invalid argument")

            def flush(self):
                raise OSError(22, "Invalid argument")

        converter.print("[ok] windowed progress", file=BrokenStdout())

    def test_cli_missing_verify_zip_returns_error_instead_of_throwing(self):
        with tempfile.TemporaryDirectory() as temp:
            missing_zip = Path(temp) / "missing.zip"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--verify-zip", str(missing_zip)])

            self.assertEqual(exit_code, 1)
            self.assertIn("ZIP not found", stdout.getvalue())

    def test_find_7z_executable_prefers_path_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_7z = Path(temp) / "7z.exe"
            fake_7z.write_text("fake", encoding="utf-8")

            def fake_which(name):
                return str(fake_7z) if name == "7z.exe" else None

            with mock.patch.object(converter.shutil, "which", side_effect=fake_which):
                self.assertEqual(Path(converter.find_7z_executable()).resolve(), fake_7z.resolve())

    def test_find_7z_executable_finds_portable_copy_without_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_7z = root / "7-Zip" / "7z.exe"
            fake_7z.parent.mkdir()
            fake_7z.write_text("fake", encoding="utf-8")

            with mock.patch.object(converter.shutil, "which", return_value=None), mock.patch.object(
                converter, "seven_zip_registry_candidates", return_value=[]
            ), mock.patch.dict(converter.os.environ, {}, clear=True):
                self.assertEqual(
                    Path(converter.find_7z_executable(extra_search_roots=[root])).resolve(),
                    fake_7z.resolve(),
                )

    def test_find_7z_executable_uses_env_override(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_7z = Path(temp) / "portable7z.exe"
            fake_7z.write_text("fake", encoding="utf-8")

            with mock.patch.object(converter.shutil, "which", return_value=None), mock.patch.object(
                converter, "seven_zip_registry_candidates", return_value=[]
            ), mock.patch.dict(converter.os.environ, {"AAC_7ZIP": str(fake_7z)}, clear=True):
                self.assertEqual(Path(converter.find_7z_executable()).resolve(), fake_7z.resolve())

    def test_hkx_header_inspection_distinguishes_le_se_and_other_havok(self):
        self.assertEqual(converter.inspect_hkx_header(hkx_packfile_bytes(4))["format"], converter.HKX_PLATFORM_SKYRIM_LE)
        self.assertEqual(converter.inspect_hkx_header(hkx_packfile_bytes(8))["format"], converter.HKX_PLATFORM_SKYRIM_SE)
        self.assertEqual(converter.inspect_hkx_header(hkx_packfile_bytes(8, file_version=11))["format"], converter.HKX_PLATFORM_OTHER_64)
        self.assertEqual(converter.inspect_hkx_header(b"not an hkx")["format"], converter.HKX_PLATFORM_UNKNOWN)

    def test_find_legacy_hkx_converter_accepts_configured_helper_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "data" / "hkx32to64.exe"
            helper.parent.mkdir()
            helper.write_bytes(b"helper")

            found = converter.find_legacy_hkx_converter(root)

            self.assertIsNotNone(found)
            self.assertEqual(found.executable.resolve(), helper.resolve())
            self.assertEqual(found.kind, "hkx32to64")
            self.assertEqual(found.source, "configured")

    def test_legacy_hkx_conversion_converts_only_confirmed_le_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "hkx32to64.exe"
            helper.write_bytes(b"helper")
            le_source = root / "Legacy_A1.hkx"
            se_source = root / "Modern_A2.hkx"
            le_bytes = hkx_packfile_bytes(4, payload=b"legacy")
            se_bytes = hkx_packfile_bytes(8, payload=b"modern")
            le_source.write_bytes(le_bytes)
            se_source.write_bytes(se_bytes)
            assets = [
                converter.HkxAsset(le_source, PurePosixPath("meshes/actors/character/animations/Pack/Legacy_A1.hkx"), PurePosixPath("meshes/actors/character/animations/Pack/Legacy_A1.hkx"), "Legacy_A1"),
                converter.HkxAsset(se_source, PurePosixPath("meshes/actors/character/animations/Pack/Modern_A2.hkx"), PurePosixPath("meshes/actors/character/animations/Pack/Modern_A2.hkx"), "Modern_A2"),
            ]

            def fake_run(command, **kwargs):
                working_dir = Path(kwargs["cwd"])
                self.assertEqual(command[0], str(helper.resolve()))
                self.assertEqual(command[1], "input.hkx")
                (working_dir / "OUTFILE64.hkx").write_bytes(hkx_packfile_bytes(8, payload=b"converted"))
                return converter.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch.object(converter.subprocess, "run", side_effect=fake_run):
                converted, warnings, summary = converter.convert_legacy_hkx_assets_for_sse(
                    assets,
                    root / "output",
                    configured_converter=helper,
                )

            self.assertEqual(warnings, [])
            self.assertEqual(summary["result"], "PASS")
            self.assertEqual(summary["legacyLeHkxCount"], 1)
            self.assertEqual(summary["alreadySeHkxCount"], 1)
            self.assertEqual(summary["convertedHkxCount"], 1)
            self.assertEqual(converter.inspect_hkx_file(converted[0].source)["format"], converter.HKX_PLATFORM_SKYRIM_SE)
            self.assertEqual(converted[1].source, se_source)
            self.assertEqual(le_source.read_bytes(), le_bytes)
            self.assertEqual(se_source.read_bytes(), se_bytes)

    def test_creation_kit_hkx_converter_uses_platformamd64_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "HavokBehaviorPostProcess.exe"
            helper.write_bytes(b"helper")
            source = root / "Legacy.hkx"
            source.write_bytes(hkx_packfile_bytes(4))
            adapter = converter.LegacyHkxConverter(helper.resolve(), "havok_behavior_post_process", "configured")

            def fake_run(command, **kwargs):
                self.assertEqual(command[0], str(helper.resolve()))
                self.assertEqual(command[1], "--platformamd64")
                self.assertEqual(Path(command[2]).name, "input.hkx")
                self.assertEqual(Path(command[3]).name, "OUTFILE64.hkx")
                Path(command[3]).write_bytes(hkx_packfile_bytes(8, payload=b"converted"))
                return converter.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch.object(converter.subprocess, "run", side_effect=fake_run):
                output = converter.run_legacy_hkx_converter(source, adapter, root / "work")

            self.assertEqual(converter.inspect_hkx_file(output)["format"], converter.HKX_PLATFORM_SKYRIM_SE)

    def test_legacy_hkx_conversion_stops_when_helper_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Legacy.hkx"
            source.write_bytes(hkx_packfile_bytes(4))
            asset = converter.HkxAsset(
                source,
                PurePosixPath("meshes/actors/character/animations/Pack/Legacy.hkx"),
                PurePosixPath("meshes/actors/character/animations/Pack/Legacy.hkx"),
                "Legacy",
            )
            with mock.patch.object(converter, "find_legacy_hkx_converter", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "no LE-to-SE HKX converter helper was found"):
                    converter.convert_legacy_hkx_assets_for_sse([asset], root / "output")

    def test_legacy_fnis_archive_build_converts_hkx_and_keeps_source_archive_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Legacy FNIS Pack.zip"
            helper = root / "hkx32to64.exe"
            helper.write_bytes(b"helper")
            legacy_bytes = hkx_packfile_bytes(4, payload=b"legacy-source")
            fnis_list = "\n".join(
                [
                    "Version V1.0",
                    "b -Tn LegacyPose_A1_S1 LegacyPose_A1_S1.hkx",
                    "b -Tn LegacyPose_A2_S1 LegacyPose_A2_S1.hkx",
                    "",
                ]
            )
            folder = "Data/meshes/actors/character/animations/LegacyPack"
            with _ZipFile(archive, "w") as source:
                source.writestr(f"{folder}/FNIS_LegacyPack_List.txt", fnis_list)
                source.writestr(f"{folder}/LegacyPose_A1_S1.hkx", legacy_bytes)
                source.writestr(f"{folder}/LegacyPose_A2_S1.hkx", legacy_bytes)

            def fake_run(command, **kwargs):
                working_dir = Path(kwargs["cwd"])
                (working_dir / "OUTFILE64.hkx").write_bytes(hkx_packfile_bytes(8, payload=b"converted"))
                return converter.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch.object(converter.subprocess, "run", side_effect=fake_run):
                result = converter.convert_archive_to_ready_zip(
                    archive,
                    legacy_hkx_converter=helper,
                )

            self.assertTrue(result.verification.ok, result.verification.report["errors"])
            self.assertEqual(result.report["legacyHkxConversion"]["legacyLeHkxCount"], 2)
            self.assertEqual(result.report["legacyHkxConversion"]["convertedHkxCount"], 2)
            self.assertEqual(result.verification.report["hkxPlatformVerification"]["legacyLeHkxCount"], 0)
            self.assertEqual(result.verification.report["hkxPlatformVerification"]["alreadySeHkxCount"], 2)

            with _ZipFile(result.zip_path, "r") as built:
                animation_entries = [
                    name
                    for name in built.namelist()
                    if name.lower().endswith(".hkx") and "/animations/" in name.lower()
                ]
                self.assertEqual(len(animation_entries), 2)
                for name in animation_entries:
                    self.assertEqual(
                        converter.inspect_hkx_header(built.read(name))["format"],
                        converter.HKX_PLATFORM_SKYRIM_SE,
                    )
            with _ZipFile(archive, "r") as original:
                self.assertEqual(original.read(f"{folder}/LegacyPose_A1_S1.hkx"), legacy_bytes)
                self.assertEqual(original.read(f"{folder}/LegacyPose_A2_S1.hkx"), legacy_bytes)

    def test_deployment_verifier_rejects_packaged_legacy_le_hkx(self):
        with tempfile.TemporaryDirectory() as temp:
            zip_path = Path(temp) / "LegacyOutput.zip"
            with _ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "SKSE/Plugins/OStim/scenes/Test/TestScene.json",
                    json.dumps(
                        {
                            "name": "Test Scene",
                            "modpack": "Test",
                            "actors": [{"intendedSex": "female"}],
                            "speeds": [{"animation": "LegacyEvent"}],
                        }
                    ),
                )
                archive.writestr(
                    "meshes/actors/character/animations/Test/LegacyEvent_0.hkx",
                    hkx_packfile_bytes(4),
                )

            result = converter.verify_ostim_converted_zip(zip_path, write_report=False)

            self.assertFalse(result.ok)
            self.assertEqual(result.report["hkxPlatformVerification"]["legacyLeHkxCount"], 1)
            self.assertTrue(any("Skyrim LE 32-bit" in error for error in result.report["errors"]))

    def test_unsafe_zip_member_fails_cleanly_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Unsafe.zip"
            output_zip = root / "Unsafe_OStimSA.zip"
            escaped_name = f"{root.name}_escaped.txt"
            with _ZipFile(archive, "w") as source:
                source.writestr(f"../{escaped_name}", "bad")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--mod-archive", str(archive), "--zip-out", str(output_zip)])

            failure_report = output_zip.with_name("Unsafe_OStimSA_conversion_failed.json")
            self.assertEqual(exit_code, 1)
            self.assertFalse((root.parent / escaped_name).exists())
            self.assertTrue(failure_report.exists())
            report = json.loads(failure_report.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["sourceArchiveName"], "Unsafe.zip")
            self.assertTrue(any("unsafe path" in error for error in report["fatalErrors"]))
            self.assertIn("unsafe path", stdout.getvalue())

    def test_7z_listing_archive_header_absolute_path_is_not_treated_as_member(self):
        listing = "\n".join(
            [
                r"Path = C:\Users\hanra\OneDrive\�rea de Trabalho\Adult Animation Converter\Nova pasta\SLAL FlufyFox_SLAL_SE_Mid 1.0.rar",
                "Type = Rar",
                "Physical Size = 1339933",
                "----------",
                "Path = Data/meshes/actors/character/animations/FlufyFox/Scene_A1_S1.hkx",
                "Size = 12",
                "Packed Size = 8",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "SLAL FlufyFox_SLAL_SE_Mid 1.0.rar"
            archive.write_bytes(b"rar")
            with mock.patch.object(converter.subprocess, "check_output", return_value=listing):
                converter.validate_7z_archive_members(archive, "7z.exe")

    def test_7z_listing_drive_qualified_archive_member_still_fails(self):
        listing = "\n".join(
            [
                r"Path = C:\Users\hanra\Downloads\Bad.rar",
                "Type = Rar",
                "Physical Size = 64",
                "----------",
                r"Path = C:\Users\hanra\Desktop\evil.hkx",
                "Size = 12",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "Bad.rar"
            archive.write_bytes(b"rar")
            with mock.patch.object(converter.subprocess, "check_output", return_value=listing):
                with self.assertRaisesRegex(RuntimeError, "drive-qualified paths are not allowed"):
                    converter.validate_7z_archive_members(archive, "7z.exe")

    def test_7z_archive_structure_report_skips_archive_header_path(self):
        listing = "\n".join(
            [
                r"Path = C:\Users\hanra\Downloads\Wrapped.rar",
                "Type = Rar",
                "Physical Size = 64",
                "----------",
                "Path = Data/SLAnims/json/Pack.json",
                "Size = 12",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "Wrapped.rar"
            archive.write_bytes(b"rar")
            with mock.patch.object(converter, "find_7z_executable", return_value="7z.exe"), mock.patch.object(
                converter.subprocess, "check_output", return_value=listing
            ):
                report = converter.archive_structure_report_for_archive(archive)

        self.assertEqual(report["paths"], ["Data/SLAnims/json/Pack.json"])
        self.assertEqual(report["fileCount"], 1)

    def test_malformed_archive_fails_cleanly_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Malformed.zip"
            output_zip = root / "Malformed_OStimSA.zip"
            archive.write_bytes(b"this is not a zip archive")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--mod-archive", str(archive), "--zip-out", str(output_zip)])

            failure_report = output_zip.with_name("Malformed_OStimSA_conversion_failed.json")
            failure_text = output_zip.with_name("Malformed_OStimSA_conversion_failed.txt")
            self.assertEqual(exit_code, 1)
            self.assertTrue(failure_report.exists())
            self.assertTrue(failure_text.exists())
            report = json.loads(failure_report.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("BadZipFile", report["fatalErrors"][0])
            self.assertNotIn("Traceback", stdout.getvalue())

    def test_verify_reports_pass_with_warnings_when_structurally_usable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "WarnOnly.zip"
            scene = {
                "name": "Warning Scene",
                "modpack": "WarnOnly",
                "length": 3,
                "speeds": [{"animation": "WarnEvent"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with _ZipFile(zip_path, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/WarnOnly/WarnOnly_Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/WarnOnly/WarnEvent_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/WarnOnly/ATT_warnpatch_animlist.txt",
                    "b -Tn WarnEvent_0 WarnEvent_0.hkx WARNPATCH_AnimationSpeed: 1.0\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/warnpatch/info.ini",
                    "name=WarnOnly\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/warnpatch/0_master/#0106.txt",
                    "<hkobject><hkcstring>WarnEvent_0</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["status"], "PASS WITH WARNINGS")
            self.assertGreater(verification.report["warningCount"], 0)
            self.assertIn("warningSeverityGroups", verification.report)

    def test_sfx_fallback_dropdown_options_map_to_valid_actions(self):
        for label, action_type in converter.SFX_FALLBACK_DROPDOWN_OPTIONS:
            self.assertEqual(converter.sfx_fallback_action_from_dropdown(label), action_type)
            converter.normalize_sfx_fallback_action_type(action_type)

        self.assertEqual(converter.sfx_fallback_action_from_dropdown("Kissing"), "kissing")
        self.assertEqual(converter.sfx_fallback_action_from_dropdown("kissing"), "kissing")

    def test_ostim_menu_hub_allows_known_external_origin(self):
        scene = converter.Scene(
            raw_id="MenuTest",
            scene_id="TestPack_MenuTest",
            name="Menu Test",
            speeds=[converter.Speed("MenuEvent")],
            actors=[
                converter.Actor(intended_sex="male", animation_index=0),
                converter.Actor(intended_sex="female", animation_index=1),
            ],
            actions=[converter.Action(type="kissing", actor=0, target=1)],
        )

        scenes, warnings = converter.add_ostim_menu_hubs(
            [scene],
            "TestPack",
            enabled=True,
            icon="Data/Interface/OStim/icons/OStim/symbols/search.dds",
        )

        self.assertEqual(warnings, [])
        hubs = [candidate for candidate in scenes if converter.is_ostim_menu_hub_scene(candidate)]
        self.assertEqual(len(hubs), 1)
        hub = hubs[0]
        self.assertEqual([speed.animation for speed in hub.speeds], ["OStim2PStandingApartMF"])
        self.assertEqual(converter.expected_animation_events(hub), [])
        origin_navs = [nav for nav in hub.navigations if nav.origin]
        return_navs = [nav for nav in hub.navigations if nav.destination == "OStim2PStandingApartMF"]
        self.assertEqual([nav.origin for nav in origin_navs], ["OStim2PStandingApartMF"])
        self.assertEqual(origin_navs[0].icon, "OStim/symbols/search")
        self.assertTrue(origin_navs[0].no_warnings)
        self.assertEqual(len(return_navs), 1)
        self.assertEqual(return_navs[0].priority, -1000)
        self.assertTrue(return_navs[0].no_warnings)
        self.assertEqual(converter.missing_scene_reference_rows(scenes), [])

        report = converter.build_conversion_report(scenes, [], "TestPack")
        self.assertEqual(report["ostimMenuHubSceneCount"], 1)
        self.assertEqual(report["externalOStimMenuOriginCount"], 1)
        self.assertEqual(report["missingSceneLinkCount"], 0)

    def test_report_accepts_pandora_compatible_output_for_pandora_profile(self):
        scene = converter.Scene(
            raw_id="LegacyMismatch",
            scene_id="LegacyMismatch_Scene",
            name="Legacy Mismatch",
            speeds=[converter.Speed("MismatchEvent")],
            actors=[
                converter.Actor(intended_sex="male", animation_index=0),
                converter.Actor(intended_sex="female", animation_index=1),
            ],
            actions=[converter.Action(type="kissing", actor=0, target=1)],
        )
        asset = converter.HkxAsset(
            source=Path("MismatchEvent.hkx"),
            data_path=PurePosixPath("meshes/actors/character/animations/LegacyMismatch/MismatchEvent.hkx"),
            packaged_path=PurePosixPath("meshes/actors/character/animations/LegacyMismatch/MismatchEvent.hkx"),
            event_name="MismatchEvent",
        )

        report = converter.build_conversion_report(
            [scene],
            [asset],
            "LegacyMismatch",
            source_engine_patch_file_count=1,
            att_animation_list_file_count=1,
            behavior_output_mode=converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE,
            source_detection={
                "compatibilityMatch": {
                    "matchResult": "Exact",
                    "matchedPack": "Pandora Preferred Pack",
                    "recommendedBehaviorTool": "Pandora creature-capable output",
                    "knownWarnings": [],
                }
            },
        )

        self.assertFalse(any("does not match the known pack recommendation" in warning for warning in report["warnings"]))
        self.assertEqual(converter.recommended_behavior_tool_from_report(report), "Pandora")

    def test_generated_ostim_navigation_adds_return_and_sequence_links(self):
        scenes = []
        for index in range(3):
            scenes.append(
                converter.Scene(
                    raw_id=f"NavTest{index + 1}",
                    scene_id=f"NavPack_Scene_{index + 1}",
                    name=f"Nav Scene {index + 1}",
                    speeds=[converter.Speed(f"NavEvent_{index + 1}")],
                    actors=[
                        converter.Actor(intended_sex="male", animation_index=0),
                        converter.Actor(intended_sex="female", animation_index=1),
                    ],
                    actions=[converter.Action(type="vaginalsex", actor=0, target=1)],
                )
            )

        scenes, warnings = converter.add_ostim_menu_hubs(
            scenes,
            "NavPack",
            enabled=True,
            icon="OStim/symbols/search",
        )
        self.assertEqual(warnings, [])
        warnings = converter.add_generated_ostim_navigation_links(
            scenes,
            enabled=True,
            icon="OStim/symbols/search",
        )
        self.assertEqual(warnings, [])

        hub = next(scene for scene in scenes if converter.is_ostim_menu_hub_scene(scene))
        return_count, sequence_count = converter.generated_ostim_navigation_counts(scenes)
        self.assertEqual(return_count, 3)
        self.assertEqual(sequence_count, 6)
        self.assertEqual(converter.missing_scene_reference_rows(scenes), [])

        first_scene = next(scene for scene in scenes if scene.scene_id == "NavPack_Scene_1")
        first_json = first_scene.to_json("NavPack")
        self.assertNotIn("destination", first_json)
        first_navs = first_json["navigations"]
        self.assertIn(
            {
                "destination": hub.scene_id,
                "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 20,
                "description": converter.OSTIM_MENU_RETURN_DESCRIPTION,
                "icon": "OStim/symbols/search",
            },
            first_navs,
        )
        self.assertIn(
            {
                "destination": "NavPack_Scene_3",
                "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 30,
                "description": converter.OSTIM_MENU_PREVIOUS_DESCRIPTION,
                "icon": "OStim/symbols/search",
            },
            first_navs,
        )
        self.assertIn(
            {
                "destination": "NavPack_Scene_2",
                "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 31,
                "description": converter.OSTIM_MENU_NEXT_DESCRIPTION,
                "icon": "OStim/symbols/search",
            },
            first_navs,
        )

        report = converter.build_conversion_report(scenes, [], "NavPack")
        self.assertEqual(report["generatedOStimReturnNavigationCount"], 3)
        self.assertEqual(report["generatedOStimSequenceNavigationCount"], 6)

    def test_sexlab_scene_builder_repo_detection_requires_expected_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "SexLab-Scene-Builder"
            for marker in converter.SLSB_REPO_MARKER_FILES:
                path = repo / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("marker", encoding="utf-8")

            self.assertTrue(converter.is_sexlab_scene_builder_repo(repo))
            self.assertEqual(converter.detect_sexlab_scene_builder_repo([repo]), repo.resolve())

            (repo / "src-tauri" / "src" / "project" / "serialize.rs").unlink()
            self.assertFalse(converter.is_sexlab_scene_builder_repo(repo))

    def test_pandora_project_names_use_vanilla_actor_roots(self):
        cases = {
            "meshes/actors/dragonpriest/animations/Pack/Foo.hkx": ["dragon_priest"],
            "meshes/actors/spriggan/animations/Pack/Foo.hkx": ["spriggan"],
            "meshes/actors/vampirelord/animations/Pack/Foo.hkx": ["vampirelord"],
            "meshes/actors/dlc01/chaurusflyer/animations/Pack/Foo.hkx": ["chaurusflyer"],
            "meshes/actors/dlc02/boarriekling/animations/Pack/Foo.hkx": ["boarproject"],
            "meshes/actors/canine/animations/Pack/Foo.hkx": ["dogproject", "wolfproject"],
        }
        for path, projects in cases.items():
            self.assertEqual(converter.infer_projects_from_path(PurePosixPath(path)), projects)

    def test_sexlab_creature_warning_keeps_dlc_actor_roots(self):
        rewritten = converter.sexlab_export_conversion_warnings(
            [
                "Source HKX files were found under non-character actor root(s): canine, dlc01/chaurusflyer. "
                "Creature animation roots are packaged for OCreatures-style support; base OStim Standalone still needs a creature extension."
            ]
        )

        self.assertIn("canine, dlc01/chaurusflyer", rewritten[0])

    def test_legacy_scene_writes_ostim_sa_json_and_package_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_dir = root / "Data" / "Meshes" / "0SA" / "mod" / "0Sex" / "scene" / "BB" / "Sy6" / "HhPo"
            anim_dir = root / "Data" / "meshes" / "actors" / "character" / "animations" / "0Sex" / "BB" / "Sy6" / "HhPo"
            scene_dir.mkdir(parents=True)
            anim_dir.mkdir(parents=True)
            (scene_dir / "MoShoPo.xml").write_text(
                """\
<scene id="BB|Sy6!KNy9|HhPo|MoShoPo" actors="2">
  <info name="Money Shot" />
  <anim id="0SxBB_HhPo-MoShoPo_S0" l="7.5" />
  <speed max="4">
    <sp qnt="1"><anim id="0SxBB_HhPo-MoShoPo_S1" /></sp>
  </speed>
  <actors>
    <actor position="0" tags="standing" sosBend="7" />
    <actor position="1" tags="kneeling openmouth" />
  </actors>
</scene>
""",
                encoding="utf-8",
            )
            (anim_dir / "0SxBB_HhPo-MoShoPo_S1_0.hkx").write_bytes(b"")
            (anim_dir / "0SxBB_HhPo-MoShoPo_S1_1.hkx").write_bytes(b"")

            output = root / "out"
            result = converter.run_conversion(scene_dir, output, "TestPack")
            json_files = list((output / "TestPack").glob("*.json"))
            self.assertEqual(len(json_files), 1)
            self.assertNotIn("|", json_files[0].name)

            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(data["modpack"], "TestPack")
            self.assertEqual(data["speeds"][0]["animation"], "0SxBB_HhPo-MoShoPo_S1")
            self.assertEqual(len(data["actors"]), 2)
            self.assertNotIn("poses", data)
            self.assertNotIn("clips", data)

            zip_path = root / "converted.zip"
            converter.make_ready_to_install_zip(
                zip_path,
                output,
                "TestPack",
                hkx_assets=result.hkx_assets,
                scenes=result.scenes,
                mod_author="Ceo",
            )
            with _ZipFile(zip_path) as raw_archive:
                raw_names = set(raw_archive.namelist())
            with ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                assert_pandora_compatible_registration(self, names, "TestPack")
                att_list_path = single_character_att_list_path(names, "TestPack")
                att_list = archive.read(mod_entry(att_list_path)).decode("utf-8")
            self.assertFalse(any(name.startswith("Data/") for name in raw_names))
            self.assertNotIn(mod_entry("Data/animationsetdatasinglefile/DefaultMale.txt"), names)
            self.assertNotIn(mod_entry("Data/animationsetdatasinglefile/DefaultMale/H2HDual.txt"), names)
            self.assertNotIn(mod_entry("Data/animdata/DefaultMale.txt"), names)
            self.assertFalse(any(name.startswith(mod_entry("Data/animdata/")) for name in names))
            self.assertFalse(any(name.startswith(mod_entry("Data/animationsetdatasinglefile/")) for name in names))
            nemesis_code = converter.safe_nemesis_patch_code("TestPack")
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini"), names)
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/0_master/#0106.txt"), names)
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/defaultmale/#0029.txt"), names)
            self.assertNotIn(mod_entry("Data/meshes/actors/character/animations/TestPack/FNIS_TestPack_List.txt"), names)
            self.assertEqual(len(character_att_list_paths(names, "TestPack")), 1)
            self.assertIn(mod_entry("Data/SKSE/Plugins/OStim/converter_metadata/TestPack/metadata.json"), names)
            self.assertIn("conversion_report.json", names)
            self.assertIn("0SxBB_HhPo-MoShoPo_S1_1", att_list)
            self.assertIn("0Sex\\BB\\Sy6\\HhPo\\0SxBB_HhPo-MoShoPo_S1_1.hkx", att_list)
            with ZipFile(zip_path) as archive:
                metadata = json.loads(
                    archive.read(mod_entry("Data/SKSE/Plugins/OStim/converter_metadata/TestPack/metadata.json")).decode("utf-8")
                )
            self.assertEqual(metadata["schema"], converter.CONVERTER_METADATA_SCHEMA)
            self.assertEqual(metadata["pack"]["name"], "TestPack")
            self.assertEqual(metadata["pack"]["author"], "Ceo")
            self.assertEqual(metadata["deployment"]["status"], "PASS")
            self.assertEqual(metadata["deployment"]["sceneCount"], 1)
            self.assertEqual(metadata["deployment"]["pandoraAnimDataFileCount"], 0)
            self.assertEqual(metadata["deployment"]["pandoraAnimSetFileCount"], 0)
            self.assertEqual(metadata["deployment"]["pandoraNamedAnimDataFileCount"], 0)
            self.assertEqual(metadata["deployment"]["pandoraNamedAnimSetFileCount"], 0)
            self.assertGreater(metadata["deployment"]["generatedEnginePatchFileCount"], 0)
            self.assertTrue(metadata["checks"]["hasGeneratedNemesisBehaviorPatch"])
            self.assertTrue(metadata["checks"]["hasPandoraCompatibleBehavior"])
            self.assertFalse(metadata["checks"]["hasPandoraFiles"])
            self.assertFalse(metadata["checks"]["hasPandoraAnimData"])
            self.assertEqual(metadata["paths"]["behaviorList"], att_list_path)
            self.assertIsNone(metadata["paths"]["pandoraAnimDataRoot"])
            self.assertIsNone(metadata["paths"]["pandoraAnimSetRoot"])
            self.assertIsNone(metadata["paths"]["pandoraInfo"])
            self.assertIsNone(metadata["paths"]["pandoraNamedAnimDataRoot"])
            self.assertIsNone(metadata["paths"]["pandoraNamedAnimSetRoot"])
            self.assertEqual(
                metadata["paths"]["nemesisInfo"],
                f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini",
            )
            self.assertIsNone(metadata["paths"]["nemesisNamedAnimDataRoot"])

            legacy_zip_path = root / "legacy.zip"
            converter.make_ready_to_install_zip(
                legacy_zip_path,
                output,
                "TestPack",
                hkx_assets=result.hkx_assets,
                scenes=result.scenes,
                mod_author="Ceo",
                nemesis_safe_output=True,
            )
            with ZipFile(legacy_zip_path) as archive:
                legacy_names = set(archive.namelist())
                att_list_path = single_character_att_list_path(legacy_names, "TestPack")
                att_list = archive.read(mod_entry(att_list_path)).decode("utf-8")
            self.assertFalse(any(name.startswith(mod_entry("Data/Pandora_Engine/")) for name in legacy_names))
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini"), legacy_names)
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/0_master/#0106.txt"), legacy_names)
            speed_var = att_speed_var_from_list_path(att_list_path)
            self.assertIn(f"b -Tn 0SxBB_HhPo-MoShoPo_S1_1 0Sex\\BB\\Sy6\\HhPo\\0SxBB_HhPo-MoShoPo_S1_1.hkx {speed_var}: 1", att_list)

    def test_generic_stage_counts_paired_actors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "xml"
            source.mkdir()
            (source / "demo.xml").write_text(
                """\
<Scene id="Demo" name="Demo Scene" actors="2">
  <Stage id="StageOne" duration="4">
    <Animation actorIndex="0" file="meshes\\actors\\character\\animations\\Demo\\DemoAnim_0.hkx" />
    <Animation actorIndex="1" file="meshes\\actors\\character\\animations\\Demo\\DemoAnim_1.hkx" />
  </Stage>
</Scene>
""",
                encoding="utf-8",
            )

            output = root / "out"
            converter.run_conversion(source, output, "GenericPack")
            data = json.loads(next((output / "GenericPack").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(data["speeds"][0]["animation"], "DemoAnim")
            self.assertEqual(len(data["actors"]), 2)

    def test_two_converted_packs_do_not_share_behavior_registration_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_paths = []

            for pack in ("FirstPack", "SecondPack"):
                source = root / f"{pack}_xml"
                source.mkdir()
                (source / "demo.xml").write_text(
                    f"""\
<Scene id="AA|Standing|HJ|{pack}" name="{pack}" actors="2">
  <Stage id="StageOne" duration="4">
    <Animation actorIndex="0" file="meshes\\actors\\character\\animations\\{pack}\\SharedEvent_0.hkx" />
    <Animation actorIndex="1" file="meshes\\actors\\character\\animations\\{pack}\\SharedEvent_1.hkx" />
  </Stage>
</Scene>
""",
                    encoding="utf-8",
                )
                anim_dir = source / "Data" / "meshes" / "actors" / "character" / "animations" / pack
                anim_dir.mkdir(parents=True)
                (anim_dir / "SharedEvent_0.hkx").write_bytes(b"")
                (anim_dir / "SharedEvent_1.hkx").write_bytes(b"")
                output = root / f"{pack}_out"
                result = converter.run_conversion(source, output, pack)
                zip_path = root / f"{pack}.zip"
                converter.make_ready_to_install_zip(zip_path, output, pack, hkx_assets=result.hkx_assets, scenes=result.scenes)
                zip_paths.append(zip_path)

            name_sets = []
            for zip_path in zip_paths:
                with ZipFile(zip_path) as archive:
                    names = set(archive.namelist())
                self.assertFalse(any(name in {"Data/animdata/DefaultMale.txt", "Data/animationsetdatasinglefile/DefaultMale.txt"} for name in names))
                self.assertFalse(any(name.startswith("Data/animdata/") for name in names))
                self.assertFalse(any(name.startswith("Data/animationsetdatasinglefile/") for name in names))
                assert_pandora_compatible_registration(self, names, zip_path.stem)
                nemesis_code = converter.safe_nemesis_patch_code(zip_path.stem)
                self.assertTrue(any(name.startswith(f"Data/Nemesis_Engine/mod/{nemesis_code}/") for name in names))
                self.assertEqual(len(character_att_list_paths(names, zip_path.stem)), 1)
                self.assertFalse(any(name.startswith("Data/Pandora_Engine/mod/") for name in names))
                name_sets.append(names)

            shared_generated_paths = {
                name
                for name in name_sets[0] & name_sets[1]
                if name not in {
                    "README_OStim_SA.txt",
                    "conversion_report.json",
                    "conversion_report.txt",
                    "conversion_report_README.txt",
                    converter.AAC_README_FILE,
                    converter.AAC_MANIFEST_FILE,
                    "Data/README_OStim_SA.txt",
                    "Data/conversion_report.json",
                    "Data/conversion_report.txt",
                    "Data/conversion_report_README.txt",
                    f"Data/{converter.AAC_README_FILE}",
                    f"Data/{converter.AAC_MANIFEST_FILE}",
                    "SKSE/Plugins/OStim/actions/ostimconvertermoan.json",
                    "Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json",
                }
            }
            self.assertFalse(shared_generated_paths)

    def test_actor_specific_generic_events_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Actor Specific.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/OSA/mod/OSex/scene/AA/Pair.xml",
                    """\
<Scene id="PairScene" name="Pair Scene" actors="2">
  <Stage id="PairStage" duration="5">
    <Animation actorIndex="0" event="PairEvent_0" />
    <Animation actorIndex="1" event="PairEvent_1" />
  </Stage>
</Scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/OSex/AA/PairEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/OSex/AA/PairEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertTrue(any("normalized animation event" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                scene_json = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Actor_Specific/Actor_Specific_PairStage.json"
                    ).decode("utf-8")
                )
            self.assertEqual([speed["animation"] for speed in scene_json["speeds"]], ["PairEvent"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_osa_sanitized_event_collisions_get_stable_unique_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "OSA Collision.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Space.xml",
                    """\
<scene id="AA|Standing|Ap|Space" actors="1">
  <info name="Space" />
  <anim id="Event A" l="3" />
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Underscore.xml",
                    """\
<scene id="AA|Standing|Ap|Underscore" actors="1">
  <info name="Underscore" />
  <anim id="Event_A" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/Event A_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/Event_A_0.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            mapping = result.report["osaEventMapping"]
            self.assertGreaterEqual(mapping["sanitizedCollisionCount"], 1)

            with ZipFile(result.zip_path) as converted:
                scene_names = [
                    name for name in converted.namelist()
                    if name.startswith("Data/SKSE/Plugins/OStim/scenes/OSA_Collision/") and name.endswith(".json")
                ]
                speeds = []
                hkx_names = [name for name in converted.namelist() if name.startswith("Data/meshes/actors/character/animations/OSA_Collision/")]
                for scene_name in scene_names:
                    data = json.loads(converted.read(scene_name).decode("utf-8"))
                    speeds.extend(speed["animation"] for speed in data["speeds"])

            self.assertEqual(len(speeds), 2)
            self.assertEqual(len(set(speeds)), 2)
            self.assertTrue(any(speed.startswith("Event_A_") for speed in speeds))
            for speed in speeds:
                self.assertTrue(any(name.endswith(f"{speed}_0.hkx") for name in hkx_names))

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["duplicateExpectedSceneEventCount"], 0)

    def test_osa_duplicate_hkx_discovery_is_deduped_before_packaging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Duplicate HKX.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Dupe.xml",
                    """\
<scene id="AA|Standing|Ap|Dupe" actors="1">
  <info name="Dupe" />
  <anim id="DupEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/DupEvent_0.hkx", b"same")
                source.writestr("meshes/actors/character/animations/0Sex/AA/Standing/Ap/DupEvent_0.hkx", b"same")

            result = converter.convert_archive_to_ready_zip(archive)

            self.assertTrue(any("duplicate HKX source" in warning for warning in result.warnings))
            self.assertEqual(result.report["hkxDiscovery"]["duplicateHkxSourceCount"], 1)
            self.assertEqual(result.report["hkxCount"], 1)

            with ZipFile(result.zip_path) as converted:
                hkx_names = [
                    name for name in converted.namelist()
                    if name.startswith("Data/meshes/actors/character/animations/Duplicate_HKX/") and name.endswith(".hkx")
                ]
            self.assertEqual(len(set(hkx_names)), 1)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_osa_speed_event_casing_is_matched_to_hkx_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Case Mismatch.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Case.xml",
                    """\
<scene id="AA|Standing|Ap|Case" actors="2">
  <info name="Case" />
  <speed>
    <sp qnt="1"><anim id="CaseEvent-MMast_S3" /></sp>
  </speed>
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/CaseEvent-mMast_S3_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/CaseEvent-mMast_S3_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["osaEventMapping"]["normalizedEventCaseSpeedCount"], 1)
            self.assertFalse(any("skipped missing animation speed" in warning for warning in result.warnings))
            self.assertEqual(result.scenes[0].speeds[0].animation, "CaseEvent-mMast_S3")

            with ZipFile(result.zip_path) as converted:
                names = converted.namelist()
                scene_name = next(name for name in names if name.endswith("Case_Mismatch_AA_Standing_Ap_Case.json"))
                scene = json.loads(converted.read(scene_name).decode("utf-8"))
            self.assertEqual([speed["animation"] for speed in scene["speeds"]], ["CaseEvent-mMast_S3"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)

    def test_osa_duplicate_speed_entries_are_dropped_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Duplicate Speed.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/DupeSpeed.xml",
                    """\
<scene id="AA|Standing|Ap|DupeSpeed" actors="1">
  <info name="Dupe Speed" />
  <speed>
    <sp qnt="1"><anim id="LoopEvent" /></sp>
    <sp qnt="2"><anim id="LoopEvent" /></sp>
  </speed>
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/LoopEvent_0.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["osaEventMapping"]["duplicateSpeedCount"], 1)
            with ZipFile(result.zip_path) as converted:
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Duplicate_Speed/Duplicate_Speed_AA_Standing_Ap_DupeSpeed.json"
                    ).decode("utf-8")
                )
            self.assertEqual([speed["animation"] for speed in scene["speeds"]], ["LoopEvent"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_osa_standing_idle_scene_with_hkx_is_playable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Standing Idle.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Idle.xml",
                    """\
<scene id="AA|Standing|Ap|Idle" actors="1">
  <info name="Standing Idle" />
  <anim id="StandingIdle" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/StandingIdle_0.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["startableSceneCount"], 1)
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)

    def test_ostim_menu_hub_idle_scene_is_not_a_playable_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Hub Idle.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Hub.xml",
                    """\
<scene id="AA|Standing|HJ|Hub" actors="2">
  <info name="Hub Source" />
  <anim id="HubEvent" l="3" />
  <actors>
    <actor position="0" tags="male" sosBend="5" />
    <actor position="1" tags="female" />
  </actors>
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/HubEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/HubEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True)
            hub_scenes = [scene for scene in result.scenes if converter.is_ostim_menu_hub_scene(scene)]
            self.assertEqual(len(hub_scenes), 1)
            self.assertTrue(hub_scenes[0].no_random_selection)
            self.assertEqual(converter.expected_animation_events(hub_scenes[0]), [])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["ostimMenuHubSceneCount"], 1)
            self.assertEqual(verification.report["startableSceneCount"], 1)

    def test_osa_missing_hkx_output_fails_deploy_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Missing OSA HKX.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Missing.xml",
                    """\
<scene id="AA|Standing|Ap|Missing" actors="1">
  <info name="Missing" />
  <anim id="MissingEvent" l="3" />
</scene>
""",
                )

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertIsNotNone(result.verification)
            self.assertFalse(result.verification.ok)
            self.assertTrue(any("No HKX animation files" in error for error in result.verification.report["errors"]))
            self.assertGreaterEqual(result.verification.report["missingAnimationEventCount"], 1)

    def test_verify_rejects_duplicate_fnis_and_pandora_rows_in_same_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "DuplicateRows.zip"
            scene = {
                "name": "Duplicate Rows",
                "modpack": "DuplicateRows",
                "length": 3,
                "speeds": [{"animation": "DupRow"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicateRows/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/DuplicateRows/DupRow_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/DuplicateRows/FNIS_DuplicateRows_List.txt",
                    "b -Tn DupRow_0 DupRow_0.hkx\nb -Tn DupRow_0 DupRow_0.hkx\n",
                )
                archive.writestr("Data/Pandora_Engine/mod/DuplicateRows/info.xml", "<mod><name>DuplicateRows</name><author>Tester</author></mod>")
                archive.writestr("Data/Pandora_Engine/mod/DuplicateRows/animationdata/DefaultMale.txt", "DupRow_0\nDupRow_0\n")
                archive.writestr(
                    "Data/Pandora_Engine/mod/DuplicateRows/animationsetdata/DefaultMale.txt",
                    "meshes\\actors\\character\\animations\\DuplicateRows\\DupRow_0.hkx\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/duplicaterows/info.ini",
                    "name=DuplicateRows\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/duplicaterows/0_master/#0106.txt",
                    "<hkobject><hkcstring>DupRow_0</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("is listed 2 times" in error for error in verification.report["errors"]))
            self.assertTrue(any("Pandora AnimData event 'DupRow_0' is listed 2 times" in error for error in verification.report["errors"]))

    def test_verify_rejects_identical_att_and_fnis_animation_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "DuplicateAttFnis.zip"
            scene = {
                "name": "Duplicate ATT FNIS",
                "modpack": "DuplicateAttFnis",
                "length": 3,
                "speeds": [{"animation": "IdleRisk"}],
                "actors": [{}, {}],
                "actions": [{"type": "kissing", "actor": 0, "target": 1}],
            }
            rows = "b -Tn IdleRisk_0 IdleRisk_0.hkx\nb -Tn IdleRisk_1 IdleRisk_1.hkx\n"
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicateAttFnis/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/DuplicateAttFnis/IdleRisk_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/DuplicateAttFnis/IdleRisk_1.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/DuplicateAttFnis/FNIS_DuplicateAttFnis_List.txt", rows)
                archive.writestr("Data/meshes/actors/character/animations/DuplicateAttFnis/ATT_duplicateattfnis_animlist.txt", rows)
                archive.writestr("Data/Nemesis_Engine/mod/duplicateattfnis/info.ini", "name=DuplicateAttFnis\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr(
                    "Data/Nemesis_Engine/mod/duplicateattfnis/0_master/#0106.txt",
                    "<hkobject><hkcstring>IdleRisk_0</hkcstring><hkcstring>IdleRisk_1</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_MIXED_INVALID)
            self.assertGreaterEqual(verification.report["duplicateAnimationListFileCount"], 1)
            self.assertTrue(any("Duplicate animation list files" in error for error in verification.report["errors"]))

    def test_verify_rejects_duplicate_pandora_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "DuplicatePandoraRoots.zip"
            scene = {
                "name": "Duplicate Pandora",
                "modpack": "DuplicatePandoraRoots",
                "length": 3,
                "speeds": [{"animation": "PandoraRisk"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            animdata = "PandoraRisk_0\n"
            animset = "meshes\\actors\\character\\animations\\DuplicatePandoraRoots\\PandoraRisk_0.hkx\n"
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicatePandoraRoots/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/DuplicatePandoraRoots/PandoraRisk_0.hkx", b"")
                archive.writestr("Data/Pandora_Engine/mod/DuplicatePandoraRoots/info.xml", "<mod><name>DuplicatePandoraRoots</name><author>Tester</author></mod>")
                archive.writestr("Data/animdata/DefaultMale.txt", animdata)
                archive.writestr("Data/Pandora_Engine/mod/DuplicatePandoraRoots/animationdata/DefaultMale.txt", animdata)
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", animset)
                archive.writestr("Data/Pandora_Engine/mod/DuplicatePandoraRoots/animationsetdata/DefaultMale.txt", animset)

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertGreaterEqual(verification.report["duplicateAnimationDataFileCount"], 1)
            self.assertGreaterEqual(verification.report["duplicateAnimationSetDataFileCount"], 1)
            self.assertTrue(any("Duplicate Pandora AnimData files" in error for error in verification.report["errors"]))
            self.assertTrue(any("Duplicate Pandora AnimSetData files" in error for error in verification.report["errors"]))

    def test_verify_rejects_duplicate_custom_behavior_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "DuplicateBehaviorGraphs.zip"
            scene = {
                "name": "Duplicate Graphs",
                "modpack": "DuplicateBehaviorGraphs",
                "length": 3,
                "speeds": [{"animation": "GraphRisk"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicateBehaviorGraphs/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/DuplicateBehaviorGraphs/GraphRisk_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/DuplicateBehaviorGraphs/FNIS_DuplicateBehaviorGraphs_List.txt", "b -Tn GraphRisk_0 GraphRisk_0.hkx\n")
                archive.writestr("Data/meshes/actors/character/behaviors/FNIS_DuplicateBehaviorGraphs_Behavior.hkx", b"same behavior graph")
                archive.writestr("Data/meshes/actors/character/behaviors/FNIS_DuplicateBehaviorGraphs_Copy_Behavior.hkx", b"same behavior graph")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertGreaterEqual(verification.report["duplicateBehaviorGraphFileCount"], 1)
            self.assertTrue(any("Duplicate custom behavior graph HKX files" in error for error in verification.report["errors"]))

    def test_verify_rejects_packaged_report_with_post_build_verification_not_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "PostBuildNotRun.zip"
            scene = {
                "name": "Post Build Not Run",
                "modpack": "PostBuildNotRun",
                "length": 3,
                "speeds": [{"animation": "CleanEvent"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/PostBuildNotRun/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/PostBuildNotRun/CleanEvent_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/PostBuildNotRun/ATT_postbuildnotrun_animlist.txt",
                    "b -Tn CleanEvent_0 CleanEvent_0.hkx POSTBUILDNOTRUN_AnimationSpeed: 1\n",
                )
                archive.writestr("Data/Nemesis_Engine/mod/postbuildnotrun/info.ini", "name=PostBuildNotRun\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr("Data/Nemesis_Engine/mod/postbuildnotrun/0_master/#0106.txt", "<hkobject><hkcstring>CleanEvent_0</hkcstring></hkobject>\n")
                archive.writestr("conversion_report.json", json.dumps({"postBuildVerification": {"status": "NOT RUN"}}))

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("post-build verification is NOT RUN" in error for error in verification.report["errors"]))

    def test_verify_rejects_k4_scene_when_actor_event_is_not_registered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "K4MissingActorEvent.zip"
            scene = {
                "name": "K4 Cowgirl",
                "modpack": "K4MissingActorEvent",
                "length": 3,
                "speeds": [{"animation": "cowgirldg_S1"}],
                "actors": [{}, {}],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/K4MissingActorEvent/Scene.json", json.dumps(scene))
                archive.writestr("Data/meshes/actors/character/animations/K4MissingActorEvent/cowgirldg_S1_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/K4MissingActorEvent/cowgirldg_S1_1.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/K4MissingActorEvent/ATT_k4missingactorevent_animlist.txt",
                    "b -Tn cowgirldg_S1_0 cowgirldg_S1_0.hkx K4MISSINGACTOREVENT_AnimationSpeed: 1\n",
                )
                archive.writestr("Data/Nemesis_Engine/mod/k4missingactorevent/info.ini", "name=K4MissingActorEvent\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr("Data/Nemesis_Engine/mod/k4missingactorevent/0_master/#0106.txt", "<hkobject><hkcstring>cowgirldg_S1_0</hkcstring></hkobject>\n")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["missingNemesisPatchEventCount"], 1)
            self.assertTrue(any("cowgirldg_S1_1" in error and "missing from a real Nemesis/ATT behavior patch" in error for error in verification.report["errors"]))

    def test_archive_build_can_add_ostim_main_menu_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Menu Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/MenuPack/Scene.json",
                    json.dumps(
                        {
                            "name": "Menu Scene",
                            "modpack": "MenuPack",
                            "length": 5,
                            "speeds": [{"animation": "MenuLoop"}],
                            "actors": [
                                {"intendedSex": "male", "animationIndex": 0},
                                {"intendedSex": "female", "animationIndex": 1},
                            ],
                            "actions": [{"type": "kissing", "actor": 0, "target": 1}],
                        }
                    ),
                )
                source.writestr("Data/meshes/actors/character/animations/MenuPack/MenuLoop_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/MenuPack/MenuLoop_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(
                archive,
                ostim_menu_entry=True,
                ostim_menu_icon="OStim/symbols/search",
            )

            self.assertEqual(result.report["ostimMenuHubSceneCount"], 1)
            self.assertEqual(result.report["externalOStimMenuOriginCount"], 1)
            self.assertEqual(result.report["missingSceneLinkCount"], 0)

            with ZipFile(result.zip_path) as converted:
                menu_scene = json.loads(
                    converted.read(
                        mod_entry("Data/SKSE/Plugins/OStim/scenes/Menu_Pack/Menu_Pack_Menu_MF.json")
                    ).decode("utf-8")
                )
                metadata = json.loads(
                    converted.read(mod_entry("Data/SKSE/Plugins/OStim/converter_metadata/Menu_Pack/metadata.json")).decode("utf-8")
                )

            origin_navs = [nav for nav in menu_scene["navigations"] if nav.get("origin")]
            self.assertEqual(origin_navs[0]["origin"], "OStim2PStandingApartMF")
            self.assertEqual(origin_navs[0]["description"], "Menu_Pack")
            self.assertEqual(origin_navs[0]["icon"], "OStim/symbols/search")
            self.assertTrue(origin_navs[0]["noWarnings"])
            self.assertTrue(menu_scene["noRandomSelection"])
            self.assertTrue(metadata["compatibility"]["hasOStimMenuEntry"])
            self.assertEqual(metadata["deployment"]["ostimMenuHubSceneCount"], 1)
            self.assertEqual(metadata["deployment"]["externalOStimMenuOriginCount"], 1)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["ostimMenuHubSceneCount"], 1)
            self.assertEqual(verification.report["missingSceneLinkCount"], 0)

    def test_verify_rejects_ostim_menu_entry_with_wrong_actor_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "BadMenuSlots.zip"
            menu_scene = {
                "name": "Bad Menu",
                "modpack": "BadMenu",
                "length": 5,
                "speeds": [{"animation": "Loop"}],
                "actors": [
                    {"intendedSex": "female", "animationIndex": 0},
                    {"intendedSex": "male", "animationIndex": 1},
                ],
                "actions": [{"type": "kissing", "actor": 1, "target": 0}],
                "tags": [converter.OSTIM_MENU_HUB_TAG],
                "noRandomSelection": True,
                "navigations": [
                    {
                        "origin": "OStim2PStandingApartMF",
                        "noWarnings": True,
                        "icon": "OStim/symbols/search",
                    }
                ],
            }
            list_text = "\n".join(
                [
                    "b -Tn Loop_0 Loop_0.hkx",
                    "b -Tn Loop_1 Loop_1.hkx",
                    "",
                ]
            )
            animset_text = "\n".join(
                [
                    "meshes\\actors\\character\\animations\\BadMenu\\Loop_0.hkx",
                    "meshes\\actors\\character\\animations\\BadMenu\\Loop_1.hkx",
                    "",
                ]
            )
            animdata_text = "Loop_0\nLoop_1\n"
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("SKSE/Plugins/OStim/scenes/BadMenu/BadMenu_Menu_MF.json", json.dumps(menu_scene))
                archive.writestr("meshes/actors/character/animations/BadMenu/Loop_0.hkx", b"")
                archive.writestr("meshes/actors/character/animations/BadMenu/Loop_1.hkx", b"")
                archive.writestr("meshes/actors/character/animations/BadMenu/FNIS_BadMenu_List.txt", list_text)
                archive.writestr("animdata/DefaultMale.txt", animdata_text)
                archive.writestr("animationsetdatasinglefile/DefaultMale.txt", animset_text)
                archive.writestr("Pandora_Engine/mod/BadMenu/info.xml", "<mod><name>BadMenu</name><author>Tester</author></mod>")
                archive.writestr("Pandora_Engine/mod/BadMenu/animdata/DefaultMale.txt", animdata_text)
                archive.writestr("Pandora_Engine/mod/BadMenu/animationsetdata/DefaultMale.txt", animset_text)

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["dataRoots"], [converter.ARCHIVE_ROOT_DATA_PREFIX])
            self.assertTrue(any("expects actor slots 'mf'" in error for error in verification.report["errors"]))

    def test_verify_rejects_duplicate_ostim_menu_navigation_destinations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "DuplicateMenuNavigation.zip"
            scene = {
                "name": "Playable",
                "modpack": "DuplicateMenuNavigation",
                "length": 3,
                "speeds": [{"animation": "Loop"}],
                "actors": [
                    {"intendedSex": "male", "animationIndex": 0},
                    {"intendedSex": "female", "animationIndex": 1},
                ],
                "actions": [{"type": "kissing", "actor": 0, "target": 1}],
            }
            menu_scene = {
                "name": "Menu",
                "modpack": "DuplicateMenuNavigation",
                "length": 2,
                "speeds": [{"animation": "OStim2PStandingApartMF"}],
                "actors": [
                    {"intendedSex": "male", "animationIndex": 0},
                    {"intendedSex": "female", "animationIndex": 1},
                ],
                "tags": [converter.OSTIM_MENU_HUB_TAG, converter.OSTIM_MENU_ROOT_TAG],
                "noRandomSelection": True,
                "navigations": [
                    {
                        "origin": "OStim2PStandingApartMF",
                        "priority": converter.OSTIM_MENU_ENTRY_PRIORITY,
                        "description": "DuplicateMenuNavigation",
                        "icon": converter.DEFAULT_OSTIM_MENU_ICON,
                        "noWarnings": True,
                    },
                    {
                        "destination": "Scene",
                        "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 10,
                        "description": "Playable",
                        "icon": converter.DEFAULT_OSTIM_MENU_ICON,
                    },
                    {
                        "destination": "Scene",
                        "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 11,
                        "description": "Playable again",
                        "icon": converter.DEFAULT_OSTIM_MENU_ICON,
                    },
                ],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicateMenuNavigation/Scene.json", json.dumps(scene))
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/DuplicateMenuNavigation/Menu.json", json.dumps(menu_scene))
                archive.writestr("Data/meshes/actors/character/animations/DuplicateMenuNavigation/Loop_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/DuplicateMenuNavigation/Loop_1.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/DuplicateMenuNavigation/ATT_duplicatemenunavigation_animlist.txt",
                    "b -Tn Loop_0 Loop_0.hkx DUPLICATEMENUNAVIGATION_AnimationSpeed: 1\n"
                    "b -Tn Loop_1 Loop_1.hkx DUPLICATEMENUNAVIGATION_AnimationSpeed: 1\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/duplicatemenunavigation/info.ini",
                    "name=DuplicateMenuNavigation\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/duplicatemenunavigation/0_master/#0106.txt",
                    "<hkobject><hkcstring>Loop_0</hkcstring><hkcstring>Loop_1</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["duplicateMenuNavigationCount"], 1)
            self.assertTrue(any("appears 2 times in the same menu node" in error for error in verification.report["errors"]))

    def test_generated_ostim_menu_navigation_dedupe_removes_duplicate_destinations(self):
        pack = "MenuDedupe"
        menu_scene = converter.ostim_menu_node_scene(
            pack,
            f"{pack}_Menu_MF",
            f"{pack}|ostim_menu|mf",
            pack,
            "mf",
            "OStim2PStandingApartMF",
            (converter.OSTIM_MENU_ROOT_TAG, "signature:mf"),
        )
        menu_scene.navigations.append(
            converter.Navigation(
                destination=f"{pack}_Scene",
                description="Scene again",
                icon=converter.DEFAULT_OSTIM_MENU_ICON,
                border="",
            )
        )
        menu_scene.navigations.append(
            converter.Navigation(
                destination=f"{pack}_Scene",
                description="Scene",
                icon=converter.DEFAULT_OSTIM_MENU_ICON,
                border="",
            )
        )

        removed = converter.dedupe_generated_ostim_menu_navigations([menu_scene])

        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["destination"], f"{pack}_Scene")
        self.assertEqual(
            [nav.destination for nav in menu_scene.navigations if nav.destination == f"{pack}_Scene"],
            [f"{pack}_Scene"],
        )

    def test_sexlab_slal_json_converts_to_ostim_events_and_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SexLab Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Ace_Test.json",
                    json.dumps(
                        {
                            "name": "Ace_Test",
                            "animations": [
                                {
                                    "id": "Ace_TestFootjob",
                                    "name": "Ace Test Footjob",
                                    "sound": "none",
                                    "tags": "Ace,footjob,feet,MF",
                                    "actors": [
                                        {
                                            "type": "Female",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A1_S1"},
                                                {"id": "Ace_TestFootjob_A1_S2"},
                                            ],
                                        },
                                        {
                                            "type": "Male",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A2_S1", "strap_on": True},
                                                {"id": "Ace_TestFootjob_A2_S2", "strap_on": True},
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S2.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S2.hkx", b"")
                source.writestr("meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx", b"old behavior")

            result = converter.convert_archive_to_ready_zip(archive, mod_author="Ace")
            self.assertEqual(len(result.scenes), 1)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        mod_entry("Data/SKSE/Plugins/OStim/scenes/SexLab_Pack/SexLab_Pack_Ace_TestFootjob.json")
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "SexLab_Pack")
                behavior_events = read_pandora_compatible_events(converted, names, "SexLab_Pack")
                nemesis_code = converter.safe_nemesis_patch_code(result.pack)

            self.assertIn(mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/Ace_TestFootjob_S1_0.hkx"), names)
            self.assertIn(mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/Ace_TestFootjob_S1_1.hkx"), names)
            self.assertIn(mod_entry(f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini"), names)
            self.assertNotIn(mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/Ace_Test/Ace_TestFootjob_A1_S1.hkx"), names)
            self.assertNotIn(mod_entry("Data/meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx"), names)
            self.assertNotIn(mod_entry("Data/meshes/actors/character/behaviors/FNIS_SexLab_Pack_Behavior.hkx"), names)
            self.assertEqual([speed["animation"] for speed in scene["speeds"]], ["Ace_TestFootjob_S1", "Ace_TestFootjob_S2"])
            self.assertEqual([actor.get("intendedSex") for actor in scene["actors"]], ["male", "female"])
            self.assertEqual(scene["actions"][0]["type"], "footjob")
            self.assertEqual(scene["actions"][0]["actor"], 0)
            self.assertEqual(scene["actions"][0]["target"], 1)
            self.assertEqual(len(character_att_list_paths(names, "SexLab_Pack")), 1)
            self.assertIn("Ace_TestFootjob_S1_0", behavior_events)
            self.assertIn("Ace_TestFootjob_S2_1", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 0)
            self.assertEqual(verification.report["missingNemesisPatchEventCount"], 0)
            self.assertGreater(verification.report["registeredNemesisPatchEventCount"], 0)
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE)
            self.assertIsNotNone(result.verification)
            self.assertTrue(result.verification.ok, result.verification.report["errors"])
            self.assertTrue(result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report.json").exists())
            self.assertTrue(result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report_README.txt").exists())
            external_report = json.loads(result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report.json").read_text(encoding="utf-8"))
            self.assertEqual(external_report["postBuildVerification"]["status"], result.verification.report["status"])
            self.assertIn("sourceArchiveFingerprint", external_report)
            with ZipFile(result.zip_path) as converted:
                self.assertIn("conversion_report_README.txt", set(converted.namelist()))

    def test_diagnose_source_archive_reports_detected_type_and_missing_hkx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Diagnose SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Diag.json",
                    json.dumps(
                        {
                            "name": "Diag",
                            "animations": [
                                {
                                    "id": "DiagScene",
                                    "tags": "MF",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "DiagScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "DiagScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Diag/DiagScene_A1_S1.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive)

            self.assertEqual(diagnosis.report["detectedSourceTypeCode"], "sexlabSlal")
            self.assertEqual(diagnosis.report["detectionConfidence"], "High")
            self.assertEqual(diagnosis.report["recommendedOutput"], "OStim Standalone or SexLab/SLAL")
            self.assertGreater(diagnosis.report["missingReferencedHkxCount"], 0)
            self.assertTrue(any(issue["severity"] == "high" for issue in diagnosis.report["knownIssues"]))
            self.assertTrue(diagnosis.report_path.exists())
            self.assertTrue(diagnosis.text_report_path.exists())

    def test_source_diagnosis_text_accepts_string_known_issues(self):
        report = {
            "status": "PASS WITH WARNINGS",
            "converterVersion": "test-version",
            "sourceArchiveName": "Known Issue Pack.zip",
            "sourceArchiveFingerprint": {"sizeBytes": 1, "sha256": "abc", "hashMode": "full"},
            "detectedSourceType": "SexLab/SLAL animation JSON",
            "detectedSourceTypes": ["SexLab/SLAL animation JSON"],
            "detectionConfidence": "High",
            "recommendedOutput": "OStim Standalone",
            "recommendedBehaviorTool": "Pandora recommended",
            "knownIssues": ["Plain compatibility issue from profile."],
            "warnings": [],
            "errors": [],
        }

        text = converter.source_diagnosis_report_to_text(report)
        summary = converter.copyable_diagnosis_summary(report)

        self.assertIn("[MEDIUM] Plain compatibility issue from profile.", text)
        self.assertIn("Plain compatibility issue from profile.", summary)

    def test_report_helpers_accept_legacy_string_fields(self):
        report = {
            "status": "PASS WITH WARNINGS",
            "compatibilityMatch": "Legacy compatibility profile text.",
            "sourceArchiveFingerprint": "legacy fingerprint text",
            "knownIssues": ["Plain compatibility issue from profile."],
        }

        self.assertEqual(
            converter.compatibility_known_pack_line(report["compatibilityMatch"]),
            "Legacy compatibility profile text.",
        )

        text = converter.source_diagnosis_report_to_text(report)
        summary = converter.copyable_diagnosis_summary(report)
        bug_report = converter.nexus_bug_report_text(report, verification_report="legacy verification")

        self.assertIn("Archive size: unknown bytes", text)
        self.assertIn("Known pack match: None", summary)
        self.assertIn("Verification result: PASS WITH WARNINGS", bug_report)

    def test_public_report_renderers_tolerate_malformed_legacy_data(self):
        legacy_report = {
            "status": "PASS WITH WARNINGS",
            "ok": True,
            "behaviorGeneration": "legacy behavior text",
            "behaviorRegistrationSummary": "legacy behavior summary text",
            "compatibilityMatch": {
                "matchResult": "Partial",
                "matchedPack": "Legacy Pack",
                "database": "legacy database text",
            },
            "generatedOStimNavigationDeduplication": "legacy navigation text",
            "sourceArchiveFingerprint": "legacy fingerprint text",
            "recommendedNextActions": ["Review the generated report."],
            "invalidRows": ["legacy row text"],
            "missingHkxRows": ["legacy missing row text"],
            "humanOnlyOStimOutput": True,
            "humanOnlyOStimFiltering": {
                "enabled": True,
                "detectedCreatureRoots": ["canine"],
                "skippedSceneExamples": {
                    "examples": [
                        {"name": "Wrapped creature scene"},
                        "Legacy string scene",
                    ],
                    "showing": 2,
                    "total": 25,
                    "truncated": True,
                },
            },
            "actorRootEventMapping": {
                "crossRootDuplicateEvents": ["LegacyCrossRoot", {"event": "MappedCrossRoot"}],
                "sameRootConflicts": ["LegacySameRoot", {"event": "MappedSameRoot"}],
            },
            "warnings": ["Legacy warning."],
        }

        renderers = [
            converter.copyable_diagnosis_summary,
            converter.source_diagnosis_report_to_text,
            converter.conversion_failure_report_to_text,
            converter.report_to_text,
            converter.recommended_workflow_panel_text,
            converter.pandora_log_analysis_to_text,
            converter.creature_runtime_check_report_to_text,
            converter.archive_structure_report_to_text,
            converter.conversion_report_readme_text,
            converter.pandora_module_diagnostics_to_text,
            converter.ocreatures_reference_comparison_to_text,
            converter.verification_report_to_text,
            converter.sexlab_verification_report_to_text,
            converter.sexlab_conversion_report_text,
            converter.install_steps_text,
            converter.result_summary_text,
            converter.aac_readme_text,
        ]

        for renderer in renderers:
            with self.subTest(renderer=renderer.__name__):
                text = renderer(legacy_report)
                self.assertIsInstance(text, str)
                self.assertTrue(text.strip())

        self.assertIn("Wrapped creature scene", converter.report_to_text(legacy_report))
        self.assertIn("Legacy string scene", converter.verification_report_to_text(legacy_report))
        self.assertIn("MappedCrossRoot", converter.report_to_text(legacy_report))

        bug_report = converter.nexus_bug_report_text(legacy_report, verification_report="legacy verification")
        manifest = converter.aac_manifest_for_report(legacy_report)
        installability = converter.installability_summary("legacy report text")
        source_selection = converter.source_selection_report("legacy source text", "legacy compatibility text")
        duplicate_hkx = converter.duplicate_hkx_handling_summary("legacy diagnostics", [], [], "legacy match")
        finalized = converter.finalize_verification_report("legacy verification text", "OStim Standalone")

        self.assertIn("Legacy Pack", bug_report)
        self.assertEqual(manifest["compatibilityProfile"]["matchedPack"], "Legacy Pack")
        self.assertEqual(installability["result"], "FAIL")
        self.assertEqual(source_selection["selectedSourceParser"], "unknown")
        self.assertEqual(duplicate_hkx, {})
        self.assertEqual(finalized["status"], "FAIL")

    def test_no_attribute_error_when_warning_entry_is_string(self):
        report = {
            "status": "PASS WITH WARNINGS",
            "sourceArchiveName": "Warnings.zip",
            "sourceArchiveFingerprint": {"sizeBytes": 1, "sha256": "abc", "hashMode": "full"},
            "warnings": ["plain warning entry"],
            "knownIssues": ["plain known issue"],
        }

        text = converter.source_diagnosis_report_to_text(report)

        self.assertIn("plain warning entry", text)
        self.assertNotIn("AttributeError", text)

    def test_no_attribute_error_when_report_section_is_string(self):
        report = {
            "status": "PASS WITH WARNINGS",
            "sourceArchiveName": "Malformed Sections.zip",
            "sourceArchiveFingerprint": "legacy fingerprint text",
            "compatibilityMatch": "legacy match text",
            "compatibilityDatabase": "legacy database text",
            "sourceSelection": "legacy source selection text",
            "detectedSourceTypes": "SexLab/SLAL animation JSON",
        }

        text = converter.source_diagnosis_report_to_text(report)

        self.assertIn("Malformed Report Sections", text)
        self.assertIn("expected object, got string", text)
        self.assertNotIn("AttributeError", text)

    def test_no_attribute_error_when_manifest_section_is_string(self):
        report = {
            "status": "PASS",
            "pack": "Manifest Pack",
            "compatibilityMatch": "legacy match text",
            "sourceArchiveFingerprint": "legacy fingerprint text",
            "behaviorRegistrationSummary": "legacy behavior summary",
            "verificationResult": "legacy verification",
        }

        manifest = converter.aac_manifest_for_report(report)
        readme = converter.aac_readme_text(report)

        self.assertEqual(manifest["compatibilityProfile"]["matchResult"], "None")
        self.assertIn("Manifest Pack", readme)

    def test_malformed_source_json_entries_do_not_raise_attribute_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slal = root / "SLAL.json"
            slal.write_text(
                json.dumps(
                    {
                        "name": "Malformed SLAL",
                        "animations": [
                            "not an animation object",
                            {
                                "id": "Valid",
                                "actors": [
                                    {"type": "Male", "stages": [{"id": "Valid_A1_S1"}]},
                                    {"type": "Female", "stages": [{"id": "Valid_A2_S1"}]},
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            slal_scenes, slal_warnings, _event_map, _rows = converter.parse_sexlab_animation_json(slal)

            slsb = root / "Source.slsb.json"
            slsb.write_text(
                json.dumps(
                    {
                        "pack_name": "Malformed SLSB",
                        "scenes": [
                            "not a scene object",
                            {
                                "id": "ValidSLSB",
                                "stages": [{"positions": [{"event": "ValidSLSB_A1_S1"}]}],
                                "positions": [{"sex": {"male": True}}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            slsb_scenes, slsb_warnings, _slsb_event_map = converter.parse_slsb_source_json(slsb)

            ostim = root / "Scene.json"
            ostim.write_text(json.dumps(["not an object"]), encoding="utf-8")
            ostim_scenes, ostim_warnings = converter.parse_ostim_scene_json_files([ostim])

            alignment = root / "alignment.json"
            alignment.write_text(json.dumps("not an object"), encoding="utf-8")
            alignment_data, alignment_warnings = converter.load_alignment_json(alignment)

            self.assertEqual(len(slal_scenes), 1)
            self.assertTrue(any("malformed SexLab/SLAL animation entry" in warning for warning in slal_warnings))
            self.assertEqual(len(slsb_scenes), 1)
            self.assertTrue(any("malformed SLSB scene entry" in warning for warning in slsb_warnings))
            self.assertEqual(ostim_scenes, [])
            self.assertTrue(any("OStim scene JSON root" in warning for warning in ostim_warnings))
            self.assertEqual(alignment_data, {})
            self.assertTrue(any("not a JSON object" in warning for warning in alignment_warnings))

    def test_diagnose_source_archive_accepts_case_variant_hkx_event_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Case Variant SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/CaseVariant.json",
                    json.dumps(
                        {
                            "name": "CaseVariant",
                            "animations": [
                                {
                                    "id": "CaseScene",
                                    "tags": "MF,footjob",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "CaseScene_Feetonface_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "CaseScene_Feetonface_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/CaseVariant/CaseScene_Feetonface_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/CaseVariant/CaseScene_FeetonFace_A2_S1.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)

            self.assertEqual(diagnosis.report["detectedSourceTypeCode"], "sexlabSlal")
            self.assertEqual(diagnosis.report["missingReferencedHkxCount"], 0)
            self.assertTrue(diagnosis.report["conversionLikelySafe"])
            self.assertFalse(any(issue["severity"] == "high" for issue in diagnosis.report["knownIssues"]))

    def test_diagnose_allows_adult_short_stature_pack_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Adult Midget SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/AdultMidget.json",
                    json.dumps(
                        {
                            "name": "AdultMidget",
                            "animations": [
                                {
                                    "id": "AdultShortScene",
                                    "tags": "MF",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "AdultShortScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "AdultShortScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Adult Midget/AdultShortScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Adult Midget/AdultShortScene_A2_S1.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)

            self.assertFalse(diagnosis.report["adultOnlyContentBlocked"])
            self.assertFalse(diagnosis.report["adultOnlyContentFlags"]["blocked"])
            self.assertEqual(diagnosis.report["detectedSourceTypeCode"], "sexlabSlal")
            self.assertTrue(diagnosis.report["conversionLikelySafe"])

    def test_diagnose_ignores_minor_terms_in_nemesis_baseline_catalogs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "OStim Pack With Nemesis Baseline.zip"
            scene = {
                "name": "Adult Scene",
                "modPack": "adultpack",
                "length": 2,
                "speeds": [{"animation": "AdultScene"}],
                "actors": [
                    {"type": "npc", "intendedSex": "male"},
                    {"type": "npc", "intendedSex": "female"},
                ],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            baseline_catalog = (
                "<hkcstring>Animations\\DLC01\\Child_CryingEnd.hkx</hkcstring>\n"
                "<hkcstring>Animations\\BYOH\\Special_ChildPlayIdle1.HKX</hkcstring>\n"
            )
            with ZipFile(archive, "w") as source:
                source.writestr("SKSE/Plugins/OStim/scenes/AdultPack/AdultScene.json", json.dumps(scene))
                source.writestr("meshes/actors/character/animations/AdultPack/AdultScene_0.hkx", b"")
                source.writestr("meshes/actors/character/animations/AdultPack/AdultScene_1.hkx", b"")
                source.writestr("Nemesis_Engine/mod/adultpack/defaultmale/#0029.txt", baseline_catalog)
                source.writestr("Nemesis_Engine/mod/adultpack/defaultfemale/#0029.txt", baseline_catalog)
                source.writestr(
                    "meshes/actors/character/animations/AdultPack/ATT_AdultPack_animlist.txt",
                    "b -Tn AdultScene_0 AdultScene_0.hkx\nb -Tn AdultScene_1 AdultScene_1.hkx\n",
                )

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)

            self.assertEqual(diagnosis.report["detectedSourceTypeCode"], "ostimStandalone")
            self.assertFalse(diagnosis.report["adultOnlyContentBlocked"])
            self.assertFalse(diagnosis.report["adultOnlyContentFlags"]["blocked"])

    def test_minor_coded_custom_animation_registration_remains_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Blocked OStim Registration.zip"
            scene = {
                "name": "Adult Scene",
                "modPack": "blockedpack",
                "length": 2,
                "speeds": [{"animation": "AdultScene"}],
                "actors": [{"type": "npc"}, {"type": "npc"}],
            }
            with ZipFile(archive, "w") as source:
                source.writestr("SKSE/Plugins/OStim/scenes/BlockedPack/AdultScene.json", json.dumps(scene))
                source.writestr("meshes/actors/character/animations/BlockedPack/AdultScene_0.hkx", b"")
                source.writestr("meshes/actors/character/animations/BlockedPack/AdultScene_1.hkx", b"")
                source.writestr(
                    "meshes/actors/character/animations/BlockedPack/ATT_BlockedPack_animlist.txt",
                    "b -Tn Child_AdultScene_0 AdultScene_0.hkx\n",
                )

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)

            self.assertTrue(diagnosis.report["adultOnlyContentBlocked"])
            hits = diagnosis.report["adultOnlyContentFlags"]["hits"]
            self.assertTrue(any(hit["kind"] == "text" and "ATT_BlockedPack" in hit["path"] for hit in hits))

    def test_minor_coded_adult_archive_is_blocked_before_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Blocked SLAL.zip"
            output_zip = root / "Blocked_Output.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Blocked.json",
                    json.dumps(
                        {
                            "name": "Blocked",
                            "animations": [
                                {
                                    "id": "BlockedScene",
                                    "tags": "MF",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "BlockedScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "BlockedScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Shota Pack/BlockedScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Shota Pack/BlockedScene_A2_S1.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)

            self.assertEqual(diagnosis.report["status"], "FAIL")
            self.assertTrue(diagnosis.report["adultOnlyContentBlocked"])
            self.assertTrue(diagnosis.report["adultOnlyContentFlags"]["blocked"])
            self.assertEqual(diagnosis.report["recommendedOutput"], "unsupported")
            self.assertFalse(diagnosis.report["conversionLikelySafe"])
            self.assertTrue(any("[minor-coded]" in hit["path"] for hit in diagnosis.report["adultOnlyContentFlags"]["hits"]))
            self.assertTrue(any("Minor-coded labels" in issue["detected"] for issue in diagnosis.report["knownIssues"]))
            candidate = diagnosis.report["compatibilityAutoFailCandidate"]
            fingerprint = diagnosis.report["sourceArchiveFingerprint"]
            self.assertTrue(candidate["readyForCompatibilityDb"])
            self.assertEqual(candidate["sourceArchiveHash"], fingerprint["sha256"])
            self.assertEqual(candidate["sourceArchiveHashMode"], "full")
            self.assertEqual(candidate["compatibilityDbEntryTemplate"]["archiveFingerprints"], [fingerprint["sha256"]])
            self.assertEqual(candidate["compatibilityDbEntryTemplate"]["recommendedOutputType"], "unsupported")
            self.assertEqual(candidate["compatibilityDbEntryTemplate"]["status"], "unsupported")
            self.assertIn("Compatibility Auto-Fail Candidate", converter.source_diagnosis_report_to_text(diagnosis.report))
            self.assertIn(fingerprint["sha256"], converter.copyable_diagnosis_summary(diagnosis.report))
            self.assertIn(fingerprint["sha256"], converter.nexus_bug_report_text(diagnosis.report))

            with self.assertRaises(RuntimeError) as raised:
                converter.convert_archive_to_ready_zip(archive, zip_path=output_zip)

            self.assertIn("minor-coded labels", str(raised.exception))
            failure_report = output_zip.with_name(f"{output_zip.stem}_conversion_failed.json")
            failure_text_report = output_zip.with_name(f"{output_zip.stem}_conversion_failed.txt")
            self.assertTrue(failure_report.exists())
            self.assertTrue(failure_text_report.exists())
            failure_data = json.loads(failure_report.read_text(encoding="utf-8"))
            self.assertTrue(failure_data["adultOnlyContentBlocked"])
            self.assertTrue(failure_data["adultOnlyContentFlags"]["blocked"])
            self.assertEqual(
                failure_data["compatibilityAutoFailCandidate"]["compatibilityDbEntryTemplate"]["archiveFingerprints"],
                [fingerprint["sha256"]],
            )
            self.assertIn("Compatibility Auto-Fail Candidate", failure_text_report.read_text(encoding="utf-8"))

    def test_compatibility_db_loads_valid_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "compatibility_db.json"
            db_path.write_text(
                json.dumps(
                    {
                        "version": "test-db",
                        "entries": [
                            {
                                "id": "known",
                                "packDisplayName": "Known Pack",
                                "sourceFramework": "sexlabSlal",
                                "status": "Working",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]), mock.patch.object(converter, "app_debug_mode", return_value=False):
                db = converter.load_compatibility_database()

            self.assertTrue(db["loaded"])
            self.assertEqual(db["version"], "test-db")
            self.assertEqual(db["entries"][0]["packDisplayName"], "Known Pack")
            self.assertEqual(db["warnings"], [])

    def test_invalid_compatibility_db_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "compatibility_db.json"
            db_path.write_text(json.dumps({"version": "bad", "entries": ["not an object"]}), encoding="utf-8")

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                db = converter.load_compatibility_database()

            self.assertTrue(db["warnings"])
            self.assertEqual(db["entries"], [])

    def test_no_attribute_error_when_compat_entry_is_string(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "compatibility_db.json"
            db_path.write_text(json.dumps({"version": "bad", "entries": ["not an object"]}), encoding="utf-8")

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                db = converter.load_compatibility_database()

            self.assertEqual(db["entries"], [])
            self.assertTrue(any("not an object" in warning for warning in db["warnings"]))

    def test_malformed_compatibility_profile_fields_warn_and_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "compatibility_db.json"
            db_path.write_text(
                json.dumps(
                    {
                        "version": "malformed-fields",
                        "entries": [
                            {
                                "id": "bad-fields",
                                "packDisplayName": "Bad Fields",
                                "sourceFramework": "sexlabSlal",
                                "warnings": "plain warning",
                                "knownWarnings": "plain known warning",
                                "recommendedOutput": {"bad": "shape"},
                                "knownPackMatch": "malformed nested match",
                                "archiveFingerprints": {"bad": "shape"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                db = converter.load_compatibility_database()

            self.assertEqual(len(db["entries"]), 1)
            self.assertNotIn("recommendedOutput", db["entries"][0])
            self.assertNotIn("knownPackMatch", db["entries"][0])
            self.assertEqual(db["entries"][0]["knownWarnings"], ["plain known warning"])
            self.assertTrue(any("expected string" in warning or "expected object" in warning for warning in db["warnings"]))

    def test_known_pack_exact_fingerprint_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Known Exact.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Known.json",
                    json.dumps(
                        {
                            "name": "Known",
                            "animations": [
                                {
                                    "id": "KnownScene",
                                    "tags": "MF",
                                    "actors": [
                                        {"type": "Male", "stages": [{"id": "KnownScene_A1_S1"}]},
                                        {"type": "Female", "stages": [{"id": "KnownScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Known/KnownScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Known/KnownScene_A2_S1.hkx", b"")
            fingerprint = converter.source_archive_fingerprint(archive)["sha256"]
            db_path = root / "compatibility_db.json"
            db_path.write_text(
                json.dumps(
                    {
                        "version": "exact-db",
                        "entries": [
                            {
                                "id": "known-exact",
                                "packDisplayName": "Known Exact Pack",
                                "sourceFramework": "sexlabSlal",
                                "archiveFingerprints": [fingerprint],
                                "recommendedOutputType": "OStim Standalone",
                                "recommendedBehaviorTool": "Pandora recommended",
                                "status": "Working",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                diagnosis = converter.diagnose_source_archive(archive, report_dir=root)

            match = diagnosis.report["compatibilityMatch"]
            self.assertEqual(match["matchResult"], "Exact")
            self.assertEqual(match["matchedPack"], "Known Exact Pack")
            self.assertEqual(match["confidence"], "High")
            self.assertEqual(diagnosis.report["recommendedOutput"], "OStim Standalone")

    def test_known_pack_partial_pattern_match_and_no_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "compatibility_db.json"
            db_path.write_text(
                json.dumps(
                    {
                        "version": "pattern-db",
                        "entries": [
                            {
                                "id": "k4-pattern",
                                "packDisplayName": "K4-style test profile",
                                "sourceFramework": "sexlabSlal",
                                "namePatterns": ["*k4*"],
                                "filePatterns": ["*/SLAnims/json/*.json"],
                                "recommendedOutputType": "OStim Standalone",
                                "recommendedBehaviorTool": "Pandora recommended",
                                "status": "Working with warnings",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source_detection = {"selectedSourceType": "sexlabSlal", "detectedSourceTypes": ["sexlabSlal"]}
            archive = root / "K4 Synthetic.zip"
            archive.write_bytes(b"fake")
            extracted = root / "extracted"
            (extracted / "SLAnims" / "json").mkdir(parents=True)
            (extracted / "SLAnims" / "json" / "K4.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                match = converter.match_compatibility_profile(archive, extracted, source_detection)
                no_match = converter.match_compatibility_profile(root / "Other.zip", extracted, {"selectedSourceType": "ostimStandalone"})

            self.assertEqual(match["matchResult"], "Partial")
            self.assertEqual(match["matchedPack"], "K4-style test profile")
            self.assertEqual(no_match["matchResult"], "None")

    def test_bakafactory_profile_records_mixed_slsb_slal_workflow(self):
        db = json.loads((ROOT / "compatibility_db.json").read_text(encoding="utf-8"))
        profile = next((entry for entry in db["entries"] if entry.get("id") == "bakafactory-slal-animation-78"), None)

        self.assertIsNotNone(profile)
        self.assertEqual(profile["sourceFramework"], "sexlabSceneBuilder")
        self.assertEqual(profile["preferredSourceParser"], "slsbSourceJson")
        self.assertEqual(profile["fallbackSourceParsers"], ["slalJson"])
        self.assertEqual(profile["sourceBranchPatterns"], ["SLSB SE"])
        self.assertIn("SLAL SE", profile["ignoredBranchPatterns"])
        self.assertIn("SLAL LE", profile["ignoredBranchPatterns"])
        self.assertIn("SexLab P+/SLSB source JSON", profile["expectedMixedSourceTypes"])
        self.assertIn("SexLab/SLAL animation JSON", profile["expectedMixedSourceTypes"])
        self.assertTrue(profile["defaultHumanOnlyOStim"])
        self.assertTrue(profile["creatureRuntimeRequired"])
        self.assertTrue(profile["expectedDuplicateHkxHighCount"])
        self.assertEqual(profile["recommendedOStimToolsMode"], "project")
        self.assertIn("SexLab P+/SLSB", profile["supportedOutputTypes"])
        self.assertIn("OStim Tools JSON project", profile["supportedOutputTypes"])

    def test_bakafactory_mixed_slsb_slal_diagnosis_prefers_slsb_and_reports_workflows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "BakaFactory SLAL Animation 78.zip"
            animations = [
                make_slsb_animation(f"Baka_Human_{index:03d}", furniture="chair" if index < 5 else "")
                for index in range(58)
            ]
            animations.extend(
                make_slsb_animation(f"Baka_Creature_{index:03d}", creature=True)
                for index in range(93)
            )

            def animation_event_rows(animation: dict) -> list[tuple[str, str]]:
                rows = []
                for actor in animation["actors"]:
                    root_name = "canine" if str(actor.get("race")).lower() == "canine" else "character"
                    for stage in actor["stages"]:
                        rows.append((root_name, stage["id"]))
                return rows

            with ZipFile(archive, "w") as source:
                source.writestr("ESP/BakaFactorySLAL.esp", b"")
                source.writestr("Fomod/ModuleConfig.xml", "<config />")
                for chunk_index in range(4):
                    chunk = animations[chunk_index * 38 : (chunk_index + 1) * 38]
                    package = converter.build_slsb_source(f"BakaFactorySynthetic{chunk_index}", "BakaFactory", chunk)
                    source.writestr(
                        f"SLSB SE/Baka/SKSE/Sexlab/Registry/Source/Baka_{chunk_index}.slsb.json",
                        json.dumps(package),
                    )
                    source.writestr(f"SLSB SE/Baka/SKSE/Sexlab/Registry/Baka_{chunk_index}.slr", b"registry")

                for chunk_index in range(8):
                    chunk = animations[chunk_index * 19 : (chunk_index + 1) * 19]
                    source.writestr(
                        f"SLAL SE/Baka/SLAnims/json/Baka_SLAL_{chunk_index}.json",
                        json.dumps({"name": f"Baka_SLAL_{chunk_index}", "animations": chunk}),
                    )
                    source.writestr(
                        f"SLAL LE/Baka/SLAnims/json/Baka_SLAL_{chunk_index}.json",
                        json.dumps({"name": f"Baka_SLAL_LE_{chunk_index}", "animations": chunk}),
                    )

                for animation in animations:
                    for root_name, event_name in animation_event_rows(animation):
                        source.writestr(
                            f"SLSB SE/Baka/meshes/actors/{root_name}/animations/BakaFactory/{event_name}.hkx",
                            f"slsb-{event_name}".encode("ascii"),
                        )
                        source.writestr(
                            f"SLAL SE/Baka/meshes/actors/{root_name}/animations/BakaFactory/{event_name}.hkx",
                            f"se-{event_name}".encode("ascii"),
                        )
                        source.writestr(
                            f"SLAL LE/Baka/meshes/actors/{root_name}/animations/BakaFactory/{event_name}.hkx",
                            f"le-{event_name}".encode("ascii"),
                        )
                for index in range(6):
                    source.writestr(
                        f"SLAL SE/Baka/meshes/actors/character/animations/BakaFactory/Unreferenced_{index}.hkx",
                        b"unused",
                    )

            with redirect_stdout(io.StringIO()):
                diagnosis = converter.diagnose_source_archive(archive, report_dir=root)
            report = diagnosis.report

            self.assertEqual(report["detectedSourceTypeCode"], "sexlabSceneBuilder")
            self.assertEqual(report["compatibilityMatch"]["profileId"], "bakafactory-slal-animation-78")
            self.assertEqual(report["selectedSourceRecordCount"], 4)
            self.assertEqual(report["sourceDetectionCounts"]["slsbSourceJsonFiles"], 4)
            self.assertEqual(report["sourceDetectionCounts"]["sexlabSlalJsonFiles"], 16)
            self.assertTrue(report["sourceSelection"]["slalJsonDetectedButNotUsed"])
            self.assertIn("Selected SLSB source JSON", report["sourceSelection"]["selectedReason"])
            self.assertEqual(report["archiveBranchLayout"]["selectedContentRoot"], "SLSB SE")
            self.assertIn("SLAL SE", report["archiveBranchLayout"]["ignoredContentRoots"])
            self.assertIn("SLAL LE", report["archiveBranchLayout"]["ignoredContentRoots"])
            self.assertTrue(report["recommendedHumanOnlyOStimOutput"])
            self.assertEqual(report["convertedScenePreviewCount"], 151)
            self.assertEqual(report["humanPreviewSceneCount"], 58)
            self.assertEqual(report["creatureScenesDetected"], 93)
            self.assertEqual(report["expectedCreatureOnlyScenes"], 93)
            self.assertEqual(report["furnitureScenesDetected"], 5)
            self.assertEqual(report["missingReferencedHkxCount"], 0)
            self.assertEqual(report["menuHubCategoryVisibility"]["totalHubLinkableScenes"], 151)
            self.assertEqual(report["menuHubCategoryVisibility"]["scenesLinkedFromGeneratedHub"], 151)
            normal = report["recommendationWorkflows"]["normalOStim"]
            self.assertTrue(normal["recommended"])
            self.assertFalse(normal["creatureRuntimeRequired"])
            self.assertEqual(normal["humanScenesRetained"], 58)
            self.assertEqual(normal["creatureMixedScenesSkipped"], 93)
            creature = report["recommendationWorkflows"]["creatureCapableOStim"]
            self.assertTrue(creature["creatureRuntimeRequired"])
            self.assertEqual(creature["creatureMixedScenesIncluded"], 93)
            duplicate_hkx = report["duplicateHkxHandling"]
            self.assertGreater(duplicate_hkx["duplicateSourceHkxCount"], 0)
            self.assertGreater(duplicate_hkx["differentContentDuplicateCount"], 0)
            self.assertLessEqual(len(duplicate_hkx["exampleRows"]), 10)
            self.assertFalse(duplicate_hkx["duplicateCreatedBehaviorConflicts"])

            self.assertTrue(diagnosis.text_report_path)
            text = diagnosis.text_report_path.read_text(encoding="utf-8")
            self.assertIn("Source Selection:", text)
            self.assertIn("Normal User Recommendation:", text)
            self.assertIn("Duplicate HKX Handling:", text)
            self.assertIn("Archive Branch Layout:", text)
            self.assertIn("SLSB source JSON files found: 4", text)
            self.assertNotIn("Expected SLAL animations: 0", text)
            self.assertNotIn(str(root), text)

    def test_diagnosis_summary_contains_support_fields(self):
        report = {
            "converterVersion": "test-version",
            "sourceArchiveName": "Pack.zip",
            "detectedSourceType": "SexLab/SLAL",
            "detectionConfidence": "High",
            "recommendedOutput": "OStim Standalone",
            "recommendedBehaviorTool": "Pandora recommended",
            "sceneOrAnimationRecordsFound": 2,
            "hkxFilesFound": 4,
            "missingReferencedHkxCount": 0,
            "creatureScenesDetected": 0,
            "furnitureScenesDetected": 1,
            "verificationPrediction": "Likely PASS",
            "compatibilityMatch": {"matchResult": "Partial", "matchedPack": "Synthetic"},
            "recommendedActionOneLine": "Build OStim Standalone ZIP.",
        }

        summary = converter.copyable_diagnosis_summary(report)

        self.assertIn("Diagnosis summary:", summary)
        self.assertIn("Known pack match: Partial - Synthetic", summary)
        self.assertIn("Recommended action: Build OStim Standalone ZIP.", summary)

    def test_nexus_bug_report_template_contains_required_support_fields(self):
        report = {
            "converterVersion": "test-version",
            "sourceArchiveName": "Source Pack.zip",
            "outputType": "OStim Standalone",
            "selectedSourceTypeLabel": "SexLab/SLAL",
            "sceneCount": 12,
            "hkxCount": 24,
            "registeredAnimationEventCount": 24,
            "missingAnimationEventCount": 0,
            "ostimMenuHubSceneCount": 1,
            "nemesisSafeOutput": True,
            "compatibilityMatch": {"matchResult": "Partial", "matchedPack": "K4-style test profile"},
        }
        verification = {
            "status": "PASS WITH WARNINGS",
            "behaviorEventsPointingToMissingHkxCount": 0,
        }

        text = converter.nexus_bug_report_text(report, verification, report_attached=True)

        required_fields = [
            "Source pack:",
            "Converter version:",
            "Output type:",
            "Behavior tool used:",
            "Mod manager:",
            "Verification result:",
            "Detected source type:",
            "Known pack match:",
            "Scenes written:",
            "HKX packaged:",
            "Behavior events generated:",
            "Scene events missing behavior registration:",
            "Behavior events missing HKX:",
            "OStim menu hub generated:",
            "Pandora-compatible behavior patch enabled:",
            "What happened in game:",
            "Report attached:",
        ]
        for field in required_fields:
            self.assertIn(field, text)
        self.assertIn("Partial - K4-style test profile", text)
        self.assertIn("Pandora-compatible behavior patch enabled: yes", text)

    def test_saved_settings_round_trip_safe_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_dir = Path(temp)
            with mock.patch.object(converter, "app_settings_dir", return_value=settings_dir):
                converter.save_app_settings(
                    {
                        "sfxFallbackAction": "ostimconvertermoan",
                        "addOStimMenuEntry": True,
                        "ostimMenuIcon": "OStim/symbols/custom",
                        "sexLabPlusExport": True,
                        "sexLabDiscoveryTags": False,
                        "nemesisSafeOutput": True,
                        "debugMode": True,
                        "lastOutputFolder": "C:/Output",
                        "sevenZipPath": "C:/Tools/7z.exe",
                        "slsbRepoPath": "C:/Repos/SLSB",
                        "sourceArchivePathShouldNotPersist": "C:/Secret/source.zip",
                    }
                )
                loaded = converter.load_app_settings()

            self.assertEqual(loaded["ostimMenuIcon"], "OStim/symbols/custom")
            self.assertEqual(loaded["sevenZipPath"], "C:/Tools/7z.exe")
            self.assertTrue(loaded["debugMode"])
            self.assertNotIn("sourceArchivePathShouldNotPersist", loaded)

    def test_can_open_path_handles_missing_paths_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "report.txt"
            existing.write_text("ok", encoding="utf-8")
            missing = Path(temp) / "missing.txt"

            self.assertTrue(converter.can_open_path(existing))
            self.assertFalse(converter.can_open_path(missing))
            self.assertFalse(converter.can_open_path(None))

    def test_gui_error_logging_writes_traceback_for_unexpected_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_dir = Path(temp)
            exc = AttributeError("'str' object has no attribute 'get'")
            tb = "Traceback (most recent call last):\n  File \"app.py\", line 1, in job\nAttributeError: 'str' object has no attribute 'get'\n"

            with mock.patch.object(converter, "app_settings_dir", return_value=settings_dir):
                log_path, summary = converter.write_internal_error_log(
                    exc,
                    tb,
                    operation="diagnosis",
                    source_archive=Path("C:/Users/Tester/Downloads/Pack.zip"),
                    debug_mode=False,
                )

            self.assertIsNotNone(log_path)
            assert log_path is not None
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("AttributeError", text)
            self.assertIn("Traceback (most recent call last)", text)
            self.assertIn("Operation: diagnosis", text)
            self.assertIn("Selected source archive: Pack.zip", text)
            self.assertIn("Error log:", summary)

    def test_release_check_validates_docs_database_and_import(self):
        release_check.check_docs()
        release_check.check_compatibility_db()
        release_check.check_version_mentions(converter.CONVERTER_VERSION)
        self.assertEqual(
            release_check.release_zip_name(converter.CONVERTER_VERSION),
            f"Adult Animation Converter {converter.CONVERTER_VERSION}.zip",
        )
        self.assertEqual(
            release_check.release_zip_name(converter.CONVERTER_VERSION, "beta"),
            f"Adult Animation Converter {converter.CONVERTER_VERSION} beta.zip",
        )
        self.assertEqual(
            release_check.release_zip_name(converter.CONVERTER_VERSION, "beta2"),
            f"Adult Animation Converter {converter.CONVERTER_VERSION} beta2.zip",
        )
        release_check.check_app_importable()

    def test_public_report_hides_full_paths_and_debug_report_can_show_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "secret" / "Pack_deploy_verify.txt"
            json_path = root / "secret" / "Pack_deploy_verify.json"
            verification = converter.ZipVerificationResult(
                zip_path=root / "Pack.zip",
                ok=True,
                report={"status": "PASS", "warnings": [], "errors": []},
                report_path=json_path,
                text_report_path=report_path,
            )
            report = {
                "status": "PASS",
                "converterVersion": converter.CONVERTER_VERSION,
                "pack": "Public Pack",
                "sourceArchiveName": "Source.zip",
                "outputType": "OStim Standalone",
                "warnings": [],
                "compatibilityMatch": {"matchResult": "None", "notes": []},
            }

            public_text = converter.conversion_report_readme_text(report, verification, debug_mode=False)
            debug_text = converter.conversion_report_readme_text(report, verification, debug_mode=True)

            self.assertNotIn(str(root), public_text)
            self.assertIn("Pack_deploy_verify.txt", public_text)
            self.assertIn(str(report_path), debug_text)

    def test_source_diagnosis_recommendations_for_common_archive_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            slal = root / "SLAL.zip"
            with ZipFile(slal, "w") as source:
                source.writestr(
                    "SLAnims/json/SLAL.json",
                    json.dumps(
                        {
                            "name": "SLAL",
                            "animations": [
                                {
                                    "id": "SlalScene",
                                    "tags": "MF",
                                    "actors": [
                                        {"type": "Male", "stages": [{"id": "SlalScene_A1_S1"}]},
                                        {"type": "Female", "stages": [{"id": "SlalScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/SLAL/SlalScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/SLAL/SlalScene_A2_S1.hkx", b"")

            flowergirls = root / "FlowerGirls Anim.zip"
            with ZipFile(flowergirls, "w") as source:
                source.writestr("meshes/actors/character/animations/FlowerGirlsSE/FNIS_FlowerGirlsSE_List.txt", "b FG_Test FG_Test.hkx\n")
                source.writestr("meshes/actors/character/animations/FlowerGirlsSE/FG_Test.hkx", b"")

            ostim = root / "OStim Native.zip"
            with ZipFile(ostim, "w") as source:
                source.writestr(
                    "SKSE/Plugins/OStim/scenes/Native/Scene.json",
                    json.dumps({"name": "Native", "speeds": [{"animation": "NativeAnim"}], "actors": [{"animationIndex": 0}]}),
                )
                source.writestr("meshes/actors/character/animations/Native/NativeAnim_0.hkx", b"")

            script_only = root / "FlowerGirls Script.zip"
            with ZipFile(script_only, "w") as source:
                source.writestr("FlowerGirls.esp", b"")
                source.writestr("scripts/FlowerGirlsQuest.pex", b"")

            slal_report = converter.diagnose_source_archive(slal, report_dir=root).report
            fg_report = converter.diagnose_source_archive(flowergirls, report_dir=root).report
            ostim_report = converter.diagnose_source_archive(ostim, report_dir=root).report
            script_report = converter.diagnose_source_archive(script_only, report_dir=root).report

            self.assertEqual(slal_report["detectedSourceTypeCode"], "sexlabSlal")
            self.assertIn("OStim Standalone", slal_report["recommendedOutput"])
            self.assertEqual(fg_report["detectedSourceTypeCode"], "flowergirlsFnis")
            self.assertEqual(ostim_report["detectedSourceTypeCode"], "ostimStandalone")
            self.assertEqual(script_report["detectedSourceTypeCode"], "unsupportedPlugin")
            self.assertTrue(script_report["containsOnlyScriptsPlugins"])

    def test_verify_rejects_menu_entry_requested_without_menu_hub(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "RequestedMenuNoHub.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "RequestedMenuNoHub",
                "length": 3,
                "speeds": [{"animation": "MenuMiss"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            metadata = {
                "schema": converter.CONVERTER_METADATA_SCHEMA,
                "schemaVersion": converter.CONVERTER_METADATA_SCHEMA_VERSION,
                "pack": {"name": "RequestedMenuNoHub", "folder": "RequestedMenuNoHub"},
                "paths": {
                    "scenesFolder": "Data/SKSE/Plugins/OStim/scenes/RequestedMenuNoHub/",
                    "behaviorList": "Data/meshes/actors/character/animations/RequestedMenuNoHub/FNIS_RequestedMenuNoHub_List.txt",
                    "pandoraInfo": "Data/Pandora_Engine/mod/RequestedMenuNoHub/info.xml",
                },
                "compatibility": {"ostimMenuEntryRequested": True},
                "deployment": {
                    "status": "NEEDS_REVIEW",
                    "sceneCount": 1,
                    "startableSceneCount": 1,
                    "categorizedStartableSceneCount": 1,
                    "hkxCount": 1,
                    "behaviorGraphHkxCount": 0,
                    "pandoraAnimDataFileCount": 2,
                    "pandoraAnimSetFileCount": 2,
                    "pandoraNamedAnimDataFileCount": 1,
                    "pandoraNamedAnimSetFileCount": 1,
                    "registeredAnimationEventCount": 1,
                    "missingAnimationEventCount": 0,
                    "missingSceneLinkCount": 0,
                    "missingNemesisPatchEventCount": 0,
                    "behaviorEventsPointingToMissingHkxCount": 0,
                    "ostimMenuEntryRequested": True,
                    "ostimMenuHubSceneCount": 0,
                },
                "checks": {},
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/RequestedMenuNoHub/Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/RequestedMenuNoHub/MenuMiss_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/RequestedMenuNoHub/FNIS_RequestedMenuNoHub_List.txt",
                    "b -Tn MenuMiss_0 MenuMiss_0.hkx\n",
                )
                archive.writestr("Data/animdata/RequestedMenuNoHub_DefaultMale.txt", "MenuMiss_0\n")
                archive.writestr("Data/animationsetdatasinglefile/RequestedMenuNoHub_DefaultMale.txt", "meshes\\actors\\character\\animations\\RequestedMenuNoHub\\MenuMiss_0.hkx\n")
                archive.writestr("Data/Pandora_Engine/mod/RequestedMenuNoHub/info.xml", "<mod><name>RequestedMenuNoHub</name><author>Tester</author></mod>")
                archive.writestr("Data/Pandora_Engine/mod/RequestedMenuNoHub/animationdata/DefaultMale.txt", "MenuMiss_0\n")
                archive.writestr("Data/Pandora_Engine/mod/RequestedMenuNoHub/animationsetdata/DefaultMale.txt", "meshes\\actors\\character\\animations\\RequestedMenuNoHub\\MenuMiss_0.hkx\n")
                archive.writestr("Data/Nemesis_Engine/mod/requestedmenu/info.ini", "name=RequestedMenuNoHub\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr("Data/Nemesis_Engine/mod/requestedmenu/0_master/#0106.txt", "<hkobject><hkcstring>MenuMiss_0</hkcstring></hkobject>\n")
                archive.writestr("Data/SKSE/Plugins/OStim/converter_metadata/RequestedMenuNoHub/metadata.json", json.dumps(metadata))

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("Add OStim menu entry was requested" in error for error in verification.report["errors"]))

    def test_verify_rejects_conflicting_behavior_event_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "ConflictingEvents.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "ConflictingEvents",
                "length": 3,
                "speeds": [{"animation": "Conflict"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/ConflictingEvents/Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/ConflictingEvents/Conflict_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/ConflictingEvents/Other_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/ConflictingEvents/FNIS_ConflictingEvents_List.txt",
                    "b -Tn Conflict_0 Conflict_0.hkx\nb -Tn Conflict_0 Other_0.hkx\n",
                )
                archive.writestr("Data/animdata/ConflictingEvents_DefaultMale.txt", "Conflict_0\n")
                archive.writestr("Data/animationsetdatasinglefile/ConflictingEvents_DefaultMale.txt", "meshes\\actors\\character\\animations\\ConflictingEvents\\Conflict_0.hkx\n")
                archive.writestr("Data/Pandora_Engine/mod/ConflictingEvents/info.xml", "<mod><name>ConflictingEvents</name><author>Tester</author></mod>")
                archive.writestr("Data/Pandora_Engine/mod/ConflictingEvents/animationdata/DefaultMale.txt", "Conflict_0\n")
                archive.writestr("Data/Pandora_Engine/mod/ConflictingEvents/animationsetdata/DefaultMale.txt", "meshes\\actors\\character\\animations\\ConflictingEvents\\Conflict_0.hkx\n")
                archive.writestr("Data/Nemesis_Engine/mod/conflicting/info.ini", "name=ConflictingEvents\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr("Data/Nemesis_Engine/mod/conflicting/0_master/#0106.txt", "<hkobject><hkcstring>Conflict_0</hkcstring></hkobject>\n")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("registered to multiple HKX files" in error for error in verification.report["errors"]))

    def test_verify_accepts_cross_root_duplicate_event_names_when_actor_root_resolves(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "CrossRootCreature.zip"
            pack = "CrossRootCreature"
            human_scene = {
                "name": "Human Shared",
                "modpack": pack,
                "length": 3,
                "speeds": [{"animation": "Shared"}],
                "actors": [
                    {"type": "male", "intendedSex": "male", "animationIndex": 0},
                    {"type": "female", "intendedSex": "female", "animationIndex": 1},
                ],
                "actions": [{"type": "kissing", "actor": 0, "target": 1}],
            }
            creature_scene = {
                "name": "Creature Shared",
                "modpack": pack,
                "length": 3,
                "speeds": [{"animation": "Shared"}],
                "actors": [
                    {"type": "male", "intendedSex": "male", "animationIndex": 0},
                    {"type": "creaturemale", "tags": ["creature"], "creatureRace": "Dog", "animationIndex": 1},
                ],
                "actions": [{"type": "analsex", "actor": 0, "target": 1}],
                "tags": ["creature"],
            }
            nemesis_code = converter.safe_nemesis_patch_code(pack)
            speed_var = f"{converter.sanitize_name(nemesis_code, 'ConvertedPack').upper()}_AnimationSpeed"

            with ZipFile(zip_path, "w") as archive:
                archive.writestr(f"Data/SKSE/Plugins/OStim/scenes/{pack}/Human.json", json.dumps(human_scene))
                archive.writestr(f"Data/SKSE/Plugins/OStim/scenes/{pack}/Creature.json", json.dumps(creature_scene))
                archive.writestr(f"Data/meshes/actors/character/animations/{pack}/Shared_0.hkx", b"")
                archive.writestr(f"Data/meshes/actors/character/animations/{pack}/Shared_1.hkx", b"")
                archive.writestr(f"Data/meshes/actors/canine/animations/{pack}/Shared_1.hkx", b"")
                archive.writestr(
                    f"Data/meshes/actors/character/animations/{pack}/ATT_{nemesis_code}_animlist.txt",
                    f"b -Tn Shared_0 Shared_0.hkx {speed_var}: 1\n"
                    f"b -Tn Shared_1 Shared_1.hkx {speed_var}: 1\n",
                )
                archive.writestr(
                    f"Data/meshes/actors/canine/animations/{pack}/FNIS_{pack}_canine_List.txt",
                    "b -Tn Shared_1 Shared_1.hkx\n",
                )
                archive.writestr(
                    f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini",
                    f"name={pack}\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    f"Data/Nemesis_Engine/mod/{nemesis_code}/0_master/#0106.txt",
                    "<hkobject><hkcstring>Shared_0</hkcstring><hkcstring>Shared_1</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(zip_path, write_report=False)

            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE_WITH_CREATURE)
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)
            self.assertEqual(verification.report["duplicateBehaviorRegistrationRiskCount"], 0)
            self.assertEqual(verification.report["sameRootHkxEventConflictCount"], 0)
            self.assertGreaterEqual(verification.report["crossRootDuplicateBehaviorEventCount"], 1)
            actor_rows = [
                row
                for scene in verification.report["scenes"]
                for row in scene.get("expectedActorRootEvents", [])
                if row.get("event") == "Shared_1"
            ]
            self.assertIn("character", {row.get("actorRoot") for row in actor_rows})
            self.assertIn("canine", {row.get("actorRoot") for row in actor_rows})

    def test_verify_rejects_same_root_duplicate_hkx_event_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "SameRootHkxConflict.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "SameRootHkxConflict",
                "length": 3,
                "speeds": [{"animation": "Conflict"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/SameRootHkxConflict/Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/SameRootHkxConflict/Conflict_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/SameRootHkxConflict/Alt/Conflict_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/SameRootHkxConflict/ATT_sameroot_animlist.txt",
                    "b -Tn Conflict_0 Conflict_0.hkx SAMEROOT_AnimationSpeed: 1\n",
                )
                archive.writestr("Data/Nemesis_Engine/mod/sameroot/info.ini", "name=SameRootHkxConflict\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n")
                archive.writestr("Data/Nemesis_Engine/mod/sameroot/0_master/#0106.txt", "<hkobject><hkcstring>Conflict_0</hkcstring></hkobject>\n")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["sameRootHkxEventConflictCount"], 1)
            self.assertTrue(any("multiple HKX files under actor root 'character'" in error for error in verification.report["errors"]))

    def test_archive_can_build_sexlab_slal_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SexLab Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Ace.json",
                    json.dumps(
                        {
                            "name": "Ace Test",
                            "animations": [
                                {
                                    "id": "Ace_TestFootjob",
                                    "name": "Ace Test Footjob",
                                    "tags": "MF,Footjob",
                                    "sound": "Squishing",
                                    "actors": [
                                        {
                                            "type": "Female",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A1_S1"},
                                                {"id": "Ace_TestFootjob_A1_S2"},
                                            ],
                                        },
                                        {
                                            "type": "Male",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A2_S1"},
                                                {"id": "Ace_TestFootjob_A2_S2"},
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S2.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S2.hkx", b"")
                source.writestr("meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx", b"old behavior")

            result = converter.convert_archive_to_sexlab_zip(archive, mod_author="Ace")
            self.assertEqual(result.report["target"], "SexLab/SLAL")
            self.assertEqual(result.report["animationCount"], 1)
            self.assertEqual(result.report["stageCount"], 2)
            self.assertFalse(result.report["sexlabDiscoveryTagsEnabled"])
            self.assertEqual(result.report["sexlabDiscoveryTags"], [])
            self.assertEqual(len(result.hkx_assets), 4)
            self.assertEqual(result.report["behaviorGraphHkxCount"], 2)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                slal = json.loads(converted.read(mod_entry("Data/SLAnims/json/SexLab_Pack.json")).decode("utf-8"))
                source_text = converted.read(mod_entry("Data/SLAnims/source/SexLab_Pack.txt")).decode("utf-8")
                fnis_list = converted.read(
                    mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/FNIS_SexLab_Pack_List.txt")
                ).decode("utf-8")
                report_text = converted.read("sexlab_conversion_report.txt").decode("utf-8")
                readme_text = converted.read("README_SexLab_SLAL.txt").decode("utf-8")
            with _ZipFile(result.zip_path) as converted:
                raw_names = set(converted.namelist())

            self.assertIn(mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/Ace_TestFootjob_A1_S1.hkx"), names)
            self.assertIn(mod_entry("Data/meshes/actors/character/animations/SexLab_Pack/Ace_TestFootjob_A2_S2.hkx"), names)
            self.assertIn(mod_entry("Data/meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx"), names)
            self.assertIn(mod_entry("Data/meshes/actors/character/behaviors/FNIS_SexLab_Pack_Behavior.hkx"), names)
            self.assertIn("SLAnims/json/SexLab_Pack.json", raw_names)
            self.assertNotIn("Data/SLAnims/json/SexLab_Pack.json", raw_names)
            self.assertEqual(slal["author"], "Ace")
            animation = slal["animations"][0]
            self.assertEqual(animation["id"], "Ace_TestFootjob")
            self.assertEqual(animation["actors"][0]["type"], "Female")
            self.assertEqual(animation["actors"][1]["type"], "Male")
            self.assertEqual(animation["actors"][0]["stages"][1]["id"], "Ace_TestFootjob_A1_S2")
            self.assertIn("Footjob", animation["tags"])
            self.assertNotIn("AdultAnimationConverter", animation["tags"])
            self.assertNotIn("Converted", animation["tags"])
            self.assertNotIn("SexLab_Pack", animation["tags"])
            self.assertIn("Animation(", source_text)
            self.assertNotIn("common_tags", source_text)
            self.assertIn("actor1=Female()", source_text)
            self.assertIn("' Ace_TestFootjob", fnis_list)
            self.assertIn("s Ace_TestFootjob_A1_S1 Ace_TestFootjob_A1_S1.hkx", fnis_list)
            self.assertIn("+ Ace_TestFootjob_A1_S2 Ace_TestFootjob_A1_S2.hkx", fnis_list)
            self.assertIn("SexLab/SLAL Conversion Report", report_text)
            self.assertEqual(result.report["pandoraRegistrationMode"], "fnis_auto_discovery")
            self.assertFalse(result.report["pandoraPatchListEntryExpected"])
            self.assertEqual(result.report["pandoraExpectedFnisModNames"], ["FNIS_SexLab_Pack_List"])
            self.assertIn("no converted-pack checkbox is expected", readme_text)
            self.assertIn('"FNIS Mod" entry matching: FNIS_SexLab_Pack_List', readme_text)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["pandoraRegistrationMode"], "fnis_auto_discovery")
            self.assertFalse(verification.report["pandoraPatchListEntryExpected"])
            self.assertEqual(verification.report["pandoraExpectedFnisModNames"], ["FNIS_SexLab_Pack_List"])
            self.assertEqual(verification.report["type"], "sexlabDeploymentVerification")
            self.assertEqual(verification.report["target"], "SexLab/SLAL")
            self.assertEqual(verification.report["dataRoots"], [converter.ARCHIVE_ROOT_DATA_PREFIX])
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 2)
            self.assertEqual(verification.report["missingHkxEventCount"], 0)
            self.assertEqual(verification.report["missingFnisEventCount"], 0)

            cli_zip = root / "cli_sexlab.zip"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--target", "sexlab", "--mod-archive", str(archive), "--zip-out", str(cli_zip)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(cli_zip.exists())
            self.assertIn("SexLab/SLAL animation", stdout.getvalue())
            self.assertIn("SexLab discovery tags", stdout.getvalue())

            enabled = converter.convert_archive_to_sexlab_zip(archive, sexlab_discovery_tags=True)
            with ZipFile(enabled.zip_path) as converted:
                enabled_slal = json.loads(converted.read(mod_entry("Data/SLAnims/json/SexLab_Pack.json")).decode("utf-8"))
                enabled_source = converted.read(mod_entry("Data/SLAnims/source/SexLab_Pack.txt")).decode("utf-8")
            self.assertTrue(enabled.report["sexlabDiscoveryTagsEnabled"])
            self.assertIn("SexLab_Pack", enabled.report["sexlabDiscoveryTags"])
            self.assertIn("AdultAnimationConverter", enabled_slal["animations"][0]["tags"])
            self.assertIn('common_tags("AdultAnimationConverter,Converted,SexLab_Pack")', enabled_source)

    def test_archive_can_build_sexlab_plus_registry_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SexLab Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Ace.json",
                    json.dumps(
                        {
                            "name": "Ace Test",
                            "animations": [
                                {
                                    "id": "Ace_TestFootjob",
                                    "name": "Ace Test Footjob",
                                    "tags": "MF,Footjob",
                                    "sound": "Squishing",
                                    "actors": [
                                        {
                                            "type": "Female",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A1_S1"},
                                                {"id": "Ace_TestFootjob_A1_S2"},
                                            ],
                                        },
                                        {
                                            "type": "Male",
                                            "stages": [
                                                {"id": "Ace_TestFootjob_A2_S1"},
                                                {"id": "Ace_TestFootjob_A2_S2"},
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S2.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S2.hkx", b"")
                source.writestr("meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx", b"old behavior")

            result = converter.convert_archive_to_sexlab_zip(archive, mod_author="Ace", sexlab_plus=True)
            self.assertEqual(result.report["target"], "SexLab P+/SLSB")
            self.assertTrue(result.report["sexlabPlusExport"])
            self.assertEqual(result.report["slsbSourceFileCount"], 1)
            self.assertEqual(result.report["sexlabPlusCompiledRegistryCount"], 1)
            self.assertTrue(result.report["readyForSexLabPlusRuntime"])
            self.assertEqual(result.report["behaviorGraphHkxCount"], 2)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                slsb = json.loads(converted.read("Data/SKSE/Sexlab/Registry/Source/SexLab_Pack.slsb.json").decode("utf-8"))
                registry = converted.read("Data/SKSE/Sexlab/Registry/SexLab_Pack.slr")
                fnis_list = converted.read(
                    "Data/meshes/actors/character/animations/SexLab_Pack/FNIS_SexLab_Pack_List.txt"
                ).decode("utf-8")
                report_text = converted.read("sexlab_conversion_report.txt").decode("utf-8")
            with _ZipFile(result.zip_path) as converted:
                raw_names = set(converted.namelist())

            self.assertIn("Data/SLAnims/json/SexLab_Pack.json", names)
            self.assertIn("Data/SKSE/Sexlab/Registry/SexLab_Pack.slr", names)
            self.assertIn("Data/meshes/actors/character/behaviors/FNIS_Ace_Test_Behavior.hkx", names)
            self.assertIn("Data/meshes/actors/character/behaviors/FNIS_SexLab_Pack_Behavior.hkx", names)
            self.assertIn("SKSE/Sexlab/Registry/SexLab_Pack.slr", raw_names)
            self.assertNotIn("Data/SKSE/Sexlab/Registry/SexLab_Pack.slr", raw_names)
            self.assertGreater(len(registry), 32)
            self.assertEqual(registry[0], converter.SLSB_PROJECT_VERSION)
            self.assertEqual(slsb["version"], converter.SLSB_PROJECT_VERSION)
            self.assertEqual(slsb["pack_author"], "Ace")
            self.assertFalse(result.report["sexlabDiscoveryTagsEnabled"])
            scene = next(iter(slsb["scenes"].values()))
            self.assertEqual(scene["name"], "Ace Test Footjob")
            self.assertEqual(len(scene["positions"]), 2)
            self.assertEqual(scene["positions"][0]["race"], "Human")
            self.assertEqual(scene["stages"][0]["positions"][0]["event"], ["Ace_TestFootjob_A1_S1"])
            self.assertIn("footjob", scene["stages"][0]["tags"])
            self.assertNotIn("adultanimationconverter", scene["stages"][0]["tags"])
            self.assertNotIn("converted", scene["stages"][0]["tags"])
            self.assertNotIn("sexlab_pack", scene["stages"][0]["tags"])
            prefix = result.report["sexLabPlusPrefixHash"]
            self.assertIn("s Ace_TestFootjob_A1_S1 Ace_TestFootjob_A1_S1.hkx", fnis_list)
            self.assertIn(f"b {prefix}Ace_TestFootjob_A1_S1 Ace_TestFootjob_A1_S1.hkx", fnis_list)
            self.assertIn("SexLab P+/SLSB Conversion Report", report_text)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["type"], "sexlabDeploymentVerification")
            self.assertEqual(verification.report["target"], "SexLab P+/SLSB")
            self.assertTrue(verification.report["readyForSexLabPlusRuntime"])
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 2)
            self.assertEqual(verification.report["missingHkxEventCount"], 0)
            self.assertEqual(verification.report["missingFnisEventCount"], 0)

            cli_zip = root / "cli_sexlabplus.zip"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--target", "sexlabplus", "--mod-archive", str(archive), "--zip-out", str(cli_zip)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(cli_zip.exists())
            self.assertIn("SexLab P+/SLSB", stdout.getvalue())

    def test_ostim_to_sexlab_uses_native_mf_actor_order_and_remaps_hkx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Native OStim.zip"
            scene = {
                "name": "Native Vaginal",
                "modpack": "NativePack",
                "length": 2.0,
                "speeds": [{"animation": "Loop"}],
                "actors": [
                    {"animationIndex": 0, "intendedSex": "male"},
                    {"animationIndex": 1, "intendedSex": "female"},
                ],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(archive, scene, {"Loop_0": b"male-slot", "Loop_1": b"female-slot"})

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Native_OStim.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Native_OStim.txt").decode("utf-8")
                female_hkx = converted.read("Data/meshes/actors/character/animations/Native_OStim/Scene_A1_S1.hkx")
                male_hkx = converted.read("Data/meshes/actors/character/animations/Native_OStim/Scene_A2_S1.hkx")

            animation = slal["animations"][0]
            self.assertEqual([actor["type"] for actor in animation["actors"]], ["Female", "Male"])
            self.assertFalse(any("_slsbSex" in actor for actor in animation["actors"]))
            self.assertEqual(animation["actors"][0]["stages"][0]["id"], "Scene_A1_S1")
            self.assertEqual(animation["actors"][1]["stages"][0]["id"], "Scene_A2_S1")
            self.assertEqual(female_hkx, b"female-slot")
            self.assertEqual(male_hkx, b"male-slot")
            self.assertEqual(animation["actors"][0]["add_cum"], "Vaginal")
            self.assertEqual(animation["actors"][1]["stages"][0]["sos"], 7)
            self.assertIn("actor1=Female(add_cum=Vaginal)", source_text)
            self.assertIn("a2_stage_params", source_text)
            self.assertIn("Stage(1, sos=7)", source_text)
            self.assertEqual(result.report["sexLabActorOrderChangedCount"], 1)
            self.assertEqual(result.report["sexLabCumMetadataCount"], 1)
            self.assertEqual(result.report["sexLabSosMetadataCount"], 1)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["maleFirstMfAnimationCount"], 0)

    def test_ostim_to_sexlab_discovery_tags_are_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Clean Tags.zip"
            scene = {
                "name": "Clean Oral",
                "modpack": "NativePack",
                "speeds": [{"animation": "OralLoop"}],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "blowjob", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(archive, scene, {"OralLoop_0": b"", "OralLoop_1": b""})

            clean = converter.convert_archive_to_sexlab_zip(archive)
            enabled = converter.convert_archive_to_sexlab_zip(archive, sexlab_discovery_tags=True)

            with ZipFile(clean.zip_path) as converted:
                clean_slal = json.loads(converted.read("Data/SLAnims/json/Clean_Tags.json").decode("utf-8"))
                clean_source = converted.read("Data/SLAnims/source/Clean_Tags.txt").decode("utf-8")
            with ZipFile(enabled.zip_path) as converted:
                enabled_slal = json.loads(converted.read("Data/SLAnims/json/Clean_Tags.json").decode("utf-8"))
                enabled_source = converted.read("Data/SLAnims/source/Clean_Tags.txt").decode("utf-8")

            self.assertFalse(clean.report["sexlabDiscoveryTagsEnabled"])
            self.assertNotIn("AdultAnimationConverter", clean_slal["animations"][0]["tags"])
            self.assertNotIn("Converted", clean_slal["animations"][0]["tags"])
            self.assertNotIn("Clean_Tags", clean_slal["animations"][0]["tags"])
            self.assertNotIn("common_tags", clean_source)
            self.assertTrue(enabled.report["sexlabDiscoveryTagsEnabled"])
            self.assertIn("AdultAnimationConverter", enabled_slal["animations"][0]["tags"])
            self.assertIn('common_tags("AdultAnimationConverter,Converted,Clean_Tags")', enabled_source)

    def test_ostim_to_sexlab_source_uses_prefixes_without_double_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Drago Pack.zip"
            scene = {
                "name": "DrG Missionary",
                "modpack": "NativePack",
                "speeds": [{"animation": "DrGLoop"}],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(
                archive,
                scene,
                {"DrGLoop_0": b"", "DrGLoop_1": b""},
                scene_name="DrG_Missionary",
            )

            result = converter.convert_archive_to_sexlab_zip(archive, sexlab_discovery_tags=True)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Drago_Pack.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Drago_Pack.txt").decode("utf-8")

            animation = slal["animations"][0]
            self.assertEqual(animation["id"], "DrG_Missionary")
            self.assertIn('anim_id_prefix("DrG_")', source_text)
            self.assertIn('anim_name_prefix("DrG ")', source_text)
            self.assertIn('id="Missionary"', source_text)
            self.assertNotIn('id="DrG_Missionary"', source_text)
            self.assertIn('name="Missionary"', source_text)
            self.assertIn('common_tags("AdultAnimationConverter,Converted,Drago_Pack")', source_text)
            self.assertNotRegex(source_text, r'tags="[^"]*AdultAnimationConverter')
            quality = result.report["sexLabSourceFnisQuality"]
            self.assertEqual(quality["animIdPrefix"], "DrG_")
            self.assertEqual(quality["animationIdsStrippedCount"], 1)
            self.assertEqual(quality["doublePrefixRiskFixedCount"], 1)
            self.assertEqual(quality["sourceFinalIdMismatchCount"], 0)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["sexLabSourceFnisQuality"]["doublePrefixRiskCount"], 0)
            self.assertEqual(verification.report["sexLabSourceFnisQuality"]["commonTagOverlapCount"], 0)

    def test_ostim_to_sexlab_collapses_repeated_sos_and_animvars_in_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Param Style.zip"
            scene = {
                "name": "Param Style",
                "modpack": "NativePack",
                "speeds": [{"animation": "Slow"}, {"animation": "Fast"}],
                "actors": [
                    {"animationIndex": 0, "intendedSex": "male", "feetOnGround": False},
                    {"animationIndex": 1, "intendedSex": "female"},
                ],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(
                archive,
                scene,
                {"Slow_0": b"", "Slow_1": b"", "Fast_0": b"", "Fast_1": b""},
            )

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                source_text = converted.read("Data/SLAnims/source/Param_Style.txt").decode("utf-8")

            self.assertIn("a2_stage_params", source_text)
            self.assertIn('Stage(1, sos=7, animvars="AVbHumanoidFootIKDisable")', source_text)
            self.assertNotIn("AVbHumanoidFootIKDisable=1", source_text)
            self.assertNotIn("Stage(2, sos=7", source_text)
            quality = result.report["sexLabSourceFnisQuality"]
            self.assertEqual(quality["sosCollapsedCount"], 1)
            self.assertEqual(quality["animvarCollapsedCount"], 1)
            self.assertEqual(quality["actorStageParamCollapsedCount"], 1)
            self.assertGreaterEqual(quality["animvarNormalizedCount"], 1)

    def test_ostim_to_sexlab_skips_invalid_single_hkx_scene_without_fnis_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Mixed Validity.zip"
            valid_scene = {
                "name": "Valid Scene",
                "modpack": "NativePack",
                "speeds": [{"animation": "ValidLoop"}],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            invalid_scene = {
                "name": "Internal Idle",
                "modpack": "NativePack",
                "speeds": [{"animation": "IdleLoop"}],
                "actors": [{"animationIndex": 0}],
                "tags": ["internal"],
            }
            with ZipFile(archive, "w") as source:
                source.writestr("Data/SKSE/Plugins/OStim/scenes/NativePack/ValidScene.json", json.dumps(valid_scene))
                source.writestr("Data/SKSE/Plugins/OStim/scenes/NativePack/InternalIdle.json", json.dumps(invalid_scene))
                source.writestr("Data/meshes/actors/character/animations/NativePack/ValidLoop_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/NativePack/ValidLoop_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/NativePack/IdleLoop_0.hkx", b"")

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Mixed_Validity.json").decode("utf-8"))
                fnis_list = converted.read(
                    "Data/meshes/actors/character/animations/Mixed_Validity/FNIS_Mixed_Validity_List.txt"
                ).decode("utf-8")

            self.assertEqual([animation["id"] for animation in slal["animations"]], ["ValidScene"])
            self.assertIn("ValidScene_A1_S1", fnis_list)
            self.assertNotIn("InternalIdle", fnis_list)
            self.assertNotIn("IdleLoop", fnis_list)
            self.assertEqual(result.report["sexLabSkippedInvalidSceneCount"], 1)
            self.assertEqual(result.report["sexLabSourceFnisQuality"]["skippedInvalidSceneCount"], 1)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["sexLabSourceFnisQuality"]["fnisDuplicateEventCount"], 0)
            self.assertEqual(verification.report["sexLabSourceFnisQuality"]["fnisOrderErrorCount"], 0)

    def test_ostim_to_sexlab_collapses_sound_and_omits_default_timers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Sound Style.zip"
            scene = {
                "name": "Sound Style",
                "modpack": "NativePack",
                "length": 2.0,
                "speeds": [{"animation": "Loop", "sound": "Squishing"}],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(archive, scene, {"Loop_0": b"", "Loop_1": b""})

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Sound_Style.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Sound_Style.txt").decode("utf-8")

            animation = slal["animations"][0]
            self.assertEqual(animation["sound"], "Squishing")
            self.assertNotIn("sound", animation["stages"][0])
            self.assertNotIn("timer", animation["stages"][0])
            self.assertNotIn("timer=2.000", source_text)
            self.assertNotIn("    stage_params=[", source_text)
            self.assertEqual(result.report["stageSoundEntryCount"], 0)
            self.assertEqual(result.report["timerEntryCount"], 0)
            self.assertEqual(result.report["sexLabRedundantStageSoundRemovedCount"], 1)

    def test_ostim_to_sexlab_keeps_real_stage_sound_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Sound Override.zip"
            scene = {
                "name": "Sound Override",
                "modpack": "NativePack",
                "speeds": [
                    {"animation": "Slow", "sound": "Squishing"},
                    {"animation": "Oral", "sound": "Sucking"},
                ],
                "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(
                archive,
                scene,
                {"Slow_0": b"", "Slow_1": b"", "Oral_0": b"", "Oral_1": b""},
            )

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Sound_Override.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Sound_Override.txt").decode("utf-8")

            stages = slal["animations"][0]["stages"]
            self.assertNotIn("sound", stages[0])
            self.assertEqual(stages[1]["sound"], "Sucking")
            self.assertIn("stage_params", source_text)
            self.assertIn("Stage(2, sound=Sucking)", source_text)
            self.assertEqual(result.report["sexLabStageLevelSoundOverrideCount"], 1)

    def test_ostim_to_sexlab_preserves_ff_mm_and_creature_order(self):
        ff_scene = {
            "name": "FF Scene",
            "modpack": "NativePack",
            "tags": ["FF"],
            "speeds": [{"animation": "FFLoop"}],
            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
            "actions": [{"type": "kissing", "actor": 0, "target": 1}],
        }
        mm_scene = {
            "name": "MM Scene",
            "modpack": "NativePack",
            "tags": ["MM"],
            "speeds": [{"animation": "MMLoop"}],
            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
            "actions": [{"type": "kissing", "actor": 0, "target": 1}],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ff_archive = root / "FF Pack.zip"
            mm_archive = root / "MM Pack.zip"
            write_ostim_scene_pack(ff_archive, ff_scene, {"FFLoop_0": b"", "FFLoop_1": b""})
            write_ostim_scene_pack(mm_archive, mm_scene, {"MMLoop_0": b"", "MMLoop_1": b""})

            ff_result = converter.convert_archive_to_sexlab_zip(ff_archive)
            mm_result = converter.convert_archive_to_sexlab_zip(mm_archive)

            with ZipFile(ff_result.zip_path) as converted:
                ff_slal = json.loads(converted.read("Data/SLAnims/json/FF_Pack.json").decode("utf-8"))
            with ZipFile(mm_result.zip_path) as converted:
                mm_slal = json.loads(converted.read("Data/SLAnims/json/MM_Pack.json").decode("utf-8"))

            self.assertEqual([actor["type"] for actor in ff_slal["animations"][0]["actors"]], ["Female", "Female"])
            self.assertEqual([actor["type"] for actor in mm_slal["animations"][0]["actors"]], ["Male", "Male"])
            self.assertEqual(ff_result.report["sexLabActorOrderChangedCount"], 0)
            self.assertEqual(mm_result.report["sexLabActorOrderChangedCount"], 0)

    def test_ostim_to_sexlab_plus_uses_native_style_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Native Plus.zip"
            scene = {
                "name": "Native Plus",
                "modpack": "NativePack",
                "speeds": [{"animation": "PlusLoop"}],
                "actors": [
                    {"animationIndex": 0, "intendedSex": "male"},
                    {"animationIndex": 1, "intendedSex": "female"},
                ],
                "actions": [{"type": "analsex", "actor": 0, "target": 1}],
            }
            write_ostim_scene_pack(archive, scene, {"PlusLoop_0": b"", "PlusLoop_1": b""})

            result = converter.convert_archive_to_sexlab_zip(archive, sexlab_plus=True)

            with ZipFile(result.zip_path) as converted:
                slsb = json.loads(converted.read("Data/SKSE/Sexlab/Registry/Source/Native_Plus.slsb.json").decode("utf-8"))
                slal = json.loads(converted.read("Data/SLAnims/json/Native_Plus.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Native_Plus.txt").decode("utf-8")

            animation = slal["animations"][0]
            self.assertEqual([actor["type"] for actor in animation["actors"]], ["Female", "Male"])
            scene_node = next(iter(slsb["scenes"].values()))
            self.assertEqual(scene_node["positions"][0]["sex"], {"male": False, "female": True, "futa": False})
            self.assertEqual(scene_node["positions"][1]["sex"], {"male": True, "female": False, "futa": False})
            self.assertEqual(scene_node["stages"][0]["positions"][1]["schlong"], 7)
            self.assertEqual(scene_node["stages"][0]["extra"]["fixed_len"], 0.0)
            self.assertNotIn("adultanimationconverter", scene_node["stages"][0]["tags"])
            self.assertNotIn("timer=2.000", source_text)
            self.assertTrue(result.report["readyForSexLabPlusRuntime"])

    def test_verify_rejects_sexlab_zip_with_unregistered_stage_hkx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Broken_SexLab.zip"
            with _ZipFile(archive, "w") as converted:
                converted.writestr(
                    "SLAnims/json/Broken.json",
                    json.dumps(
                        {
                            "name": "Broken",
                            "animations": [
                                {
                                    "id": "BrokenScene",
                                    "name": "Broken Scene",
                                    "tags": "MF,Sex",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "BrokenScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "BrokenScene_A2_S1"}]},
                                    ],
                                    "stages": [{"number": 1, "timer": 6.0, "sound": "Squishing"}],
                                }
                            ],
                        }
                    ),
                )
                converted.writestr("SLAnims/source/Broken.txt", "Animation(id=\"BrokenScene\")\n")
                converted.writestr("meshes/actors/character/animations/Broken/BrokenScene_A1_S1.hkx", b"")
                converted.writestr(
                    "meshes/actors/character/animations/Broken/FNIS_Broken_List.txt",
                    "Version V1.0\n\ns BrokenScene_A1_S1 BrokenScene_A1_S1.hkx\n",
                )

            verification = converter.verify_converted_zip(archive, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["type"], "sexlabDeploymentVerification")
            self.assertEqual(verification.report["missingHkxEventCount"], 1)
            self.assertEqual(verification.report["missingFnisEventCount"], 1)
            self.assertTrue(
                any("BrokenScene_A2_S1" in error for error in verification.report["errors"]),
                verification.report["errors"],
            )

    def test_slsb_source_only_archive_converts_to_ostim_and_sexlab_plus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SLSB Source Pack.zip"
            slsb_source = {
                "version": converter.SLSB_PROJECT_VERSION,
                "pack_name": "Ace SLSB Source",
                "pack_author": "Ace",
                "prefix_hash": "abcd",
                "scenes": {
                    "scene001": {
                        "id": "scene001",
                        "name": "Ace Source Footjob",
                        "stages": [
                            {
                                "id": "stage001",
                                "name": "",
                                "positions": [
                                    {
                                        "sex": {"male": False, "female": True, "futa": False},
                                        "race": "Human",
                                        "event": ["AceSource_A1_S1"],
                                        "scale": 1.0,
                                        "extra": {"submissive": False, "vampire": False, "climax": False, "dead": False, "custom": []},
                                        "offset": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0},
                                        "anim_obj": "",
                                        "strip_data": {"default": True, "everything": False, "nothing": False, "helmet": False, "gloves": False, "boots": False},
                                    },
                                    {
                                        "sex": {"male": True, "female": False, "futa": True},
                                        "race": "Human",
                                        "event": ["AceSource_A2_S1"],
                                        "scale": 1.0,
                                        "extra": {"submissive": False, "vampire": False, "climax": False, "dead": False, "custom": []},
                                        "offset": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0},
                                        "anim_obj": "",
                                        "strip_data": {"default": True, "everything": False, "nothing": False, "helmet": False, "gloves": False, "boots": False},
                                    },
                                ],
                                "tags": ["footjob", "squishing"],
                                "extra": {"fixed_len": 7.0, "nav_text": ""},
                            }
                        ],
                        "root": "stage001",
                        "graph": {"stage001": {"dest": [], "x": 40.0, "y": 40.0}},
                        "furniture": {"furni_types": ["None"], "allow_bed": False, "offset": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0}},
                        "private": False,
                        "has_warnings": False,
                    }
                },
            }
            with ZipFile(archive, "w") as source:
                source.writestr("SKSE/Sexlab/Registry/Source/Ace_Source.slsb.json", json.dumps(slsb_source))
                source.writestr("meshes/actors/character/animations/Ace_Source/AceSource_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Source/AceSource_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 1)
            self.assertEqual(result.scenes[0].name, "Ace Source Footjob")
            self.assertEqual(result.scenes[0].speeds[0].sexlab_sound, "Squishing")
            self.assertTrue(any(action.type == "footjob" for action in result.scenes[0].actions))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
            self.assertIn("Data/meshes/actors/character/animations/SLSB_Source_Pack/scene001_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/SLSB_Source_Pack/scene001_S1_1.hkx", names)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

            plus_result = converter.convert_archive_to_sexlab_zip(archive, sexlab_plus=True)
            self.assertEqual(plus_result.report["target"], "SexLab P+/SLSB")
            self.assertEqual(plus_result.report["slsbSourceFileCount"], 1)
            self.assertEqual(plus_result.report["sexlabPlusCompiledRegistryCount"], 1)
            with ZipFile(plus_result.zip_path) as converted:
                names = set(converted.namelist())
                roundtrip_slsb = json.loads(
                    converted.read("Data/SKSE/Sexlab/Registry/Source/SLSB_Source_Pack.slsb.json").decode("utf-8")
                )
            self.assertIn("Data/SKSE/Sexlab/Registry/SLSB_Source_Pack.slr", names)
            self.assertIn("Data/SKSE/Sexlab/Registry/Source/SLSB_Source_Pack.slsb.json", names)
            roundtrip_scene = next(iter(roundtrip_slsb["scenes"].values()))
            self.assertEqual(roundtrip_scene["positions"][1]["sex"], {"male": True, "female": False, "futa": True})

    def test_sexlab_export_writes_action_based_sfx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "OSex Blowjob.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/BJ/Blow.xml",
                    """\
<scene id="AA|Standing|BJ|Blow" actors="2">
  <info name="Blow Test" />
  <anim id="BlowLoop" l="6" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/BJ/BlowLoop_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/BJ/BlowLoop_1.hkx", b"")

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/OSex_Blowjob.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/OSex_Blowjob.txt").decode("utf-8")
                report = json.loads(converted.read("sexlab_conversion_report.json").decode("utf-8"))

            animation = slal["animations"][0]
            self.assertEqual(animation["sound"], "Sucking")
            self.assertNotIn("sound", animation["stages"][0])
            self.assertIn("sound=Sucking", source_text)
            self.assertNotIn("    stage_params=[", source_text)
            self.assertNotIn('sound="Sucking"', source_text)
            self.assertEqual(report["soundProfileCount"], 1)
            self.assertEqual(report["stageSoundEntryCount"], 0)
            self.assertTrue(report["usesSexLabSfx"])

    def test_sexlab_export_uses_sfx_fallback_for_uncategorized_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Mystery Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Scenes/Mystery.xml",
                    """\
<Scene id="MysteryPose" name="Mystery Pose" actors="2">
  <Stage id="MysteryLoop" duration="5">
    <Animation actorIndex="0" file="meshes\\actors\\character\\animations\\Mystery\\MysteryLoop_0.hkx" />
    <Animation actorIndex="1" file="meshes\\actors\\character\\animations\\Mystery\\MysteryLoop_1.hkx" />
  </Stage>
</Scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/Mystery/MysteryLoop_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Mystery/MysteryLoop_1.hkx", b"")

            result = converter.convert_archive_to_sexlab_zip(archive, sfx_fallback_action="ostimconvertermoan")

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Mystery_Pack.json").decode("utf-8"))
                report_text = converted.read("sexlab_conversion_report.txt").decode("utf-8")

            animation = slal["animations"][0]
            self.assertEqual(animation["sound"], "Squishing")
            self.assertNotIn("sound", animation["stages"][0])
            self.assertTrue(result.report["usesConverterSfxFallback"])
            self.assertEqual(result.report["sfxFallbackAction"], "ostimconvertermoan")
            self.assertIn("SexLab/SLAL exports receive fallback sound metadata", "\n".join(result.report["warnings"]))
            self.assertIn("SexLab SFX profiles: 1", report_text)

    def test_sexlab_export_preserves_source_no_sound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Quiet SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Quiet.json",
                    json.dumps(
                        {
                            "name": "Quiet",
                            "animations": [
                                {
                                    "id": "QuietPose",
                                    "name": "Quiet Pose",
                                    "tags": "FF,Foreplay",
                                    "sound": "none",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "QuietPose_A1_S1"}]},
                                        {"type": "Female", "stages": [{"id": "QuietPose_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Quiet/QuietPose_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Quiet/QuietPose_A2_S1.hkx", b"")

            result = converter.convert_archive_to_sexlab_zip(archive)

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Quiet_SLAL.json").decode("utf-8"))
                source_text = converted.read("Data/SLAnims/source/Quiet_SLAL.txt").decode("utf-8")

            animation = slal["animations"][0]
            self.assertEqual(animation["sound"], "none")
            self.assertNotIn("sound", animation["stages"][0])
            self.assertIn("sound=NoSound", source_text)
            self.assertEqual(result.report["soundProfileCount"], 0)
            self.assertEqual(result.report["noSoundStageEntryCount"], 0)
            self.assertFalse(result.report["usesConverterSfxFallback"])

    def test_sexlab_export_keeps_furniture_and_creature_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Furniture Creature SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Mixed.json",
                    json.dumps(
                        {
                            "name": "Mixed",
                            "animations": [
                                {
                                    "id": "ChairScene",
                                    "name": "Chair Scene",
                                    "tags": "MF,Sex,Furniture,Chair",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "ChairScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "ChairScene_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "CanineScene",
                                    "name": "Canine Scene",
                                    "tags": "MF,Sex,Creature,Canine,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "CanineScene_A1_S1"}]},
                                        {"type": "CreatureMale", "race": "Canines", "stages": [{"id": "CanineScene_A2_S1"}]},
                                    ],
                                    "creature_race": "Canines",
                                },
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Mixed/ChairScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/ChairScene_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/CanineScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/Mixed/CanineScene_A2_S1.hkx", b"")

            result = converter.convert_archive_to_sexlab_zip(archive)

            self.assertEqual(result.report["animationCount"], 2)
            self.assertEqual(result.report["furnitureAnimationCount"], 1)
            self.assertEqual(result.report["creatureAnimationCount"], 1)
            self.assertEqual(result.report["creatureActorRoots"], ["canine"])

            with ZipFile(result.zip_path) as converted:
                slal = json.loads(converted.read("Data/SLAnims/json/Furniture_Creature_SLAL.json").decode("utf-8"))
                names = set(converted.namelist())

            animations = {animation["id"]: animation for animation in slal["animations"]}
            self.assertEqual(animations["ChairScene"]["furniture"], "chair")
            self.assertIn("Furniture", animations["ChairScene"]["tags"])
            self.assertIn("chair", animations["ChairScene"]["tags"])
            self.assertEqual(animations["CanineScene"]["actors"][0]["type"], "CreatureMale")
            self.assertEqual(animations["CanineScene"]["actors"][0]["race"], "Canines")
            self.assertEqual(animations["CanineScene"]["creature_race"], "Canines")
            self.assertIn(
                "Data/meshes/actors/canine/animations/Furniture_Creature_SLAL/CanineScene_A1_S1.hkx",
                names,
            )
            self.assertIn(
                "Data/meshes/actors/canine/animations/Furniture_Creature_SLAL/FNIS_Furniture_Creature_SLAL_canine_List.txt",
                names,
            )

    def test_flowergirls_fnis_list_converts_to_ostim_events_and_pandora_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "FlowerGirls Pack.zip"
            fnis_list = "\n".join(
                [
                    "Version V3.0.2",
                    "' Commented FNIS lines should not be treated as animations.",
                    "b -Tn FG_Missionary_A1_S1 FG_Missionary_A1_S1.hkx",
                    "b -Tn FG_Missionary_A2_S1 FG_Missionary_A2_S1.hkx",
                    "b -Tn FG_Missionary_A1_S2 FG_Missionary_A1_S2.hkx",
                    "b -Tn FG_Missionary_A2_S2 FG_Missionary_A2_S2.hkx",
                    "",
                ]
            )
            with ZipFile(archive, "w") as source:
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FNIS_FlowerGirlsSE_List.txt", fnis_list)
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FG_Missionary_A1_S1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FG_Missionary_A2_S1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FG_Missionary_A1_S2.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FG_Missionary_A2_S2.hkx", b"")
                source.writestr("Data/meshes/actors/character/behaviors/FNIS_FlowerGirlsSE_Behavior.hkx", b"old behavior")

            result = converter.convert_archive_to_ready_zip(archive, mod_author="Xider")
            self.assertEqual(len(result.scenes), 1)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/FlowerGirls_Pack/FlowerGirls_Pack_FG_Missionary.json"
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "FlowerGirls_Pack")
                generated_behavior = read_pandora_compatible_events(converted, names, "FlowerGirls_Pack")
                metadata = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/converter_metadata/FlowerGirls_Pack/metadata.json"
                    ).decode("utf-8")
                )

            self.assertEqual([speed["animation"] for speed in scene["speeds"]], ["FG_Missionary_S1", "FG_Missionary_S2"])
            self.assertEqual(len(scene["actors"]), 2)
            self.assertIn("flowergirls", scene["tags"])
            self.assertIn("Data/meshes/actors/character/animations/FlowerGirls_Pack/FG_Missionary_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/FlowerGirls_Pack/FG_Missionary_S1_1.hkx", names)
            self.assertNotIn("Data/meshes/actors/character/behaviors/FNIS_FlowerGirlsSE_Behavior.hkx", names)
            self.assertNotIn("Data/meshes/actors/character/behaviors/FNIS_FlowerGirls_Pack_Behavior.hkx", names)
            self.assertEqual(len(character_att_list_paths(names, "FlowerGirls_Pack")), 1)
            self.assertIn("FG_Missionary_S1_0", generated_behavior)
            self.assertIn("FG_Missionary_S2_1", generated_behavior)
            self.assertEqual(metadata["pack"]["author"], "Xider")
            self.assertEqual(metadata["deployment"]["status"], "PASS")
            self.assertFalse(metadata["checks"]["hasPandoraAnimData"])
            self.assertFalse(metadata["checks"]["hasPandoraFiles"])
            self.assertTrue(metadata["checks"]["hasGeneratedNemesisBehaviorPatch"])
            self.assertTrue(metadata["checks"]["hasPandoraCompatibleBehavior"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 0)
            self.assertEqual(verification.report["pandoraAnimDataFileCount"], 0)
            self.assertEqual(verification.report["pandoraNamedAnimDataFileCount"], 0)
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE)

    def test_flowergirls_missing_referenced_hkx_is_not_packaged_as_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "FlowerGirls Missing.zip"
            fnis_list = "\n".join(
                [
                    "b -Tn FG_Kissing_A1_S1 FG_Kissing_A1_S1.hkx",
                    "b -Tn FG_Kissing_A2_S1 FG_Kissing_A2_S1.hkx",
                    "",
                ]
            )
            with ZipFile(archive, "w") as source:
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FNIS_FlowerGirlsSE_List.txt", fnis_list)
                source.writestr("Data/meshes/actors/character/animations/FlowerGirlsSE/FG_Kissing_A1_S1.hkx", b"")

            with self.assertRaises(RuntimeError) as raised:
                converter.convert_archive_to_ready_zip(archive)
            self.assertIn("none contained playable animation events", str(raised.exception))

    def test_legacy_sexlab_fnis_scene_list_preserves_actor_groups_and_adds_menu_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SAP Legacy.zip"
            fnis_list = "\n".join(
                [
                    "Version V1.0",
                    "b -Tn SAP_Missionary_A1_S1 SAP_Missionary_A1_S1.hkx",
                    "b -Tn SAP_Missionary_A2_S1 SAP_Missionary_A2_S1.hkx",
                    "b -Tn SAP_Missionary_A1_S2 SAP_Missionary_A1_S2.hkx",
                    "b -Tn SAP_Missionary_A2_S2 SAP_Missionary_A2_S2.hkx",
                    "",
                ]
            )
            source_folder = "Data/meshes/actors/character/animations/SAP"
            with ZipFile(archive, "w") as source:
                source.writestr(f"{source_folder}/FNIS_SAP_List.txt", fnis_list)
                for actor in (1, 2):
                    for stage in (1, 2):
                        source.writestr(f"{source_folder}/SAP_Missionary_A{actor}_S{stage}.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive, report_dir=root).report
            self.assertEqual(diagnosis["detectedSourceTypeCode"], "legacyFnis")
            self.assertEqual(diagnosis["sourceDetectionCounts"]["legacyFnisSceneLists"], 1)

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True)
            normal_scenes = [scene for scene in result.scenes if not converter.is_ostim_menu_hub_scene(scene)]
            self.assertEqual(len(normal_scenes), 1)
            self.assertEqual(len(normal_scenes[0].actors), 2)
            self.assertEqual([speed.animation for speed in normal_scenes[0].speeds], ["SAP_Missionary_S1", "SAP_Missionary_S2"])
            self.assertGreaterEqual(result.report["ostimMenuHubSceneCount"], 3)
            self.assertEqual(result.report["ostimMenuUnreachableDeployableSceneCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/SAP_Legacy/SAP_Legacy_SAP_Missionary.json"
                    ).decode("utf-8")
                )
                mf_menu = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/SAP_Legacy/SAP_Legacy_Menu_MF.json"
                    ).decode("utf-8")
                )

            self.assertIn("legacy-fnis", scene["tags"])
            self.assertEqual(len(scene["actors"]), 2)
            self.assertIn("Data/meshes/actors/character/animations/SAP_Legacy/SAP_Missionary_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/SAP_Legacy/SAP_Missionary_S1_1.hkx", names)
            self.assertTrue(any(nav.get("origin") == "OStim2PStandingApartMF" for nav in mf_menu["navigations"]))
            self.assertTrue(any(nav.get("destination") == "SAP_Legacy_SAP_Missionary" for nav in mf_menu["navigations"]))

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["ostimMenuUnreachableDeployableSceneCount"], 0)

    def test_ordinary_fnis_behavior_list_is_not_misdetected_as_legacy_scene_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "Data" / "meshes" / "actors" / "character" / "animations" / "Movement"
            folder.mkdir(parents=True)
            (folder / "FNIS_Movement_List.txt").write_text(
                "b WalkForward WalkForward.hkx\nb WalkBackward WalkBackward.hkx\n",
                encoding="utf-8",
            )
            (folder / "WalkForward.hkx").write_bytes(b"")
            (folder / "WalkBackward.hkx").write_bytes(b"")

            self.assertEqual(converter.discover_legacy_fnis_scene_list_files(root), [])
            detection = converter.detect_source_pack_summary(root)
            self.assertEqual(detection["selectedSourceType"], "hkxOnly")

    def test_flowergirls_dialogue_plugin_without_animation_data_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "FlowerGirls Dialogue Plugin.zip"
            script_source = """
Scriptname shadow_sex Extends TopicInfo Hidden
Function Fragment_0(ObjectReference akSpeakerRef)
FlowerGirls.MissionaryScene(game.getplayer(), akSpeakerRef as Actor)
EndFunction
dxFlowerGirlsScript Property FlowerGirls Auto
"""
            with ZipFile(archive, "w") as source:
                source.writestr("shadowman_wylandriah.esp", b"plugin")
                source.writestr("Scripts/shadow_sex.pex", b"compiled")
                source.writestr("Scripts/Source/shadow_sex.psc", script_source)
                source.writestr("SEQ/shadowman_wylandriah.seq", b"\x00")

            with self.assertRaises(RuntimeError) as raised:
                converter.convert_archive_to_ready_zip(archive)
            message = str(raised.exception)
            self.assertIn("FlowerGirls dialogue/script plugin", message)
            self.assertIn("not a FlowerGirls animation pack", message)
            self.assertIn("FNIS_FlowerGirlsSE_List.txt", message)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = converter.run_cli(["--mod-archive", str(archive)])

            self.assertEqual(exit_code, 1)
            self.assertIn("FlowerGirls dialogue/script plugin", stdout.getvalue())
            self.assertNotIn("Traceback", stdout.getvalue())

    def test_sexlab_static_furniture_scene_gets_ostim_furniture_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SexLab Furniture Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Furniture.json",
                    json.dumps(
                        {
                            "name": "Furniture_Test",
                            "animations": [
                                {
                                    "id": "Furniture_TestChair",
                                    "name": "Furniture Test Chair",
                                    "tags": "MF,sex,furniture,chair",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "Furniture_TestChair_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "Furniture_TestChair_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Furniture/Furniture_TestChair_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Furniture/Furniture_TestChair_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["startableSceneCount"], 0)
            self.assertEqual(result.report["furnitureSceneCount"], 1)
            self.assertEqual(result.report["deployableSceneCount"], 1)

            with ZipFile(result.zip_path) as converted:
                scene_files = [
                    name
                    for name in converted.namelist()
                    if name.startswith("Data/SKSE/Plugins/OStim/scenes/") and name.endswith(".json")
                ]
                self.assertEqual(len(scene_files), 1)
                scene = json.loads(converted.read(scene_files[0]).decode("utf-8"))
                metadata = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/converter_metadata/SexLab_Furniture_Pack/metadata.json"
                    ).decode("utf-8")
                )

            self.assertEqual(scene["furniture"], "chair")
            self.assertTrue(scene["noRandomSelection"])
            self.assertTrue(scene["scaleOffsetWithFurniture"])
            self.assertIn("furniture", scene["tags"])
            self.assertIn("chair", scene["tags"])
            self.assertEqual(metadata["deployment"]["status"], "PASS")
            self.assertEqual(metadata["deployment"]["furnitureSceneCount"], 1)
            self.assertEqual(metadata["deployment"]["deployableSceneCount"], 1)
            self.assertFalse(metadata["checks"]["hasStartableScenes"])
            self.assertTrue(metadata["checks"]["hasFurnitureScenes"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["startableSceneCount"], 0)
            self.assertEqual(verification.report["furnitureSceneCount"], 1)
            self.assertEqual(verification.report["deployableSceneCount"], 1)

    def test_custom_spawned_furniture_without_builtin_type_remains_startable(self):
        scene = converter.Scene(
            raw_id="Billyy_GloryHole",
            name="Billyy GloryHole",
            speeds=[converter.Speed(animation="Billyy_GloryHole_S1")],
            actors=[converter.Actor(animation_index=0), converter.Actor(animation_index=1)],
            tags=["furniture", "animobject", "gloryhole"],
            actions=[converter.Action(type="vaginalsex", actor=0, target=1)],
        )

        warnings = converter.apply_scene_furniture_metadata([scene])

        self.assertEqual(scene.furniture, "none")
        self.assertFalse(scene.no_random_selection)
        self.assertTrue(converter.is_startable_scene(scene))
        self.assertFalse(converter.is_furniture_scene(scene))
        self.assertTrue(any("custom furniture/animobject" in warning for warning in warnings))

    def test_spawned_object_chair_dildo_does_not_become_static_chair(self):
        scene = converter.Scene(
            raw_id="B_Billyy_ChairDildo",
            name="Billyy Chair Dildo",
            speeds=[converter.Speed(animation="B_Billyy_ChairDildo_S1")],
            actors=[converter.Actor(animation_index=0)],
            tags=["billyy", "sex", "dildo", "object", "chair", "sextoy", "masturbation"],
            actions=[converter.Action(type="femalemasturbation", actor=0, target=0)],
        )

        warnings = converter.apply_scene_furniture_metadata([scene])

        self.assertEqual(scene.furniture, "none")
        self.assertFalse(converter.is_furniture_scene(scene))
        self.assertTrue(converter.is_startable_scene(scene))
        self.assertTrue(any("custom furniture/animobject" in warning for warning in warnings))

    def test_workbench_furniture_aliases_prefer_specific_ostim_types(self):
        alchemy = converter.Scene(
            raw_id="B_Billyy_AlchemyWorkBenchFuck",
            name="Billyy Alchemy WorkBench Fuck",
            speeds=[converter.Speed(animation="Alchemy_S1")],
            actors=[converter.Actor(animation_index=0), converter.Actor(animation_index=1)],
            tags=["furniture", "object", "animobject", "alchemywb", "workbench"],
            actions=[converter.Action(type="vaginalsex", actor=0, target=1)],
        )
        enchanting = converter.Scene(
            raw_id="B_Billyy_EnchantingWorkBenchFuck",
            name="Billyy Enchanting WorkBench Fuck",
            speeds=[converter.Speed(animation="Enchanting_S1")],
            actors=[converter.Actor(animation_index=0), converter.Actor(animation_index=1)],
            tags=["furniture", "object", "animobject", "enchantingwb", "workbench"],
            actions=[converter.Action(type="vaginalsex", actor=0, target=1)],
        )

        converter.apply_scene_furniture_metadata([alchemy, enchanting])

        self.assertEqual(alchemy.furniture, "alchemytable")
        self.assertEqual(enchanting.furniture, "enchantingtable")
        self.assertTrue(converter.is_furniture_scene(alchemy))
        self.assertTrue(converter.is_furniture_scene(enchanting))

    def test_legacy_ostim_custom_furniture_type_file_is_copied_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Legacy Furniture Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/Legacy/PilloryScene.json",
                    json.dumps(
                        {
                            "name": "Legacy Pillory Scene",
                            "modpack": "Legacy Furniture Pack",
                            "length": 5,
                            "furniture": "pillory",
                            "speeds": [{"animation": "PilloryScene_S1"}],
                            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                            "actions": [{"type": "vaginalsex", "actor": 0, "target": 1}],
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/furniture types/pillory.json",
                    json.dumps({"name": "pillory", "types": ["pillory"]}),
                )
                source.writestr("Data/meshes/actors/character/animations/Legacy/PilloryScene_S1_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/PilloryScene_S1_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["furnitureSceneCount"], 1)
            self.assertEqual(result.report["furnitureTypeFileCount"], 1)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene_files = [
                    name
                    for name in names
                    if name.startswith("Data/SKSE/Plugins/OStim/scenes/") and name.endswith(".json")
                ]
                scene = json.loads(converted.read(scene_files[0]).decode("utf-8"))

            self.assertIn("Data/SKSE/Plugins/OStim/furniture types/pillory.json", names)
            self.assertEqual(scene["furniture"], "pillory")
            self.assertTrue(scene["noRandomSelection"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["furnitureSceneCount"], 1)
            self.assertEqual(verification.report["ostimFurnitureTypeFileCount"], 1)

    def test_sexlab_remaps_anubs_stage_actor_swapped_hkx_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Anubs Animation Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Anubs.json",
                    json.dumps(
                        {
                            "name": "Anubs_Test",
                            "animations": [
                                {
                                    "id": "Anubs_A_bedsleep",
                                    "name": "Anubs Bed Sleep",
                                    "tags": "Anubs,MF,sex",
                                    "actors": [
                                        {
                                            "type": "Female",
                                            "stages": [
                                                {"id": "Anubs_A_bedsleep_A1_S1"},
                                                {"id": "Anubs_A_bedsleep_A1_S2"},
                                            ],
                                        },
                                        {
                                            "type": "Male",
                                            "stages": [
                                                {"id": "Anubs_A_bedsleep_A2_S1"},
                                                {"id": "Anubs_A_bedsleep_A2_S2"},
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Anubs/Anubs_A_bedsleep_S1_A1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Anubs/Anubs_A_bedsleep_S2_A1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Anubs/Anubs_A_bedsleep_S1_A2.hkx", b"")
                source.writestr("meshes/actors/character/animations/Anubs/Anubs_A_bedsleep_S2_A2.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, mod_author="Anub")
            self.assertEqual(len(result.hkx_assets), 4)
            self.assertFalse(any("missing from the source archive" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "Anubs_Animation_Pack")
                behavior_events = read_pandora_compatible_events(converted, names, "Anubs_Animation_Pack")

            self.assertIn("Data/meshes/actors/character/animations/Anubs_Animation_Pack/Anubs_A_bedsleep_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Anubs_Animation_Pack/Anubs_A_bedsleep_S1_1.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Anubs_Animation_Pack/Anubs_A_bedsleep_S2_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Anubs_Animation_Pack/Anubs_A_bedsleep_S2_1.hkx", names)
            self.assertEqual(len(character_att_list_paths(names, "Anubs_Animation_Pack")), 1)
            self.assertIn("Anubs_A_bedsleep_S1_0", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_sexlab_remaps_vendor_prefixed_stage_ids_to_unprefixed_hkx_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SLSB Anub.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Anubs.json",
                    json.dumps(
                        {
                            "name": "Anubs_Test",
                            "animations": [
                                {
                                    "id": "Anubs_A_bedsleep",
                                    "name": "Anubs Bed Sleep",
                                    "tags": "Anubs,MF,sex",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "Anubs_A_bedsleep_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "Anubs_A_bedsleep_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Anubs Human/A_bedsleep_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Anubs Human/A_bedsleep_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, mod_author="Anub")
            self.assertEqual(len(result.hkx_assets), 2)
            self.assertFalse(any("missing from the source archive" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "SLSB_Anub")
                behavior_events = read_pandora_compatible_events(converted, names, "SLSB_Anub")

            self.assertIn("Data/meshes/actors/character/animations/SLSB_Anub/Anubs_A_bedsleep_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/SLSB_Anub/Anubs_A_bedsleep_S1_1.hkx", names)
            self.assertEqual(len(character_att_list_paths(names, "SLSB_Anub")), 1)
            self.assertIn("Anubs_A_bedsleep_S1_0", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_sexlab_duplicate_sanitized_ids_get_unique_animation_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Duplicate SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Dupe.json",
                    json.dumps(
                        {
                            "name": "Dupe",
                            "animations": [
                                {
                                    "id": "Duplicate Scene",
                                    "name": "Duplicate Scene One",
                                    "tags": "MF,sex",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "UniqueOne_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "UniqueOne_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "Duplicate/Scene",
                                    "name": "Duplicate Scene Two",
                                    "tags": "MF,sex",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "UniqueTwo_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "UniqueTwo_A2_S1"}]},
                                    ],
                                },
                            ],
                        }
                    ),
                )
                for event in ("UniqueOne_A1_S1", "UniqueOne_A2_S1", "UniqueTwo_A1_S1", "UniqueTwo_A2_S1"):
                    source.writestr(f"meshes/actors/character/animations/Dupe/{event}.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.hkx_assets), 4)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "Duplicate_SLAL")
                behavior_events = read_pandora_compatible_events(converted, names, "Duplicate_SLAL")

            self.assertIn("Data/meshes/actors/character/animations/Duplicate_SLAL/Duplicate_Scene_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Duplicate_SLAL/Duplicate_Scene_S1_1.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Duplicate_SLAL/Duplicate_Scene_2_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/Duplicate_SLAL/Duplicate_Scene_2_S1_1.hkx", names)
            self.assertEqual(len(character_att_list_paths(names, "Duplicate_SLAL")), 1)
            self.assertIn("Duplicate_Scene_S1_0", behavior_events)
            self.assertIn("Duplicate_Scene_2_S1_0", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_sexlab_creature_actor_roots_are_packaged_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Creature Root.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/CreatureRoot.json",
                    json.dumps(
                        {
                            "name": "CreatureRoot",
                            "animations": [
                                {
                                    "id": "HorseScene",
                                    "name": "Horse Scene",
                                    "tags": "MF,sex,Creature,Horse,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "HorseScene_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "HorseScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/CreatureRoot/HorseScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/horse/animations/CreatureRoot/HorseScene_A2_S1.hkx", b"")
                source.writestr("meshes/actors/horse/behaviors/FNIS_SourceHorse_Behavior.hkx", b"same horse behavior graph")
                source.writestr("meshes/actors/horse/behaviors/FNIS_DuplicateHorse_Behavior.hkx", b"same horse behavior graph")

            result = converter.convert_archive_to_ready_zip(archive, human_only_ostim=False)
            deployable_scenes = [
                scene for scene in result.scenes
                if converter.is_deployable_scene(scene) and not converter.is_ostim_menu_hub_scene(scene)
            ]
            self.assertEqual(len(deployable_scenes), 1)
            self.assertTrue(result.report["creatureMenuIntegration"]["ocreaturesMenuEntryGenerated"])
            self.assertEqual(result.report["creatureSceneCount"], 1)
            self.assertEqual(result.report["creatureActorRoots"], ["horse"])
            self.assertTrue(result.report["requiresCreatureSupport"])

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/OCreatures/Creature_Root/Creature_Root_HorseScene.json"
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "Creature_Root")
                assert_pandora_compatible_registration(self, names, "Creature_Root", "horseproject")
                human_events = read_pandora_compatible_events(converted, names, "Creature_Root")
                horse_events = read_pandora_compatible_events(converted, names, "Creature_Root", "horse")

            self.assertIn("Data/meshes/actors/character/animations/Creature_Root/HorseScene_S1_1.hkx", names)
            self.assertIn("Data/meshes/actors/horse/animations/Creature_Root/HorseScene_S1_0.hkx", names)
            horse_behavior_graphs = sorted(
                name for name in names
                if name.startswith("Data/meshes/actors/horse/behaviors/")
                and name.endswith("_Behavior.hkx")
            )
            self.assertEqual(
                horse_behavior_graphs,
                ["Data/meshes/actors/horse/behaviors/FNIS_Creature_Root_horse_Behavior.hkx"],
            )
            self.assertEqual(len(character_att_list_paths(names, "Creature_Root")), 1)
            self.assertTrue(any(name.startswith(f"Data/Nemesis_Engine/mod/{converter.safe_nemesis_patch_code(result.pack)}/") for name in names))
            self.assertEqual([actor.get("intendedSex") for actor in scene["actors"]], ["male", "female"])
            self.assertIn("creature", scene["actors"][0]["tags"])
            self.assertEqual(scene["actors"][0]["creatureRace"], "Horses")
            self.assertIn("HorseScene_S1_1", human_events)
            self.assertIn("HorseScene_S1_0", horse_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["creatureSceneCount"], 1)
            self.assertEqual(verification.report["creatureActorRoots"], ["horse"])
            self.assertEqual(
                verification.report["behaviorOutputMode"],
                converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE_WITH_CREATURE,
            )
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 1)
            self.assertEqual(verification.report["duplicateBehaviorGraphFileCount"], 0)

    def test_sexlab_nested_creature_actor_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Nested Creature Root.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/NestedCreatureRoot.json",
                    json.dumps(
                        {
                            "name": "NestedCreatureRoot",
                            "animations": [
                                {
                                    "id": "NestedScene",
                                    "name": "Nested Scene",
                                    "tags": "MF,sex,Creature,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "NestedScene_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "NestedScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/NestedCreatureRoot/NestedScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/dlc01/chaurusflyer/animations/NestedCreatureRoot/NestedScene_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, human_only_ostim=False)
            self.assertEqual(result.report["creatureActorRoots"], ["dlc01/chaurusflyer"])

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "Nested_Creature_Root")
                assert_pandora_compatible_registration(self, names, "Nested_Creature_Root", "chaurusflyer")
                nested_events = read_pandora_compatible_events(
                    converted,
                    names,
                    "Nested_Creature_Root",
                    "dlc01/chaurusflyer",
                )

            self.assertIn(
                "Data/meshes/actors/dlc01/chaurusflyer/animations/Nested_Creature_Root/NestedScene_S1_0.hkx",
                names,
            )
            self.assertTrue(any(name.startswith(f"Data/Nemesis_Engine/mod/{converter.safe_nemesis_patch_code(result.pack)}/") for name in names))
            self.assertIn("NestedScene_S1_0", nested_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["creatureActorRoots"], ["dlc01/chaurusflyer"])

    def test_sexlab_creature_actor_entries_are_converted_for_ostim_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Mixed SLSB.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Mixed.json",
                    json.dumps(
                        {
                            "name": "MixedSLSB",
                            "animations": [
                                {
                                    "id": "HumanScene",
                                    "name": "Human Scene",
                                    "tags": "MF,sex",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "HumanScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "HumanScene_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "UnsupportedScene",
                                    "name": "Unsupported Scene",
                                    "tags": "Creature,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "UnsupportedScene_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "UnsupportedScene_A2_S1"}]},
                                    ],
                                },
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Mixed/HumanScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/HumanScene_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/UnsupportedScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/Mixed/UnsupportedScene_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, human_only_ostim=False)
            deployable_scenes = [
                scene for scene in result.scenes
                if converter.is_deployable_scene(scene) and not converter.is_ostim_menu_hub_scene(scene)
            ]
            self.assertEqual(len(deployable_scenes), 2)
            self.assertEqual({scene.raw_id for scene in deployable_scenes}, {"HumanScene", "UnsupportedScene"})
            self.assertTrue(result.report["creatureMenuIntegration"]["ocreaturesMenuEntryGenerated"])
            self.assertEqual(len(result.hkx_assets), 4)
            self.assertEqual(result.report["creatureSceneCount"], 1)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())

            self.assertIn("Data/meshes/actors/character/animations/Mixed_SLSB/HumanScene_S1_0.hkx", names)
            self.assertIn("Data/SKSE/Plugins/OStim/scenes/OCreatures/Mixed_SLSB/Mixed_SLSB_UnsupportedScene.json", names)
            self.assertIn("Data/meshes/actors/character/animations/Mixed_SLSB/UnsupportedScene_S1_1.hkx", names)
            self.assertIn("Data/meshes/actors/canine/animations/Mixed_SLSB/UnsupportedScene_S1_0.hkx", names)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_mixed_slal_defaults_to_human_only_ostim_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Mixed Human Creature SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/Mixed.json",
                    json.dumps(
                        {
                            "name": "MixedHumanCreature",
                            "animations": [
                                {
                                    "id": "HumanScene",
                                    "name": "Human Scene",
                                    "tags": "MF,sex,vaginal",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "HumanScene_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "HumanScene_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "CanineScene",
                                    "name": "Canine Scene",
                                    "tags": "MF,sex,Creature,Canine,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "CanineScene_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "CanineScene_A2_S1"}]},
                                    ],
                                },
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/Mixed/HumanScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/HumanScene_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Mixed/CanineScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/Mixed/CanineScene_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True)

            self.assertEqual({scene.raw_id for scene in result.scenes if not converter.is_ostim_menu_hub_scene(scene)}, {"HumanScene"})
            self.assertTrue(result.report["humanOnlyOStimOutput"])
            self.assertEqual(result.report["creatureScenesSkipped"], 1)
            self.assertEqual(result.report["mixedHumanCreatureScenesSkipped"], 1)
            self.assertEqual(result.report["creatureSceneCount"], 0)
            self.assertEqual(result.report["creatureActorRoots"], [])
            self.assertIn("canine", result.report["humanOnlyOStimFiltering"]["detectedCreatureRoots"])
            self.assertTrue(any("Human-only OStim output was enabled" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                report_text = converted.read("conversion_report.txt").decode("utf-8")

            self.assertIn("Data/SKSE/Plugins/OStim/scenes/Mixed_Human_Creature_SLAL/Mixed_Human_Creature_SLAL_HumanScene.json", names)
            self.assertNotIn("Data/SKSE/Plugins/OStim/scenes/OCreatures/Mixed_Human_Creature_SLAL/Mixed_Human_Creature_SLAL_CanineScene.json", names)
            self.assertNotIn("Data/meshes/actors/canine/animations/Mixed_Human_Creature_SLAL/CanineScene_S1_0.hkx", names)
            self.assertNotIn("Data/meshes/actors/character/animations/Mixed_Human_Creature_SLAL/CanineScene_S1_1.hkx", names)
            self.assertIn("Creature / Human-only Filtering:", report_text)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertTrue(verification.report["humanOnlyOStimOutput"])
            self.assertEqual(verification.report["creatureScenesSkipped"], 1)
            self.assertEqual(verification.report["creatureSceneCount"], 0)
            self.assertEqual(verification.report["creatureActorRoots"], [])

    def test_creature_only_slal_human_only_output_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Creature Only SLAL.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/CreatureOnly.json",
                    json.dumps(
                        {
                            "name": "CreatureOnly",
                            "animations": [
                                {
                                    "id": "HorseScene",
                                    "name": "Horse Scene",
                                    "tags": "MF,sex,Creature,Horse,Bestiality",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "HorseScene_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "HorseScene_A2_S1"}]},
                                    ],
                                }
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/CreatureOnly/HorseScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/horse/animations/CreatureOnly/HorseScene_A2_S1.hkx", b"")

            diagnosis = converter.diagnose_source_archive(archive, write_report=False)
            self.assertTrue(diagnosis.report["humanOnlyWouldSkipAllScenes"])
            self.assertIn("Do not build this archive with default Human-only OStim output", diagnosis.report["recommendedUserAction"])
            self.assertIn("creature output only", diagnosis.report["recommendedUserAction"])

            with self.assertRaises(RuntimeError) as raised:
                converter.convert_archive_to_ready_zip(archive)

            message = str(raised.exception)
            self.assertIn("Human-only OStim output was enabled", message)
            self.assertIn("every playable scene was creature or mixed creature content", message)

    def test_human_only_default_can_be_overridden_by_compatibility_profile(self):
        args = converter.build_arg_parser().parse_args(["--mod-archive", "Creature Regression.zip"])
        self.assertIsNone(args.human_only_ostim)
        self.assertTrue(
            converter.build_arg_parser()
            .parse_args(["--mod-archive", "Creature Regression.zip", "--human-only-ostim"])
            .human_only_ostim
        )
        self.assertFalse(
            converter.build_arg_parser()
            .parse_args(["--mod-archive", "Creature Regression.zip", "--include-creature-ostim"])
            .human_only_ostim
        )
        self.assertTrue(converter.default_human_only_ostim_from_source_detection({}))
        self.assertFalse(
            converter.default_human_only_ostim_from_source_detection(
                {"compatibilityMatch": {"defaultHumanOnlyOStim": False}}
            )
        )
        self.assertFalse(
            converter.default_human_only_ostim_from_source_detection(
                {"compatibilityMatch": {"creatureOutputRecommended": True}}
            )
        )

    def test_recommended_build_uses_profile_behavior_output(self):
        self.assertTrue(
            converter.recommended_nemesis_safe_output_from_report(
                {
                    "recommendedBehaviorTool": "Pandora creature-capable output",
                    "compatibilityMatch": {"recommendedBehaviorTool": "Pandora creature-capable output"},
                },
                fallback=True,
            )
        )
        self.assertTrue(
            converter.recommended_nemesis_safe_output_from_report(
                {
                    "recommendedBehaviorTool": "Legacy Nemesis/ATT output",
                    "compatibilityMatch": {"recommendedBehaviorTool": "Legacy Nemesis/ATT output"},
                },
                fallback=False,
            )
        )

    def test_flufyfox_creature_profile_recommends_creature_output(self):
        db_path = Path(__file__).resolve().parents[1] / "compatibility_db.json"
        entries = json.loads(db_path.read_text(encoding="utf-8"))["entries"]
        profile = next(entry for entry in entries if entry.get("id") == "flufyfox-slal-se-c-3-6")

        self.assertFalse(profile["defaultHumanOnlyOStim"])
        self.assertTrue(profile["creatureOutputRecommended"])
        self.assertTrue(profile["creatureRuntimeRequired"])
        self.assertEqual(profile["creatureMenuIntegrationRequired"], converter.CREATURE_MENU_INTEGRATION_OCREATURES_MENU)
        self.assertTrue(profile["warnIfOCreaturesMenuEntryMissing"])
        self.assertEqual(profile["expectedOStimSceneJsonFiles"], 106)
        self.assertEqual(profile["expectedDeployableCreatureScenes"], 46)
        self.assertEqual(profile["expectedHkxPackaged"], 521)
        self.assertEqual(profile["expectedBehaviorEvents"], 521)
        self.assertIn("dlc01/chaurusflyer", profile["expectedCreatureRoots"])
        self.assertIn("werewolfbeast", profile["expectedCreatureRoots"])
        self.assertTrue(profile["ocreaturesOutputRecommended"])
        self.assertTrue(profile["ocreaturesMenuIntegrationRequired"])
        self.assertEqual(profile["ocreaturesActorMapping"], converter.OCREATURES_SLAL_ACTOR_MAPPING_RULE)
        self.assertFalse(profile["ocreaturesPreserveHkxNames"])
        self.assertEqual(profile["ocreaturesHkxFilenamePolicy"], converter.OCREATURES_HKX_FILENAME_POLICY_EVENT_RENAMED)
        self.assertTrue(profile["ocreaturesRequiresTnFlag"])

    def test_ocreatures_slal_actor_mapping_report_and_tn_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Tri Creature SLAL.zip"
            animation = {
                "id": "TriCreature",
                "name": "Tri Creature",
                "tags": "MF,MMF,sex,Creature,Canine,Bestiality",
                "actors": [
                    {"type": "Female", "stages": [{"id": "TriCreature_A1_S1"}]},
                    {"type": "CreatureMale", "race": "canine", "stages": [{"id": "TriCreature_A2_S1"}]},
                    {"type": "Male", "stages": [{"id": "TriCreature_A3_S1"}]},
                ],
            }
            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/json/TriCreature.json", json.dumps({"name": "TriCreature", "animations": [animation]}))
                source.writestr("meshes/actors/character/animations/TriCreature/TriCreature_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/TriCreature/TriCreature_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/TriCreature/TriCreature_A3_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(
                archive,
                ostim_menu_entry=True,
                human_only_ostim=False,
                nemesis_safe_output=True,
            )

            self.assertIsNotNone(result.verification)
            self.assertTrue(result.verification.ok, result.verification.report["errors"])
            self.assertEqual(
                result.report["behaviorRegistrationSummary"]["selectedBehaviorOutputMode"],
                converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE_WITH_CREATURE,
            )
            self.assertEqual(result.report["behaviorRegistrationSummary"]["creatureFnisTnFlagMissing"], 0)
            self.assertGreater(result.report["behaviorRegistrationSummary"]["creatureFnisTnFlagPresent"], 0)

            ocreatures_output = result.report["ocreaturesOutput"]
            self.assertTrue(ocreatures_output["enabled"])
            self.assertIn(converter.OCREATURES_OUTPUT_MODE_FROM_SLAL, ocreatures_output["selectedModes"])
            self.assertTrue(ocreatures_output["ocreaturesMenuIndexGenerated"])
            self.assertEqual(ocreatures_output["actorMapping"]["finalActorMappingResult"], "PASS")
            self.assertEqual(ocreatures_output["actorMappingRule"], converter.OCREATURES_SLAL_ACTOR_MAPPING_RULE)
            self.assertEqual(ocreatures_output["hkxFilenamePolicy"], converter.OCREATURES_HKX_FILENAME_POLICY_EVENT_RENAMED)
            self.assertFalse(ocreatures_output["preserveOriginalHkxFilenames"])
            self.assertTrue(ocreatures_output["referencePreservesOriginalHkxFilenames"])

            rows_by_source_slot = {
                row["sourceActorSlot"]: row
                for row in ocreatures_output["actorMapping"]["mappingRows"]
                if row.get("sourceAnimationId") == "TriCreature"
            }
            self.assertEqual(rows_by_source_slot[1]["targetOCreaturesActorSlot"], 1)
            self.assertEqual(rows_by_source_slot[1]["targetEventName"], "TriCreature_S1_1")
            self.assertEqual(rows_by_source_slot[1]["sourceHkxFilename"], "TriCreature_A1_S1.hkx")
            self.assertEqual(rows_by_source_slot[2]["targetOCreaturesActorSlot"], 0)
            self.assertEqual(rows_by_source_slot[2]["targetEventName"], "TriCreature_S1_0")
            self.assertEqual(rows_by_source_slot[2]["actorRoot"], "canine")
            self.assertEqual(rows_by_source_slot[3]["targetOCreaturesActorSlot"], 2)
            self.assertEqual(rows_by_source_slot[3]["targetEventName"], "TriCreature_S1_2")

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                creature_lists = [
                    name
                    for name in names
                    if name.startswith("Data/meshes/actors/canine/animations/Tri_Creature_SLAL/")
                    and name.endswith("_List.txt")
                ]
                self.assertEqual(len(creature_lists), 1)
                creature_list = converted.read(mod_entry(creature_lists[0])).decode("utf-8")
                self.assertIn("b -Tn TriCreature_S1_0 TriCreature_S1_0.hkx", creature_list)
                self.assertIn("Data/SKSE/Plugins/OStim/scenes/OCreatures/OCrCanine/", "\n".join(names))

            verification = result.verification.report
            self.assertEqual(verification["ocreaturesOutput"]["actorMapping"]["finalActorMappingResult"], "PASS")
            self.assertEqual(verification["ocreaturesOutput"]["tnFlagHandling"]["verificationResult"], "PASS")
            self.assertEqual(verification["creatureFnisTnFlagMissingCount"], 0)
            self.assertGreater(verification["creatureFnisTnFlagPresentCount"], 0)
            validation = verification["ocreaturesValidation"]
            self.assertTrue(validation["enabled"])
            self.assertEqual(validation["slotMappingRule"], converter.OCREATURES_SLAL_ACTOR_MAPPING_RULE)
            self.assertEqual(validation["actorMappingResult"], "PASS")
            self.assertEqual(validation["runtimeDependencyCheckResult"], "NOT CHECKED")
            self.assertEqual(validation["failedCheckCount"], 0)
            self.assertEqual(validation["finalOCreaturesValidationResult"], "PASS WITH WARNINGS")
            checks_by_id = {check["id"]: check for check in validation["hardChecks"]}
            self.assertEqual(checks_by_id["scene_hkx_references"]["result"], "PASS")
            self.assertEqual(checks_by_id["behavior_hkx_targets"]["result"], "PASS")
            self.assertEqual(checks_by_id["sanitized_event_duplicates"]["result"], "PASS")
            self.assertEqual(checks_by_id["slot_mapping"]["result"], "PASS")
            self.assertEqual(checks_by_id["duplicate_backend_registration"]["result"], "PASS")

            report_text = result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report_README.txt").read_text(encoding="utf-8")
            self.assertIn("OCreatures Output", report_text)
            self.assertIn("OCreatures Validation", report_text)
            self.assertIn("Actor mapping rule:", report_text)
            self.assertIn("-Tn flag handling:", report_text)

    def test_ocreatures_mapping_is_source_slot_based_even_when_creature_is_not_a2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Five Slot Creature SLAL.zip"
            animation = {
                "id": "FiveSlotCreature",
                "name": "Five Slot Creature",
                "tags": "MMMMF,Creature,Canine,Bestiality",
                "actors": [
                    {"type": "Female", "stages": [{"id": "FiveSlotCreature_A1_S1"}]},
                    {"type": "Male", "stages": [{"id": "FiveSlotCreature_A2_S1"}]},
                    {"type": "CreatureMale", "race": "canine", "stages": [{"id": "FiveSlotCreature_A3_S1"}]},
                    {"type": "Female", "stages": [{"id": "FiveSlotCreature_A4_S1"}]},
                    {"type": "Male", "stages": [{"id": "FiveSlotCreature_A5_S1"}]},
                ],
            }
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/FiveSlotCreature.json",
                    json.dumps({"name": "FiveSlotCreature", "animations": [animation]}),
                )
                source.writestr("meshes/actors/character/animations/FiveSlotCreature/FiveSlotCreature_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/FiveSlotCreature/FiveSlotCreature_A2_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/FiveSlotCreature/FiveSlotCreature_A3_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/FiveSlotCreature/FiveSlotCreature_A4_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/FiveSlotCreature/FiveSlotCreature_A5_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(
                archive,
                ostim_menu_entry=True,
                human_only_ostim=False,
                nemesis_safe_output=True,
            )

            self.assertIsNotNone(result.verification)
            self.assertTrue(result.verification.ok, result.verification.report["errors"])

            rows_by_source_slot = {
                row["sourceActorSlot"]: row
                for row in result.report["ocreaturesOutput"]["actorMapping"]["mappingRows"]
                if row.get("sourceAnimationId") == "FiveSlotCreature"
            }
            expected_slots = {1: 1, 2: 0, 3: 2, 4: 3, 5: 4}
            self.assertEqual(set(rows_by_source_slot), set(expected_slots))
            for source_slot, target_slot in expected_slots.items():
                with self.subTest(source_slot=source_slot):
                    row = rows_by_source_slot[source_slot]
                    self.assertEqual(row["targetOCreaturesActorSlot"], target_slot)
                    self.assertEqual(row["targetEventName"], f"FiveSlotCreature_S1_{target_slot}")

            self.assertEqual(rows_by_source_slot[3]["actorRoot"], "canine")
            validation = result.verification.report["ocreaturesValidation"]
            self.assertEqual(validation["actorMappingResult"], "PASS")
            self.assertEqual(validation["failedCheckCount"], 0)
            self.assertEqual(validation["finalOCreaturesValidationResult"], "PASS WITH WARNINGS")

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                creature_lists = [
                    name
                    for name in names
                    if name.startswith("Data/meshes/actors/canine/animations/Five_Slot_Creature_SLAL/")
                    and name.endswith("_List.txt")
                ]
                self.assertEqual(len(creature_lists), 1)
                creature_list = converted.read(mod_entry(creature_lists[0])).decode("utf-8")
                self.assertIn("b -Tn FiveSlotCreature_S1_2 FiveSlotCreature_S1_2.hkx", creature_list)

    def test_ocreatures_menu_helpers_keep_group_actor_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Mixed Creature Group SLAL.zip"
            animations = [
                {
                    "id": "AardvarkPair",
                    "name": "Aardvark Pair",
                    "tags": "MF,Creature,Canine,Bestiality",
                    "actors": [
                        {"type": "Female", "stages": [{"id": "AardvarkPair_A1_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "AardvarkPair_A2_S1"}]},
                    ],
                },
                {
                    "id": "ZebraFive",
                    "name": "Zebra Five",
                    "tags": "FMMMM,Creature,Canine,Bestiality",
                    "actors": [
                        {"type": "Female", "stages": [{"id": "ZebraFive_A1_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "ZebraFive_A2_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "ZebraFive_A3_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "ZebraFive_A4_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "ZebraFive_A5_S1"}]},
                    ],
                },
            ]
            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/json/MixedCreatureGroup.json", json.dumps({"name": "MixedCreatureGroup", "animations": animations}))
                source.writestr("meshes/actors/character/animations/MixedCreatureGroup/AardvarkPair_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/MixedCreatureGroup/AardvarkPair_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/MixedCreatureGroup/ZebraFive_A1_S1.hkx", b"")
                for slot in range(2, 6):
                    source.writestr(f"meshes/actors/canine/animations/MixedCreatureGroup/ZebraFive_A{slot}_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(
                archive,
                ostim_menu_entry=True,
                human_only_ostim=False,
                nemesis_safe_output=True,
            )

            self.assertIsNotNone(result.verification)
            self.assertTrue(result.verification.ok, result.verification.report["errors"])

            creature_menu = result.report["creatureMenuIntegration"]
            self.assertTrue(creature_menu["ocreaturesMenuEntryGenerated"])
            self.assertEqual(creature_menu["menuActorCoverageResult"], "PASS")
            self.assertEqual(creature_menu["menuActorCoverageMismatchCount"], 0)
            verify_menu = result.verification.report["creatureMenuIntegration"]
            self.assertEqual(verify_menu["menuActorCoverageResult"], "PASS")
            self.assertEqual(verify_menu["menuActorCoverageMismatchCount"], 0)

            checks_by_id = {
                check["id"]: check
                for check in result.verification.report["ocreaturesValidation"]["hardChecks"]
            }
            self.assertEqual(checks_by_id["ocreatures_menu_actor_coverage"]["result"], "PASS")

            with ZipFile(result.zip_path) as converted:
                menu_entry = creature_menu["ocreaturesMenuEntryFile"]
                root_scene = json.loads(converted.read(mod_entry(menu_entry)).decode("utf-8"))
                self.assertGreaterEqual(len(root_scene["actors"]), 5)

    def test_human_slal_actor_mapping_is_unchanged_by_ocreatures_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            json_path = Path(temp) / "HumanPair.json"
            json_path.write_text(
                json.dumps(
                    {
                        "name": "HumanPair",
                        "animations": [
                            {
                                "id": "HumanPairScene",
                                "tags": "MF,sex",
                                "actors": [
                                    {"type": "Female", "stages": [{"id": "HumanPair_A1_S1"}]},
                                    {"type": "Male", "stages": [{"id": "HumanPair_A2_S1"}]},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            scenes, warnings, event_map, lifecycle_rows = converter.parse_sexlab_animation_json(json_path)

            self.assertFalse(warnings)
            self.assertEqual(len(scenes), 1)
            self.assertEqual(event_map["HumanPair_A1_S1"], "HumanPairScene_S1_0")
            self.assertEqual(event_map["HumanPair_A2_S1"], "HumanPairScene_S1_1")
            mapping_rows = lifecycle_rows[0]["actorStageMappings"]
            self.assertEqual([row["mappingRule"] for row in mapping_rows], ["normal_ostim_actor_order", "normal_ostim_actor_order"])
            self.assertEqual([row["targetOStimActorSlot"] for row in mapping_rows], [0, 1])

    def test_ocreatures_reference_comparison_reports_missing_menu_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            aac = root / "aac"
            reference = root / "reference"

            playable_scene = {
                "name": "Tri Creature",
                "modpack": "Tri Creature",
                "speeds": [{"animation": "TriCreature_S1"}],
                "actors": [
                    {"animationIndex": 1, "type": "female"},
                    {"animationIndex": 0, "type": "creature", "creatureRace": "Canine", "tags": ["creature"]},
                    {"animationIndex": 2, "type": "male"},
                ],
                "actions": [{"type": "analsex", "actor": 1, "target": 0}],
            }
            menu_scene = {
                "name": "Tri Creature",
                "modpack": "Tri Creature",
                "tags": [
                    converter.OSTIM_MENU_HUB_TAG,
                    converter.OCREATURES_MENU_TAG,
                    converter.OCREATURES_MENU_ROOT_TAG,
                    f"{converter.OCREATURES_MENU_FOLDER_TAG_PREFIX}OCrCanine",
                ],
                "navigations": [{"destination": "TriScene"}],
            }
            for base in (aac, reference):
                scene_dir = base / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / "OCreatures" / "OCrCanine"
                anim_dir = base / "Data" / "meshes" / "actors" / "canine" / "animations" / "TriCreature"
                scene_dir.mkdir(parents=True)
                anim_dir.mkdir(parents=True)
                (scene_dir / "TriScene.json").write_text(json.dumps(playable_scene), encoding="utf-8")
                (anim_dir / "TriCreature_S1_0.hkx").write_bytes(b"")
                (anim_dir / "FNIS_TriCreature_canine_List.txt").write_text(
                    "b -Tn TriCreature_S1_0 TriCreature_S1_0.hkx\n",
                    encoding="utf-8",
                )
            (reference / "Data" / "SKSE" / "Plugins" / "OStim" / "scenes" / "OCreatures" / "OCrCanine" / "TriMenu.json").write_text(
                json.dumps(menu_scene),
                encoding="utf-8",
            )

            comparison = converter.compare_ocreatures_reference(aac, reference)

            self.assertFalse(comparison["ok"])
            self.assertTrue(comparison["missingOCreaturesMenuIndex"])
            self.assertEqual(comparison["missingBehaviorEventCount"], 0)
            self.assertEqual(comparison["missingSceneEventCount"], 0)
            self.assertTrue(any("menu registration" in item for item in comparison["recommendedFixCandidates"]))
            self.assertEqual(comparison["aacProfile"]["sceneEventNames"], ["TriCreature_S1_0", "TriCreature_S1_1", "TriCreature_S1_2"])

            json_path, text_path = converter.write_ocreatures_reference_comparison_reports(comparison, root / "reports")
            self.assertTrue(json_path.exists())
            self.assertTrue(text_path.exists())
            self.assertIn("OCreatures Reference Comparison", text_path.read_text(encoding="utf-8"))

    def test_flufyfox_style_creature_only_regression_public_reports_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Creature Regression.zip"
            animations = []
            roots = ["canine", "dlc01/chaurusflyer", "troll"]
            scene_total = 21
            for index in range(1, scene_total + 1):
                actor_root = roots[(index - 1) % len(roots)]
                scene_id = f"CreatureScene{index}"
                animations.append(
                    {
                        "id": scene_id,
                        "name": f"Creature Scene {index}",
                        "tags": f"MF,sex,anal,Creature,{actor_root},Bestiality",
                        "actors": [
                            {"type": "Female", "stages": [{"id": f"{scene_id}_A1_S1"}]},
                            {"type": "CreatureMale", "race": actor_root, "stages": [{"id": f"{scene_id}_A2_S1"}]},
                        ],
                    }
                )

            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/json/CreatureRegression.json", json.dumps({"name": "CreatureRegression", "animations": animations}))
                for index in range(1, scene_total + 1):
                    actor_root = roots[(index - 1) % len(roots)]
                    scene_id = f"CreatureScene{index}"
                    source.writestr(f"meshes/actors/character/animations/CreatureRegression/{scene_id}_A1_S1.hkx", b"")
                    source.writestr(f"meshes/actors/{actor_root}/animations/CreatureRegression/{scene_id}_A2_S1.hkx", b"")

            fingerprint = converter.source_archive_fingerprint(archive)["sha256"]
            db_path = root / "compatibility_db.json"
            db_path.write_text(
                json.dumps(
                    {
                        "version": "test",
                        "entries": [
                            {
                                "id": "flufyfox-style-creature-regression",
                                "packDisplayName": "FlufyFox-style Creature Regression",
                                "sourceFramework": "sexlabSlal",
                                "sourceTypeCodes": ["sexlabSlal"],
                                "archiveFingerprints": [fingerprint],
                                "recommendedOutputType": "OStim Standalone",
                                "recommendedBehaviorTool": "Pandora creature-capable output",
                                "defaultHumanOnlyOStim": False,
                                "creatureOutputRecommended": True,
                                "creatureRuntimeRequired": True,
                                "status": "needs creature runtime",
                                "knownWarnings": [
                                    "Creature-only source archive: generic human-only OStim output would remove all scenes.",
                                    "Creature output requires OCreatures or another OStim creature extension, Creature Framework, matching creature assets, and creature-capable behavior generation.",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(converter, "compatibility_database_paths", return_value=[db_path]):
                result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True)

                self.assertIsNotNone(result.verification)
                self.assertTrue(result.verification.ok, result.verification.report["errors"])
                self.assertEqual(result.verification.report["status"], "PASS WITH WARNINGS")
                self.assertEqual(result.report["status"], "PASS WITH WARNINGS")
                self.assertFalse(result.report["humanOnlyOStimOutput"])
                self.assertEqual(result.report["creatureSceneCount"], scene_total)
                self.assertEqual(result.report["retainedHumanSceneCount"], 0)
                self.assertEqual(
                    result.report["behaviorGeneration"]["outputMode"],
                    converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE_WITH_CREATURE,
                )
                self.assertEqual(converter.recommended_behavior_tool_from_report(result.report), "Pandora")

                menu_report = result.report["menuHubCategoryGeneration"]
                self.assertIn("all", menu_report["categoriesGenerated"])
                self.assertIn("creature", menu_report["categoriesGenerated"])
                self.assertEqual(menu_report["hubLinkableSceneCount"], scene_total)
                self.assertEqual(menu_report["hubLinkedSceneCount"], scene_total)
                self.assertEqual(menu_report["hubUnlinkedSceneCount"], 0)

                creature_menu = result.report["creatureMenuIntegration"]
                self.assertTrue(creature_menu["creatureOnlyPack"])
                self.assertTrue(creature_menu["ocreaturesMenuEntryGenerated"])
                self.assertEqual(
                    creature_menu["creatureMenuIntegrationMode"],
                    converter.CREATURE_MENU_INTEGRATION_OCREATURES_MENU_PLUS_HUMAN_DEBUG,
                )
                self.assertEqual(
                    creature_menu["recommendedCreatureMenuIntegrationMode"],
                    converter.CREATURE_MENU_INTEGRATION_OCREATURES_MENU,
                )
                self.assertEqual(creature_menu["creatureScenesWritten"], scene_total)
                self.assertEqual(creature_menu["creatureScenesReachableFromOCreaturesMenu"], scene_total)
                self.assertEqual(creature_menu["creatureScenesNotReachableFromOCreaturesMenu"], 0)
                self.assertGreater(creature_menu["creatureMenuRootNodeCount"], 0)
                self.assertGreater(creature_menu["creatureMenuCategoryNodeCount"], 0)
                self.assertIn("OCrCanine", creature_menu["ocreaturesMenuFolders"])
                self.assertIn("OCrTroll", creature_menu["ocreaturesMenuFolders"])

                installability = converter.installability_summary(result.verification.report)
                self.assertEqual(installability["installable"], "Yes, with warnings")
                self.assertIn("creature runtime requirements", installability["reason"])

                with ZipFile(result.zip_path) as converted:
                    manifest_entries = [
                        name
                        for name in set(converted.namelist())
                        if PurePosixPath(name).name == converter.AAC_MANIFEST_FILE
                    ]
                    self.assertTrue(manifest_entries)
                    for entry in manifest_entries:
                        manifest = json.loads(converted.read(mod_entry(entry)).decode("utf-8"))
                        self.assertEqual(manifest["verificationResult"]["status"], "PASS WITH WARNINGS")
                        self.assertTrue(manifest["verificationResult"]["ok"])
                        self.assertNotEqual(manifest["verificationResult"]["status"], "NOT RUN")
                        self.assertTrue(manifest["creatureMenuIntegration"]["ocreaturesMenuEntryGenerated"])
                    names = set(converted.namelist())
                    self.assertTrue(
                        any(name.startswith("Data/SKSE/Plugins/OStim/scenes/OCreatures/OCrCanine/") for name in names)
                    )
                    self.assertTrue(
                        any(name.startswith("Data/SKSE/Plugins/OStim/scenes/OCreatures/OCrTroll/") for name in names)
                    )

                public_report_path = result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report.json")
                public_report_text = public_report_path.read_text(encoding="utf-8")
                self.assertIn('"publicSafePathMode": true', public_report_text)
                self.assertIn('"zipFilename"', public_report_text)
                self.assertNotRegex(public_report_text, r"[A-Za-z]:[\\/]")
                self.assertNotIn(str(root), public_report_text)

                public_verify_json = json.loads(result.verification.report_path.read_text(encoding="utf-8"))
                self.assertEqual(public_verify_json["installabilityResult"], "Yes, with warnings")
                self.assertIn("creature runtime requirements", public_verify_json["installabilityReason"])
                self.assertTrue(public_verify_json["creatureMenuIntegration"]["ocreaturesMenuEntryGenerated"])
                self.assertEqual(
                    public_verify_json["creatureMenuIntegration"]["creatureScenesReachableFromOCreaturesMenu"],
                    scene_total,
                )

                verify_text = result.verification.text_report_path.read_text(encoding="utf-8")
                self.assertIn("Scene JSON files written:", verify_text)
                self.assertIn("Deployable/playable scenes:", verify_text)
                self.assertIn("Menu/helper scenes:", verify_text)
                self.assertIn("Creature Runtime Requirements:", verify_text)
                self.assertIn("OCreatures Menu Integration:", verify_text)
                self.assertIn("OCreatures menu entry generated: yes", verify_text)
                self.assertIn("Creature-only menu:", verify_text)
                self.assertIn("Recommended behavior tool: Pandora", verify_text)
                self.assertNotIn("Pandora native registration..", verify_text)
                self.assertNotIn("Pandora/Nemesis", verify_text)

                runtime_warnings = [
                    warning
                    for warning in result.verification.report["warnings"]
                    if converter.warning_code(warning) == "creature_runtime_requirements"
                ]
                self.assertEqual(len(runtime_warnings), 1)

                with self.assertRaises(RuntimeError) as raised:
                    converter.convert_archive_to_ready_zip(
                        archive,
                        zip_path=root / "Creature Regression Human Only.zip",
                        ostim_menu_entry=True,
                        human_only_ostim=True,
                    )
                self.assertIn("Human-only OStim output was enabled", str(raised.exception))
                self.assertIn("every playable scene was creature or mixed creature content", str(raised.exception))

    def test_verify_fails_creature_only_zip_without_ocreatures_menu_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Creature Missing Menu.zip"
            animations = [
                {
                    "id": "CanineScene",
                    "name": "Canine Scene",
                    "tags": "MF,sex,Creature,Canine",
                    "actors": [
                        {"type": "Female", "stages": [{"id": "CanineScene_A1_S1"}]},
                        {"type": "CreatureMale", "race": "canine", "stages": [{"id": "CanineScene_A2_S1"}]},
                    ],
                }
            ]
            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/json/MissingMenu.json", json.dumps({"name": "MissingMenu", "animations": animations}))
                source.writestr("meshes/actors/character/animations/MissingMenu/CanineScene_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/MissingMenu/CanineScene_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=False, human_only_ostim=False)
            self.assertTrue(result.verification.ok, result.verification.report["errors"])

            broken_zip = root / "Creature Missing Menu Broken.zip"
            with ZipFile(result.zip_path) as source, ZipFile(broken_zip, "w") as broken:
                for info in source.infolist():
                    data_rel = converter.zip_entry_data_relative(info.filename)
                    rel = data_rel[0] if data_rel else PurePosixPath(info.filename)
                    ocreatures_rel = converter.path_after_prefix(rel, converter.OCREATURES_SCENES_REL)
                    if ocreatures_rel is not None and ocreatures_rel.parts and ocreatures_rel.parts[0].startswith("OCr"):
                        continue
                    broken.writestr(info, source.read(info.filename))

            verification = converter.verify_ostim_converted_zip(broken_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(
                any("no OCreatures menu entry" in error or "no OCreatures menu entry was created" in error for error in verification.report["errors"]),
                verification.report["errors"],
            )
            self.assertFalse(verification.report["creatureMenuIntegration"]["ocreaturesMenuEntryGenerated"])

    def test_synthetic_working_ocreatures_folder_layout_is_recognized(self):
        playable = converter.Scene(
            raw_id="WorkingCanineScene",
            scene_id="WorkingCanineScene",
            name="Working Canine Scene",
            speeds=[converter.Speed("WorkingCanineEvent")],
            actors=[
                converter.Actor(intended_sex="female", type="Female", animation_index=0),
                converter.Actor(intended_sex="male", type="CreatureMale", animation_index=1, creature_race="Canines", tags=["creature:canines"]),
            ],
            actions=[converter.Action(type="analsex", actor=0, target=1)],
        )
        category = converter.Scene(
            raw_id="WorkingOCrCanineCategory",
            scene_id="WorkingOCrCanineCategory",
            name="Creature",
            speeds=[converter.Speed("WorkingCanineEvent")],
            actors=copy.deepcopy(playable.actors),
            no_random_selection=True,
            tags=[
                converter.OSTIM_MENU_HUB_TAG,
                converter.OCREATURES_MENU_TAG,
                f"{converter.OCREATURES_MENU_FOLDER_TAG_PREFIX}OCrCanine",
                f"{converter.OCREATURES_MENU_CATEGORY_TAG_PREFIX}creature",
            ],
            navigations=[converter.Navigation(destination="WorkingCanineScene")],
        )
        root = converter.Scene(
            raw_id="WorkingOCrCanineRoot",
            scene_id="WorkingOCrCanineRoot",
            name="Working Pack",
            speeds=[converter.Speed("WorkingCanineEvent")],
            actors=copy.deepcopy(playable.actors),
            no_random_selection=True,
            tags=[
                converter.OSTIM_MENU_HUB_TAG,
                converter.OCREATURES_MENU_TAG,
                converter.OCREATURES_MENU_ROOT_TAG,
                f"{converter.OCREATURES_MENU_FOLDER_TAG_PREFIX}OCrCanine",
            ],
            navigations=[converter.Navigation(destination="WorkingOCrCanineCategory")],
        )

        report = converter.ocreatures_menu_integration_report([playable, category, root], "Working Pack")
        self.assertEqual(report["creatureMenuIntegrationMode"], converter.CREATURE_MENU_INTEGRATION_OCREATURES_MENU)
        self.assertTrue(report["ocreaturesMenuEntryGenerated"])
        self.assertEqual(report["ocreaturesMenuFolders"], ["OCrCanine"])
        self.assertEqual(report["creatureScenesReachableFromOCreaturesMenu"], 1)
        self.assertEqual(report["creatureScenesNotReachableFromOCreaturesMenu"], 0)

    def test_large_slal_pack_uses_reachable_category_page_menu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Large SLAL Menu.zip"
            animations = []
            hkx_roots: dict[str, str] = {}

            def add_animation(animation_id: str, tags: str, actor2_type: str = "Male", actor2_root: str = "character", furniture: str = "") -> None:
                animation = {
                    "id": animation_id,
                    "name": "Repeated Display Name" if len(animations) % 7 == 0 else f"Menu Fixture {len(animations):02d}",
                    "tags": tags,
                    "actors": [
                        {"type": "Female", "stages": [{"id": f"{animation_id}_A1_S1"}]},
                        {"type": actor2_type, "stages": [{"id": f"{animation_id}_A2_S1"}]},
                    ],
                }
                if furniture:
                    animation["furniture"] = furniture
                animations.append(animation)
                hkx_roots[f"{animation_id}_A1_S1"] = "character"
                hkx_roots[f"{animation_id}_A2_S1"] = actor2_root

            for index in range(45):
                if index % 5 == 0:
                    tags = ""
                elif index % 5 == 1:
                    tags = "MF,oral"
                elif index % 5 == 2:
                    tags = "MF,vaginal"
                elif index % 5 == 3:
                    tags = "MF,anal"
                else:
                    tags = "MF,sex"
                add_animation(f"HumanScene{index:02d}", tags)
            for index in range(4):
                add_animation(f"BenchScene{index:02d}", "MF,sex,furniture,bench,anal", furniture="bench")
            for index in range(4):
                add_animation(f"CanineScene{index:02d}", "MF,sex,Creature,Canine,Bestiality,anal", actor2_type="CreatureMale", actor2_root="canine")

            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/json/LargeMenu.json", json.dumps({"name": "LargeMenu", "animations": animations}))
                for event, actor_root in hkx_roots.items():
                    source.writestr(f"meshes/actors/{actor_root}/animations/LargeMenu/{event}.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True, human_only_ostim=False)
            menu_report = result.report["menuHubCategoryGeneration"]

            self.assertEqual(result.report["deployableSceneCount"], len(animations))
            self.assertGreater(result.report["ostimMenuHubSceneCount"], 1)
            self.assertGreaterEqual(menu_report["categoryNodeCount"], 5)
            self.assertGreaterEqual(menu_report["pageNodeCount"], 3)
            self.assertIn("all", menu_report["categoriesGenerated"])
            self.assertIn("furniture", menu_report["categoriesGenerated"])
            self.assertIn("creature", menu_report["categoriesGenerated"])
            self.assertEqual(menu_report["hubLinkableSceneCount"], len(animations))
            self.assertEqual(menu_report["hubLinkedSceneCount"], len(animations))
            self.assertEqual(menu_report["hubUnlinkedSceneCount"], 0)
            self.assertEqual(menu_report["missingMenuDestinationCount"], 0)
            self.assertEqual(menu_report["duplicateMenuNavigationCount"], 0)
            self.assertEqual(result.report["ostimMenuReachableDeployableSceneCount"], len(animations))
            self.assertEqual(result.report["ostimMenuUnreachableDeployableSceneCount"], 0)
            self.assertGreaterEqual(result.report["generatedOStimReturnNavigationCount"], len(animations))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                metadata = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/converter_metadata/Large_SLAL_Menu/metadata.json"
                    ).decode("utf-8")
                )
                menu_json_names = [
                    name
                    for name in names
                    if name.startswith("Data/SKSE/Plugins/OStim/scenes/Large_SLAL_Menu/")
                    and "_Menu_" in Path(name).stem
                ]

            self.assertTrue(any("_Page_" in Path(name).stem for name in menu_json_names))
            self.assertEqual(metadata["deployment"]["status"], "PASS")
            self.assertEqual(metadata["deployment"]["ostimMenuReachableDeployableSceneCount"], len(animations))
            self.assertEqual(metadata["deployment"]["ostimMenuUnreachableDeployableSceneCount"], 0)
            self.assertEqual(metadata["deployment"]["ostimMenuPageNodeCount"], menu_report["pageNodeCount"])
            self.assertTrue(metadata["checks"]["hasOStimMenuDeployableCoverage"])
            self.assertTrue(metadata["checks"]["hasOStimMenuCategories"])
            self.assertTrue(metadata["checks"]["hasOStimMenuPages"])

            report_text = converter.report_to_text(result.report)
            self.assertIn("Menu Hub / Category Generation:", report_text)
            self.assertIn(f"- Linked scenes: {len(animations)}", report_text)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["hubLinkedSceneCount"], len(animations))
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["hubUnlinkedSceneCount"], 0)
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["missingMenuDestinationCount"], 0)
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["duplicateMenuNavigationCount"], 0)

    def test_slal_lifecycle_tracks_sparse_tags_and_other_category_visibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Sparse Tags SLAL.zip"
            animations = []

            with ZipFile(archive, "w") as source:
                for index in range(converter.OSTIM_MENU_DIRECT_LINK_LIMIT + 1):
                    animation_id = f"SparseScene{index:02d}"
                    animations.append(
                        {
                            "id": animation_id,
                            "name": f"Sparse Scene {index:02d}",
                            "tags": "",
                            "actors": [
                                {"type": "Female", "stages": [{"id": f"{animation_id}_A1_S1"}]},
                                {"type": "Male", "stages": [{"id": f"{animation_id}_A2_S1"}]},
                            ],
                        }
                    )
                    source.writestr(f"meshes/actors/character/animations/SparseTags/{animation_id}_A1_S1.hkx", b"")
                    source.writestr(f"meshes/actors/character/animations/SparseTags/{animation_id}_A2_S1.hkx", b"")
                source.writestr("SLAnims/json/SparseTags.json", json.dumps({"name": "SparseTags", "animations": animations}))

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True)
            lifecycle = result.report["sexlabSceneLifecycle"]
            visibility = result.report["menuHubCategoryVisibility"]

            self.assertEqual(len(lifecycle), len(animations))
            self.assertTrue(all(row["ostimSceneWritten"] for row in lifecycle))
            self.assertTrue(all(row["menuLinked"] for row in lifecycle))
            self.assertTrue(all("other" in row["classifiedCategories"] for row in lifecycle))
            self.assertIn("other", result.report["menuHubCategoryGeneration"]["categoriesGenerated"])
            self.assertIn("other", visibility["fallbackCategoriesUsed"])
            self.assertEqual(visibility["scenesLinkedFromGeneratedHub"], len(animations))
            self.assertEqual(visibility["finalMenuVisibilityResult"], "PASS")

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["menuHubCategoryVisibility"]["finalMenuVisibilityResult"], "PASS")
            self.assertEqual(verification.report["menuHubCategoryGeneration"]["duplicateMenuNavigationCount"], 0)

    def test_verify_rejects_empty_visible_ostim_menu_hub(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "Empty Visible Menu.zip"
            pack = "EmptyVisibleMenu"
            play_scene = converter.Scene(
                raw_id="Playable",
                scene_id=f"{pack}_Playable",
                name="Playable Scene",
                modpack=pack,
                speeds=[converter.Speed("PlayableLoop")],
                actors=[
                    converter.Actor(type="Male", intended_sex="male", animation_index=0),
                    converter.Actor(type="Female", intended_sex="female", animation_index=1),
                ],
                actions=[converter.Action(type="vaginalsex", actor=0, target=1)],
            )
            menu_scene = converter.ostim_menu_node_scene(
                pack,
                f"{pack}_Menu_MF",
                f"{pack}|ostim_menu|mf",
                pack,
                "mf",
                "OStim2PStandingApartMF",
                (converter.OSTIM_MENU_ROOT_TAG, "signature:mf"),
            )
            menu_scene.navigations.append(
                converter.Navigation(
                    origin="OStim2PStandingApartMF",
                    priority=converter.OSTIM_MENU_ENTRY_PRIORITY,
                    description=pack,
                    icon=converter.DEFAULT_OSTIM_MENU_ICON,
                    no_warnings=True,
                )
            )
            menu_scene.navigations.append(
                converter.Navigation(
                    destination="OStim2PStandingApartMF",
                    priority=-1000,
                    description=converter.OSTIM_MENU_RETURN_DESCRIPTION,
                    icon="OStim/symbols/return",
                    no_warnings=True,
                )
            )
            nemesis_code = converter.safe_nemesis_patch_code(pack)
            speed_var = f"{converter.sanitize_name(nemesis_code, 'ConvertedPack').upper()}_AnimationSpeed"

            with ZipFile(bad_zip, "w") as archive:
                archive.writestr(f"Data/SKSE/Plugins/OStim/scenes/{pack}/Playable.json", json.dumps(play_scene.to_json(pack)))
                archive.writestr(f"Data/SKSE/Plugins/OStim/scenes/{pack}/Menu.json", json.dumps(menu_scene.to_json(pack)))
                archive.writestr(f"Data/meshes/actors/character/animations/{pack}/PlayableLoop_0.hkx", b"")
                archive.writestr(f"Data/meshes/actors/character/animations/{pack}/PlayableLoop_1.hkx", b"")
                archive.writestr(
                    f"Data/meshes/actors/character/animations/{pack}/ATT_{nemesis_code}_animlist.txt",
                    f"b -Tn PlayableLoop_0 PlayableLoop_0.hkx {speed_var}: 1\n"
                    f"b -Tn PlayableLoop_1 PlayableLoop_1.hkx {speed_var}: 1\n",
                )
                archive.writestr(
                    f"Data/Nemesis_Engine/mod/{nemesis_code}/info.ini",
                    f"name={pack}\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    f"Data/Nemesis_Engine/mod/{nemesis_code}/0_master/#0106.txt",
                    "<hkobject><hkcstring>PlayableLoop_0</hkcstring><hkcstring>PlayableLoop_1</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("Generated OStim menu is empty" in error for error in verification.report["errors"]))
            self.assertTrue(verification.report["menuHubCategoryVisibility"]["emptyVisibleMenuFailure"])
            self.assertEqual(verification.report["menuHubCategoryVisibility"]["finalMenuVisibilityResult"], "FAIL")

    def test_mixed_k4_style_creature_and_furniture_pack_has_menu_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "K4 Style Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "SLAnims/json/K4.json",
                    json.dumps(
                        {
                            "name": "K4_Test",
                            "animations": [
                                {
                                    "id": "cowgirldg",
                                    "name": "K4 Enthused Cowgirl",
                                    "tags": "MF,sex,vaginal,anal",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "cowgirldg_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "cowgirldg_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "roughbench",
                                    "name": "K4 Bench Rough Anal",
                                    "tags": "MF,sex,furniture,bench,anal",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "roughbench_A1_S1"}]},
                                        {"type": "Male", "stages": [{"id": "roughbench_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "k4dogds",
                                    "name": "K4 Dog Doggystyle Anal",
                                    "tags": "MF,sex,Creature,Canine,Bestiality,anal",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "k4dogds_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "k4dogds_A2_S1"}]},
                                    ],
                                },
                                {
                                    "id": "k4benchriek",
                                    "name": "K4 Riekling Anal Bench",
                                    "tags": "MF,sex,Creature,Riekling,Bestiality,furniture,bench,anal",
                                    "actors": [
                                        {"type": "Female", "stages": [{"id": "k4benchriek_A1_S1"}]},
                                        {"type": "CreatureMale", "stages": [{"id": "k4benchriek_A2_S1"}]},
                                    ],
                                },
                            ],
                        }
                    ),
                )
                source.writestr("meshes/actors/character/animations/K4/cowgirldg_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/K4/cowgirldg_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/K4/roughbench_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/K4/roughbench_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/K4/k4dogds_A1_S1.hkx", b"")
                source.writestr("meshes/actors/canine/animations/K4/k4dogds_A2_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/K4/k4benchriek_A1_S1.hkx", b"")
                source.writestr("meshes/actors/dlc02/riekling/animations/K4/k4benchriek_A2_S1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, ostim_menu_entry=True, human_only_ostim=False)

            self.assertEqual(result.report["deployableSceneCount"], 4)
            self.assertEqual(result.report["furnitureSceneCount"], 2)
            self.assertEqual(result.report["creatureSceneCount"], 2)
            self.assertEqual(result.report["ostimMenuHubSceneCount"], 1)
            self.assertEqual(result.report["ostimMenuReachableDeployableSceneCount"], 4)
            self.assertEqual(result.report["ostimMenuUnreachableDeployableSceneCount"], 0)
            self.assertEqual(result.report["generatedOStimReturnNavigationCount"], 4)
            self.assertEqual(result.report["generatedOStimSequenceNavigationCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                cowgirl_scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/K4_Style_Pack/K4_Style_Pack_cowgirldg.json"
                    ).decode("utf-8")
                )
                dog_scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/OCreatures/K4_Style_Pack/K4_Style_Pack_k4dogds.json"
                    ).decode("utf-8")
                )
                riekling_scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/OCreatures/K4_Style_Pack/K4_Style_Pack_k4benchriek.json"
                    ).decode("utf-8")
                )
                hub = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/K4_Style_Pack/K4_Style_Pack_Menu_MF.json"
                    ).decode("utf-8")
                )
                metadata = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/converter_metadata/K4_Style_Pack/metadata.json"
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "K4_Style_Pack")
                assert_pandora_compatible_registration(self, names, "K4_Style_Pack", "dogproject")
                assert_pandora_compatible_registration(self, names, "K4_Style_Pack", "wolfproject")
                assert_pandora_compatible_registration(self, names, "K4_Style_Pack", "rieklingproject")
                behavior_events = read_pandora_compatible_events(converted, names, "K4_Style_Pack")
                nemesis_code = converter.safe_nemesis_patch_code(result.pack)

            self.assertIn("Data/SKSE/Plugins/OStim/scenes/K4_Style_Pack/K4_Style_Pack_roughbench.json", names)
            self.assertIn("Data/meshes/actors/character/animations/K4_Style_Pack/cowgirldg_S1_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/K4_Style_Pack/cowgirldg_S1_1.hkx", names)
            self.assertNotIn("Data/meshes/actors/character/animations/K4_Style_Pack/FNIS_K4_Style_Pack_List.txt", names)
            self.assertFalse(any(name.startswith("Data/animdata/") for name in names))
            self.assertFalse(any(name.startswith("Data/animationsetdatasinglefile/") for name in names))
            self.assertEqual([speed["animation"] for speed in cowgirl_scene["speeds"]], ["cowgirldg_S1"])
            self.assertEqual(len(character_att_list_paths(names, "K4_Style_Pack")), 1)
            self.assertTrue(any(name.startswith(f"Data/Nemesis_Engine/mod/{nemesis_code}/") for name in names))
            self.assertIn("cowgirldg_S1_0", behavior_events)
            self.assertIn("cowgirldg_S1_1", behavior_events)
            self.assertEqual(dog_scene["actors"][0]["creatureRace"], "Canines")
            self.assertEqual(riekling_scene["actors"][0]["creatureRace"], "Rieklings")
            self.assertEqual(riekling_scene["furniture"], "bench")
            hub_destinations = {
                nav.get("destination")
                for nav in hub["navigations"]
                if nav.get("destination") and nav.get("destination") not in converter.OSTIM_EXTERNAL_MENU_ORIGINS
            }
            self.assertEqual(
                hub_destinations,
                {
                    "K4_Style_Pack_cowgirldg",
                    "K4_Style_Pack_roughbench",
                    "K4_Style_Pack_k4dogds",
                    "K4_Style_Pack_k4benchriek",
                },
            )
            cowgirl_navs = cowgirl_scene.get("navigations") or []
            self.assertIn(
                {
                    "destination": "K4_Style_Pack_Menu_MF",
                    "priority": converter.OSTIM_MENU_ENTRY_PRIORITY + 20,
                    "description": converter.OSTIM_MENU_RETURN_DESCRIPTION,
                    "icon": converter.DEFAULT_OSTIM_MENU_ICON,
                },
                cowgirl_navs,
            )
            self.assertEqual(metadata["deployment"]["status"], "PASS")
            self.assertEqual(metadata["deployment"]["ostimMenuReachableDeployableSceneCount"], 4)
            self.assertEqual(metadata["deployment"]["ostimMenuUnreachableDeployableSceneCount"], 0)
            self.assertEqual(metadata["deployment"]["generatedOStimReturnNavigationCount"], 4)
            self.assertEqual(metadata["deployment"]["generatedOStimSequenceNavigationCount"], 0)
            self.assertTrue(metadata["checks"]["hasOStimMenuDeployableCoverage"])
            self.assertTrue(metadata["checks"]["hasGeneratedOStimReturnNavigation"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(
                verification.report["behaviorOutputMode"],
                converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE_WITH_CREATURE,
            )
            self.assertEqual(verification.report["duplicateAnimationListFileCount"], 0)
            self.assertEqual(verification.report["duplicateAnimationDataFileCount"], 0)
            self.assertEqual(verification.report["duplicateAnimationSetDataFileCount"], 0)
            self.assertEqual(verification.report["duplicateBehaviorGraphFileCount"], 0)
            self.assertEqual(verification.report["behaviorRegistrationSummary"]["finalBehaviorValidationResult"], "PASS")
            self.assertEqual(verification.report["ostimMenuReachableDeployableSceneCount"], 4)
            self.assertEqual(verification.report["ostimMenuUnreachableDeployableSceneCount"], 0)
            self.assertEqual(verification.report["generatedOStimReturnNavigationCount"], 4)
            self.assertEqual(verification.report["generatedOStimSequenceNavigationCount"], 0)
            self.assertEqual(verification.report["creatureActorRoots"], ["canine", "dlc02/riekling"])

    def test_sexlab_name_inference_handles_69_and_nipple_licking(self):
        oral = {
            "id": "MNBV_69",
            "name": "MNBV 69",
            "tags": "Moon,lesbian,FF",
            "actors": [{"type": "Female"}, {"type": "Female"}],
        }
        mixed_oral = {
            "id": "MNBV_MF69",
            "name": "MNBV MF 69",
            "tags": "MF",
            "actors": [{"type": "Female"}, {"type": "Male"}],
        }
        nipple = {
            "id": "MNBV_NippleLicking",
            "name": "MNBV NippleLicking",
            "tags": "Moon,lesbian,FF",
            "actors": [{"type": "Female"}, {"type": "Female"}],
        }

        self.assertEqual(converter.infer_sexlab_action_types(oral), ["cunnilingus"])
        self.assertEqual(converter.infer_sexlab_action_types(mixed_oral), ["cunnilingus", "blowjob"])
        self.assertEqual(converter.infer_sexlab_action_types(nipple), ["lickingnipple"])

    def test_slsb_normalizes_legacy_sexlab_race_keys(self):
        self.assertEqual(converter.slsb_normalize_race_name("Canines"), "Canine")
        self.assertEqual(converter.slsb_normalize_race_name("Horses"), "Horse")
        self.assertEqual(converter.slsb_normalize_race_name("FlameAtronach"), "Flame Atronach")
        self.assertEqual(converter.slsb_normalize_race_name("DragonPriests"), "Dragon Priest")

    def test_sexlab_foot_position_tags_do_not_force_vaginalsex(self):
        feet_on_face = {
            "id": "Drago_Feetonface",
            "name": "Drago Feetonface",
            "tags": "Drago,feet,cowgirl,mounted,feetonface,MF",
            "actors": [{"type": "Female"}, {"type": "Male"}],
        }
        feet_on_chest = {
            "id": "Drago_FeetonChest",
            "name": "Drago FeetonChest",
            "tags": "Drago,feet,cowgirl,mounted,MF",
            "actors": [{"type": "Female"}, {"type": "Male"}],
        }
        holding_feet_sex = {
            "id": "Drago_HoldingFeetSex",
            "name": "Drago HoldingFeetSex",
            "tags": "Drago,feet,Kneeling,Wheelbarrow,Sex,HoldingFeet,MF",
            "actors": [{"type": "Female"}, {"type": "Male"}],
        }

        self.assertEqual(converter.infer_sexlab_action_types(feet_on_face), ["holdingfoot"])
        self.assertEqual(converter.infer_sexlab_action_types(feet_on_chest), ["holdingfoot"])
        self.assertEqual(converter.infer_sexlab_action_types(holding_feet_sex), ["vaginalsex", "holdingfoot"])

    def test_compiled_sexlab_registry_without_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Compiled SexLab Registry.zip"
            with ZipFile(archive, "w") as source:
                source.writestr("SKSE/SexLab/Registry/Ace_Test.slr", b"compiled registry")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")

            with self.assertRaisesRegex(RuntimeError, "SLAnims/json"):
                converter.convert_archive_to_ready_zip(archive)

    def test_slal_source_text_without_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SLAL Source Text.zip"
            with ZipFile(archive, "w") as source:
                source.writestr("SLAnims/source/Ace_Test.txt", "Animation(Ace_TestFootjob)\n")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A1_S1.hkx", b"")
                source.writestr("meshes/actors/character/animations/Ace_Test/Ace_TestFootjob_A2_S1.hkx", b"")

            with self.assertRaisesRegex(RuntimeError, "SLAnims/json"):
                converter.convert_archive_to_ready_zip(archive)

    def test_transition_without_animation_id_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Transition Only.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/Smoke.xml",
                    """\
<scene id="AA|Pos|HJ|Smoke" actors="2">
  <info name="Smoke" />
  <anim id="AA_HJ_Smoke" l="3" />
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/GoSmoke.xml",
                    """\
<scene id="AA|Pos|HJ|GoSmoke" actors="2">
  <info name="Go Smoke" />
  <anim t="T" l="1" dest="AA|Pos|HJ|Smoke" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Pos/HJ/AA_HJ_Smoke_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Pos/HJ/AA_HJ_Smoke_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 2)

            with ZipFile(result.zip_path) as converted:
                transition = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Transition_Only/Transition_Only_AA_Pos_HJ_GoSmoke.json"
                    ).decode("utf-8")
                )
            self.assertEqual(transition["destination"], "Transition_Only_AA_Pos_HJ_Smoke")
            self.assertEqual(transition["speeds"], [])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_duplicate_legacy_xml_attributes_are_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Duplicate Attr.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Dupe.xml",
                    """\
<scene id="AA|Standing|HJ|Dupe" actors="2">
  <info name="Duplicate Attr" />
  <anim id="DupeEvent" l="3" />
  <autonav>
    <fam KNy6MUy9="AA|Standing|HJ|Dupe" KNy6MUy9="AA|Standing|HJ|Dupe" />
  </autonav>
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/DupeEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/DupeEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 1)
            self.assertTrue(any("removed duplicate XML attribute" in warning for warning in result.warnings))

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_relative_navigation_refs_are_resolved_and_actions_are_inferred(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Relative Nav.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Base.xml",
                    """\
<scene id="AA|Standing|HJ|Base" actors="2">
  <info name="Base" />
  <anim id="BaseHandjob" l="3" />
  <nav><tab><page><option go="^+10A" /></page></tab></nav>
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Base+10A.xml",
                    """\
<scene id="AA|Standing|HJ|Base+10A" actors="2">
  <info name="Base Plus" />
  <anim id="BaseHandjobPlus" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjobPlus_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjobPlus_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)

            with ZipFile(result.zip_path) as converted:
                base = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Relative_Nav/Relative_Nav_AA_Standing_HJ_Base.json"
                    ).decode("utf-8")
                )
            self.assertEqual(base["navigations"][0]["destination"], "Relative_Nav_AA_Standing_HJ_Base+10A")
            self.assertEqual(base["actions"][0]["type"], "handjob")

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["unresolvedRelativeSceneLinkCount"], 0)

    def test_missing_external_transition_destination_is_repaired(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "External Destination.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Base.xml",
                    """\
<scene id="AA|Standing|HJ|Base" actors="2">
  <info name="Base" />
  <anim id="BaseHandjob" l="3" />
  <nav><tab><page><option go="^+10A" /></page></tab></nav>
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Base+10A.xml",
                    """\
<scene id="AA|Standing|HJ|Base+10A" actors="2">
  <info name="Base Plus" />
  <anim id="BaseHandjobPlus" t="T" l="3" dest="vanilla|AA|Standing|HJ|Missing" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjob_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjobPlus_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/BaseHandjobPlus_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertTrue(any("missing external transition destination" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                repaired = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/External_Destination/External_Destination_AA_Standing_HJ_Base+10A.json"
                    ).decode("utf-8")
                )
            self.assertNotIn("destination", repaired)
            self.assertTrue(repaired["noRandomSelection"])
            self.assertEqual(repaired["actions"][0]["type"], "handjob")

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingSceneLinkCount"], 0)

    def test_missing_non_transition_speeds_are_pruned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Missing Speeds.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/Mixed.xml",
                    """\
<scene id="AA|Standing|HJ|Mixed" actors="2">
  <info name="Mixed Speeds" />
  <speed>
    <sp qnt="1"><anim id="PresentEvent" /></sp>
    <sp qnt="2"><anim id="MissingEvent" /></sp>
  </speed>
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/HJ/OnlyMissing.xml",
                    """\
<scene id="AA|Standing|HJ|OnlyMissing" actors="2">
  <info name="Only Missing" />
  <speed>
    <sp qnt="1"><anim id="MissingOnly" /></sp>
  </speed>
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/PresentEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/HJ/PresentEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 1)
            self.assertTrue(any("skipped missing animation speed 'MissingEvent'" in warning for warning in result.warnings))
            self.assertTrue(any("dropped scene because all animation speeds were missing" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                mixed = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Missing_Speeds/Missing_Speeds_AA_Standing_HJ_Mixed.json"
                    ).decode("utf-8")
                )
            self.assertIn("Data/SKSE/Plugins/OStim/scenes/Missing_Speeds/Missing_Speeds_AA_Standing_HJ_Mixed.json", names)
            self.assertNotIn("Data/SKSE/Plugins/OStim/scenes/Missing_Speeds/Missing_Speeds_AA_Standing_HJ_OnlyMissing.json", names)
            self.assertEqual([speed["animation"] for speed in mixed["speeds"]], ["PresentEvent"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)

    def test_uncategorized_legacy_scene_gets_ostim_sfx_fallback_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "SFX Fallback.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Pose.xml",
                    """\
<scene id="AA|Standing|Ap|Pose" actors="2">
  <info name="Pose" />
  <anim id="PoseEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertTrue(result.report["usesConverterSfxFallback"])
            self.assertTrue(any("Added converter SFX fallback actions" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/SFX_Fallback/SFX_Fallback_AA_Standing_Ap_Pose.json"
                    ).decode("utf-8")
                )
                action = json.loads(
                    converted.read("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json").decode("utf-8")
                )
            self.assertIn("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json", names)
            self.assertEqual([action["type"] for action in scene["actions"]], ["ostimconvertermoan", "ostimconvertermoan"])
            self.assertTrue(action["actor"]["moan"])
            self.assertTrue(action["target"]["moan"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertTrue(verification.report["usesConverterSfxFallback"])
            self.assertEqual(verification.report["ostimActionFileCount"], 1)

    def test_legacy_po_action_uses_sfx_fallback_instead_of_unknown_masturbation_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "PO Fallback.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/PO/Perpendicular.xml",
                    """\
<scene id="AA|Standing|PO|Perpendicular" actors="1">
  <info name="Perpendicular" />
  <anim id="PoseSolo" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/PO/PoseSolo.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            with ZipFile(result.zip_path) as converted:
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/PO_Fallback/PO_Fallback_AA_Standing_PO_Perpendicular.json"
                    ).decode("utf-8")
                )
            self.assertEqual(scene["actions"][0]["type"], "ostimconvertermoan")
            self.assertNotEqual(scene["actions"][0]["type"], "masturbation")

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_custom_sfx_fallback_action_name_is_packaged_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Custom Fallback.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Pose.xml",
                    """\
<scene id="AA|Standing|Ap|Pose" actors="2">
  <info name="Pose" />
  <anim id="PoseEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, sfx_fallback_action="custom_moan")
            self.assertTrue(result.report["usesConverterSfxFallback"])
            self.assertEqual(result.report["sfxFallbackAction"], "custom_moan")
            self.assertTrue(result.report["usesGeneratedSfxFallbackActionFile"])

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Custom_Fallback/Custom_Fallback_AA_Standing_Ap_Pose.json"
                    ).decode("utf-8")
                )
            self.assertIn("Data/SKSE/Plugins/OStim/actions/custom_moan.json", names)
            self.assertNotIn("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json", names)
            self.assertEqual([action["type"] for action in scene["actions"]], ["custom_moan", "custom_moan"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertTrue(verification.report["usesConverterSfxFallback"])
            self.assertEqual(verification.report["sfxFallbackActions"], ["custom_moan"])

    def test_builtin_sfx_fallback_action_does_not_package_custom_action_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Builtin Fallback.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Pose.xml",
                    """\
<scene id="AA|Standing|Ap|Pose" actors="1">
  <info name="Pose" />
  <anim id="PoseEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, sfx_fallback_action="kissing")
            self.assertTrue(result.report["usesConverterSfxFallback"])
            self.assertEqual(result.report["sfxFallbackAction"], "kissing")
            self.assertFalse(result.report["usesGeneratedSfxFallbackActionFile"])

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Builtin_Fallback/Builtin_Fallback_AA_Standing_Ap_Pose.json"
                    ).decode("utf-8")
                )
            self.assertEqual([action["type"] for action in scene["actions"]], ["kissing"])
            self.assertNotIn("Data/SKSE/Plugins/OStim/actions/kissing.json", names)
            self.assertNotIn("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json", names)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertTrue(verification.report["usesConverterSfxFallback"])
            self.assertEqual(verification.report["sfxFallbackActions"], ["kissing"])

    def test_sfx_fallback_action_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "No Fallback.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Standing/Ap/Pose.xml",
                    """\
<scene id="AA|Standing|Ap|Pose" actors="1">
  <info name="Pose" />
  <anim id="PoseEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Standing/Ap/PoseEvent.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, sfx_fallback_action="none")
            self.assertFalse(result.report["usesConverterSfxFallback"])
            self.assertIsNone(result.report["sfxFallbackAction"])

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/No_Fallback/No_Fallback_AA_Standing_Ap_Pose.json"
                    ).decode("utf-8")
                )
            self.assertNotIn("actions", scene)
            self.assertNotIn("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json", names)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertFalse(verification.report["usesConverterSfxFallback"])
            self.assertEqual(verification.report["sfxFallbackActions"], [])

    def test_invalid_zero_length_is_repaired_instead_of_dropping_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Zero Length.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scenes/AA/pos/hj/Zero.xml",
                    """\
<scene id="AA|pos|hj|Zero" actors="2">
  <info name="Zero" />
  <anim id="ZeroEvent" l="0" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/pos/hj/ZeroEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/pos/hj/ZeroEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 1)
            self.assertTrue(any("using default length" in warning for warning in result.warnings))

            with ZipFile(result.zip_path) as converted:
                scene_json = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Zero_Length/Zero_Length_AA_pos_hj_Zero.json"
                    ).decode("utf-8")
                )
            self.assertEqual(scene_json["length"], converter.DEFAULT_LENGTH)
            self.assertEqual(scene_json["actions"][0]["type"], "handjob")

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_hkx_only_archive_converts_from_one_click_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "HKX Only Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr("Data/meshes/actors/character/animations/FNISOnly/PlainPose_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/FNISOnly/PlainPose_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/FNISOnly/FNIS_Fake_Behavior.hkx", b"old behavior")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 1)
            self.assertEqual(result.scenes[0].raw_id, "PlainPose")

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/HKX_Only_Pack/HKX_Only_Pack_PlainPose.json"
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "HKX_Only_Pack")
                behavior_events = read_pandora_compatible_events(converted, names, "HKX_Only_Pack")

            self.assertEqual(scene["speeds"][0]["animation"], "PlainPose")
            self.assertIn("Data/meshes/actors/character/animations/HKX_Only_Pack/FNISOnly/PlainPose_0.hkx", names)
            self.assertIn("Data/meshes/actors/character/animations/HKX_Only_Pack/FNISOnly/PlainPose_1.hkx", names)
            self.assertNotIn("Data/meshes/actors/character/animations/HKX_Only_Pack/FNISOnly/FNIS_Fake_Behavior.hkx", names)
            self.assertEqual(len(character_att_list_paths(names, "HKX_Only_Pack")), 1)
            self.assertIn("PlainPose_0", behavior_events)
            self.assertIn("PlainPose_1", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 0)

    def test_single_actor_unsuffixed_hkx_gets_behavior_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Solo Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Solo.xml",
                    """\
<scene id="AA|Solo" actors="1">
  <info name="Solo" />
  <anim id="SoloEvent" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/SoloEvent.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["missingAnimationEventCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "Solo_Pack")
                behavior_events = read_pandora_compatible_events(converted, names, "Solo_Pack")
            self.assertEqual(len(character_att_list_paths(names, "Solo_Pack")), 1)
            self.assertIn("SoloEvent", behavior_events)
            self.assertIn("SoloEvent_0", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_legacy_fnis_behavior_hkx_is_not_packaged_as_animation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Legacy Behavior Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/Smoke.xml",
                    """\
<scene id="AA|Pos|HJ|Smoke" actors="2">
  <info name="Smoke" />
  <anim id="AA_HJ_Smoke" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/Legacy/AA_HJ_Smoke_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/AA_HJ_Smoke_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/FNIS_Legacy_Behavior.hkx", b"")
                source.writestr("Data/meshes/actors/character/behaviors/FNIS_Legacy_Behavior.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.report["hkxCount"], 2)
            self.assertEqual(result.report["missingAnimationEventCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                assert_pandora_compatible_registration(self, names, "Legacy_Behavior_Pack")
                behavior_events = read_pandora_compatible_events(converted, names, "Legacy_Behavior_Pack")

            self.assertIn(
                "Data/meshes/actors/character/animations/Legacy/AA_HJ_Smoke_0.hkx",
                names,
            )
            self.assertNotIn("Data/meshes/actors/character/behaviors/FNIS_Legacy_Behavior.hkx", names)
            self.assertNotIn(
                "Data/meshes/actors/character/animations/Legacy/FNIS_Legacy_Behavior.hkx",
                names,
            )
            self.assertEqual(len(character_att_list_paths(names, "Legacy_Behavior_Pack")), 1)
            self.assertNotIn("FNIS_Legacy_Behavior.hkx", behavior_events)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_archive_one_click_conversion_defaults_everything(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Sample Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/Smoke.xml",
                    """\
<scene id="AA|Pos|HJ|Smoke" actors="2">
  <info name="Smoke" />
  <anim id="AA_HJ_Smoke" l="3" />
</scene>
""",
                )
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/SmokeTransition.xml",
                    """\
<scene id="AA|Pos|HJ|SmokeTransition" actors="2">
  <info name="Smoke Transition" />
  <anim id="MissingTransitionAnim" t="T" l="3" dest="AA|Pos|HJ|Smoke" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Pos/HJ/AA_HJ_Smoke_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/0Sex/AA/Pos/HJ/AA_HJ_Smoke_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive, mod_author="Original Author")
            self.assertEqual(result.pack, "Sample_Pack")
            self.assertEqual(result.mod_author, "Original Author")
            self.assertEqual(result.zip_path, root / "Sample Pack_OStimSA.zip")
            self.assertTrue(result.zip_path.exists())
            self.assertEqual(result.report["missingAnimationEventCount"], 0)
            self.assertEqual(result.report["modAuthor"], "Original Author")
            self.assertTrue(any("skipped missing transition animation" in warning for warning in result.report["warnings"]))

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                report = json.loads(converted.read("conversion_report.json").decode("utf-8"))
                metadata = json.loads(
                    converted.read("Data/SKSE/Plugins/OStim/converter_metadata/Sample_Pack/metadata.json").decode("utf-8")
                )
                transition = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Sample_Pack/Sample_Pack_AA_Pos_HJ_SmokeTransition.json"
                    ).decode("utf-8")
                )
                assert_pandora_compatible_registration(self, names, "Sample_Pack")
            self.assertIn("Data/SKSE/Plugins/OStim/scenes/Sample_Pack/Sample_Pack_AA_Pos_HJ_Smoke.json", names)
            self.assertEqual(len(character_att_list_paths(names, "Sample_Pack")), 1)
            self.assertFalse(any(name.startswith("Data/Pandora_Engine/") for name in names))
            self.assertEqual(transition["destination"], "Sample_Pack_AA_Pos_HJ_Smoke")
            self.assertEqual(transition["speeds"], [])
            self.assertEqual(report["sceneCount"], 2)
            self.assertEqual(report["hkxCount"], 2)
            self.assertEqual(report["missingAnimationEventCount"], 0)
            self.assertEqual(report["behaviorGeneration"]["outputMode"], converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE)
            self.assertEqual(report["postBuildVerification"]["status"], result.verification.report["status"])
            self.assertIn("Data/SKSE/Plugins/OStim/converter_metadata/Sample_Pack/metadata.json", names)
            self.assertEqual(metadata["pack"]["author"], "Original Author")
            self.assertEqual(metadata["deployment"]["sceneCount"], 2)
            self.assertEqual(metadata["checks"]["hasStartableScenes"], True)

            verification = converter.verify_converted_zip(result.zip_path)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertTrue(verification.report_path.exists())
            self.assertTrue(verification.text_report_path.exists())
            self.assertEqual(verification.report["sceneJsonFileCount"], 2)
            self.assertEqual(verification.report["converterMetadataCount"], 1)
            self.assertEqual(verification.report["converterMetadataPackNames"], ["Sample_Pack"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)
            self.assertEqual(verification.report["pandoraNamedModCount"], 0)
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_NEMESIS_SAFE)
            recognition = verification.report["pandoraRecognition"]
            self.assertEqual(recognition["status"], "not_applicable")
            self.assertFalse(recognition["expectedVisibleInPandora"])
            self.assertEqual(recognition["expectedModListNames"], [])
            self.assertEqual(recognition["expectedInfoFiles"], [])
            self.assertEqual(recognition["mods"], [])
            text_report = converter.verification_report_to_text(verification.report)
            self.assertIn("Behavior output mode: pandora_compatible", text_report)

    def test_osex_plus_archive_uses_scene_root_and_ignores_installer_xml(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "OSex Plus Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "fomod/ModuleConfig.xml",
                    "<config><moduleName>Installer</moduleName></config>",
                )
                source.writestr(
                    "Data/OSA+/mod/OSex+/scene/AA/Standing/HJ/PlusSmoke.xml",
                    """\
<scene id="AA|Standing|HJ|PlusSmoke" actorCount="2">
  <info title="Plus Smoke" />
  <anim id="OSexPlus_PlusSmoke" length="4.5" />
</scene>
""",
                )
                source.writestr(
                    "Data/OSA+/mod/OSex+/scene/BB/Sitting/BJ/PlusKiss.xml",
                    """\
<scene sceneId="BB|Sitting|BJ|PlusKiss" actors="2">
  <info name="Plus Kiss" />
  <speed>
    <sp qnt="1"><anim animId="OSexPlus_PlusKiss" /></sp>
  </speed>
</scene>
""",
                )
                source.writestr(
                    "Data/meshes/actors/character/animations/OSex+/AA/Standing/HJ/OSexPlus_PlusSmoke_0.hkx",
                    b"",
                )
                source.writestr(
                    "Data/meshes/actors/character/animations/OSex+/AA/Standing/HJ/OSexPlus_PlusSmoke_1.hkx",
                    b"",
                )
                source.writestr(
                    "Data/meshes/actors/character/animations/OSex+/BB/Sitting/BJ/OSexPlus_PlusKiss_0.hkx",
                    b"",
                )
                source.writestr(
                    "Data/meshes/actors/character/animations/OSex+/BB/Sitting/BJ/OSexPlus_PlusKiss_1.hkx",
                    b"",
                )

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(result.pack, "OSex_Plus_Pack")
            self.assertEqual(len(result.scenes), 2)
            self.assertEqual(result.report["missingAnimationEventCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                smoke = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/OSex_Plus_Pack/OSex_Plus_Pack_AA_Standing_HJ_PlusSmoke.json"
                    ).decode("utf-8")
                )

            self.assertIn(
                "Data/SKSE/Plugins/OStim/scenes/OSex_Plus_Pack/OSex_Plus_Pack_BB_Sitting_BJ_PlusKiss.json",
                names,
            )
            self.assertEqual(smoke["length"], 4.5)
            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_ostim_standalone_archive_round_trips_modern_scene_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Modern OStim Standalone Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/ModernPack/ModernScene.json",
                    json.dumps(
                        {
                            "name": "Modern Scene",
                            "modpack": "ModernPack",
                            "length": 8.5,
                            "defaultSpeed": 1,
                            "fadeOnEntry": True,
                            "furniture": "chair",
                            "scaleOffsetWithFurniture": True,
                            "tags": ["tease", "custom"],
                            "offset": {"x": 1.0, "z": 2.0, "r": 15.0},
                            "speeds": [
                                {"animation": "ModernLoop", "playbackSpeed": 1.25, "displaySpeed": 0.75},
                                {"animation": "ModernFast", "playbackSpeed": 1.5},
                            ],
                            "actors": [
                                {
                                    "type": "npc",
                                    "intendedSex": "female",
                                    "sosBend": 4,
                                    "scale": 1.05,
                                    "scaleHeight": 0.96,
                                    "animationIndex": 1,
                                    "underlyingExpression": "happy",
                                    "expressionAction": 2,
                                    "expressionOverride": "smile",
                                    "lookUp": 12,
                                    "noStrip": True,
                                    "feetOnGround": True,
                                    "offset": {"x": 3.0, "y": 4.0, "z": 5.0, "r": 6.0},
                                    "requirements": ["ModernRequirement"],
                                    "tags": ["lead"],
                                    "autoTransitions": {"climax": "ModernNext"},
                                },
                                {
                                    "type": "npc",
                                    "intendedSex": "male",
                                    "animationIndex": 0,
                                    "lookRight": 9,
                                },
                            ],
                            "actions": [
                                {
                                    "type": "customtease",
                                    "actor": 0,
                                    "target": 1,
                                    "muted": True,
                                    "doPeaks": False,
                                    "peaksAnnotated": True,
                                }
                            ],
                            "navigations": [
                                {
                                    "destination": "ModernNext",
                                    "priority": 7,
                                    "description": "Next scene",
                                    "icon": "heart",
                                    "border": "gold",
                                    "noWarnings": True,
                                }
                            ],
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/ModernPack/ModernNext.json",
                    json.dumps(
                        {
                            "name": "Modern Next",
                            "modpack": "ModernPack",
                            "length": 4.0,
                            "speeds": [{"animation": "ModernNextLoop"}],
                            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                            "actions": [{"type": "kissing", "actor": 0, "target": 1}],
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/actions/customtease.json",
                    json.dumps({"info": "Custom tease action", "tags": ["custom"]}),
                )
                for event in (
                    "ModernLoop_0",
                    "ModernLoop_1",
                    "ModernFast_0",
                    "ModernFast_1",
                    "ModernNextLoop_0",
                    "ModernNextLoop_1",
                ):
                    source.writestr(f"Data/meshes/actors/character/animations/ModernPack/{event}.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 2)
            self.assertEqual(len(result.ostim_action_files), 1)
            self.assertEqual(result.report["furnitureSceneCount"], 1)
            self.assertEqual(result.report["missingAnimationEventCount"], 0)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Modern_OStim_Standalone_Pack/Modern_OStim_Standalone_Pack_ModernScene.json"
                    ).decode("utf-8")
                )
                copied_action = json.loads(
                    converted.read("Data/SKSE/Plugins/OStim/actions/customtease.json").decode("utf-8")
                )

            self.assertEqual(scene["defaultSpeed"], 1)
            self.assertTrue(scene["fadeOnEntry"])
            self.assertEqual(scene["furniture"], "chair")
            self.assertTrue(scene["scaleOffsetWithFurniture"])
            self.assertEqual(scene["offset"], {"x": 1.0, "z": 2.0, "r": 15.0})
            self.assertEqual(scene["speeds"][0]["playbackSpeed"], 1.25)
            self.assertEqual(scene["speeds"][0]["displaySpeed"], 0.75)
            self.assertEqual(scene["actors"][0]["animationIndex"], 1)
            self.assertEqual(scene["actors"][0]["underlyingExpression"], "happy")
            self.assertEqual(scene["actors"][0]["expressionAction"], 2)
            self.assertEqual(scene["actors"][0]["expressionOverride"], "smile")
            self.assertEqual(scene["actors"][0]["requirements"], ["ModernRequirement"])
            self.assertEqual(scene["actors"][0]["autoTransitions"]["climax"], "Modern_OStim_Standalone_Pack_ModernNext")
            self.assertEqual(scene["actors"][1]["lookRight"], 9)
            self.assertEqual(scene["actions"][0]["type"], "customtease")
            self.assertTrue(scene["actions"][0]["muted"])
            self.assertFalse(scene["actions"][0]["doPeaks"])
            self.assertTrue(scene["actions"][0]["peaksAnnotated"])
            self.assertEqual(scene["navigations"][0]["destination"], "Modern_OStim_Standalone_Pack_ModernNext")
            self.assertEqual(scene["navigations"][0]["priority"], 7)
            self.assertEqual(scene["navigations"][0]["description"], "Next scene")
            self.assertEqual(scene["navigations"][0]["icon"], "heart")
            self.assertEqual(scene["navigations"][0]["border"], "gold")
            self.assertTrue(scene["navigations"][0]["noWarnings"])
            self.assertIn("Data/SKSE/Plugins/OStim/actions/customtease.json", names)
            self.assertTrue(copied_action["actor"]["moan"])
            self.assertTrue(copied_action["target"]["talk"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)

            sexlab_plus = converter.convert_archive_to_sexlab_zip(archive, sexlab_plus=True)
            self.assertEqual(sexlab_plus.report["target"], "SexLab P+/SLSB")
            self.assertEqual(sexlab_plus.report["animationCount"], 2)
            self.assertEqual(sexlab_plus.report["sexlabPlusCompiledRegistryCount"], 1)
            with ZipFile(sexlab_plus.zip_path) as converted:
                names = set(converted.namelist())
            self.assertIn("Data/SKSE/Sexlab/Registry/Modern_OStim_Standalone_Pack.slr", names)
            self.assertIn("Data/SKSE/Sexlab/Registry/Source/Modern_OStim_Standalone_Pack.slsb.json", names)

    def test_legacy_ostim_json_archive_converts_without_xml_and_copies_custom_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Legacy OStim Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/LegacyPack/LegacyScene.json",
                    json.dumps(
                        {
                            "name": "Legacy Scene",
                            "modpack": "LegacyPack",
                            "length": 5,
                            "clips": [
                                {"animation": "LegacyEvent_0", "actorIndex": 0},
                                {"animation": "LegacyEvent_1", "actorIndex": 1},
                            ],
                            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                            "actions": [{"type": "legacycustom", "actor": 0, "target": 1}],
                            "navigations": [{"destination": "LegacyNext"}],
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/LegacyPack/LegacyNext.json",
                    json.dumps(
                        {
                            "name": "Legacy Next",
                            "modpack": "LegacyPack",
                            "length": 4,
                            "poses": [{"clips": [{"file": "meshes/actors/character/animations/Legacy/NextEvent_0.hkx"}]}],
                            "clips": [{"file": "meshes/actors/character/animations/Legacy/NextEvent_1.hkx"}],
                            "actors": [{"animationIndex": 0}, {"animationIndex": 1}],
                            "action": "kissing",
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/LegacyPack/LegacySilent.json",
                    json.dumps(
                        {
                            "name": "Legacy Silent",
                            "modpack": "LegacyPack",
                            "length": 3,
                            "clips": [{"animation": "SilentEvent_0", "actorIndex": 0}],
                            "actors": [{"animationIndex": 0}],
                        }
                    ),
                )
                source.writestr(
                    "Data/SKSE/Plugins/OStim/actions/legacycustom.json",
                    json.dumps({"info": "Legacy custom action"}),
                )
                source.writestr("Data/meshes/actors/character/animations/Legacy/LegacyEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/LegacyEvent_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/NextEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/NextEvent_1.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/SilentEvent_0.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)
            self.assertEqual(len(result.scenes), 3)
            self.assertEqual(len(result.ostim_action_files), 1)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Legacy_OStim_Pack/Legacy_OStim_Pack_LegacyScene.json"
                    ).decode("utf-8")
                )
                silent = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Legacy_OStim_Pack/Legacy_OStim_Pack_LegacySilent.json"
                    ).decode("utf-8")
                )
                copied_action = json.loads(
                    converted.read("Data/SKSE/Plugins/OStim/actions/legacycustom.json").decode("utf-8")
                )
            self.assertEqual([speed["animation"] for speed in scene["speeds"]], ["LegacyEvent"])
            self.assertEqual(scene["actions"][0]["type"], "legacycustom")
            self.assertEqual(silent["actions"][0]["type"], "ostimconvertermoan")
            self.assertNotIn("clips", scene)
            self.assertNotIn("poses", scene)
            self.assertIn("Data/SKSE/Plugins/OStim/actions/legacycustom.json", names)
            self.assertIn("Data/SKSE/Plugins/OStim/actions/ostimconvertermoan.json", names)
            self.assertTrue(copied_action["actor"]["moan"])
            self.assertTrue(copied_action["actor"]["talk"])
            self.assertTrue(copied_action["target"]["moan"])
            self.assertTrue(copied_action["target"]["talk"])
            self.assertIn("legacyostim", copied_action["tags"])

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertEqual(verification.report["missingAnimationEventCount"], 0)
            self.assertEqual(verification.report["sfxFallbackActions"], ["ostimconvertermoan"])

    def test_legacy_ostim_action_aliases_normalize_to_builtin_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Legacy Alias Pack.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/LegacyAlias/AliasScene.json",
                    json.dumps(
                        {
                            "name": "Alias Scene",
                            "modpack": "LegacyAlias",
                            "length": 3,
                            "speeds": [{"animation": "AliasEvent"}],
                            "actors": [{}, {}],
                            "actions": [{"type": "fingering", "actor": 0, "target": 1}],
                        }
                    ),
                )
                source.writestr("Data/meshes/actors/character/animations/LegacyAlias/AliasEvent_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/LegacyAlias/AliasEvent_1.hkx", b"")

            result = converter.convert_archive_to_ready_zip(archive)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                scene = json.loads(
                    converted.read(
                        "Data/SKSE/Plugins/OStim/scenes/Legacy_Alias_Pack/Legacy_Alias_Pack_AliasScene.json"
                    ).decode("utf-8")
                )

            self.assertEqual(scene["actions"][0]["type"], "vaginalfingering")
            self.assertNotIn("Data/SKSE/Plugins/OStim/actions/fingering.json", names)
            self.assertTrue(any("normalized OStim action 'fingering' to 'vaginalfingering'" in warning for warning in result.warnings))

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])
            self.assertNotIn("fingering", verification.report["customSceneActionTypes"])

    def test_verify_converted_zip_reports_deploy_blockers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "BadConverted.zip"
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr(
                    "Data/SKSE/Plugins/OStim/scenes/BadPack/BadScene.json",
                    json.dumps(
                        {
                            "name": "Bad Scene",
                            "modpack": "BadPack",
                            "length": 3,
                            "speeds": [{"animation": "BadScene"}],
                            "actors": [{}, {}],
                        }
                    ),
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("No HKX animation files" in error for error in verification.report["errors"]))
            self.assertTrue(any("No behavior animation list" in error for error in verification.report["errors"]))
            self.assertTrue(any("Behavior output mode is unsupported" in error for error in verification.report["errors"]))
            self.assertEqual(verification.report["missingAnimationEventCount"], 2)

    def test_verify_rejects_obsolete_pandora_registration_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good_zip = root / "GoodPandora.zip"
            write_pandora_ostim_zip(good_zip)

            verification = converter.verify_converted_zip(good_zip, write_report=False)

            self.assertFalse(verification.ok)
            self.assertTrue(any("obsolete Pandora info.xml" in error for error in verification.report["errors"]))
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertFalse(diagnostics["expectedPandoraGeneration"])
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertEqual(diagnostics["pandoraModuleCode"], "PandoraPack")
            self.assertEqual(diagnostics["namedAnimationDataFileCount"], 1)
            self.assertEqual(diagnostics["namedAnimationSetDataFileCount"], 2)
            self.assertEqual(diagnostics["animationDataEventRowCount"], 2)
            self.assertEqual(diagnostics["hkxRowsMatchingPackagedFiles"], 4)
            text_report = converter.verification_report_to_text(verification.report)
            self.assertIn("Pandora Module Diagnostics:", text_report)
            self.assertIn("Expected Pandora generation: no", text_report)

    def test_verify_rejects_pandora_checkbox_without_animationdata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "NoAnimData.zip"
            write_pandora_ostim_zip(bad_zip, animationdata_text=None)

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertFalse(diagnostics["expectedPandoraGeneration"])
            self.assertEqual(diagnostics["namedAnimationDataFileCount"], 0)
            self.assertTrue(any("checkbox-only" in error for error in verification.report["errors"]))

    def test_verify_rejects_pandora_checkbox_without_animationsetdata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "NoAnimSetData.zip"
            write_pandora_ostim_zip(bad_zip, animationsetdata_text=None)

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertEqual(diagnostics["namedAnimationSetDataFileCount"], 0)
            self.assertTrue(any("named animationsetdata is missing" in issue for issue in diagnostics["blockingIssues"]))

    def test_verify_rejects_empty_pandora_payload_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "EmptyPandoraPayload.zip"
            write_pandora_ostim_zip(bad_zip, animationdata_text="", animationsetdata_text="")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertEqual(diagnostics["nonEmptyAnimationDataFiles"], 0)
            self.assertEqual(diagnostics["nonEmptyAnimationSetDataFiles"], 0)
            self.assertTrue(any("empty" in issue for issue in diagnostics["blockingIssues"]))

    def test_verify_rejects_pandora_payload_that_does_not_match_scene_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "MismatchedPandoraEvents.zip"
            write_pandora_ostim_zip(
                bad_zip,
                animationdata_text="Other_0\nOther_1\n",
                animationsetdata_text=(
                    "meshes\\actors\\character\\animations\\PandoraPack\\Pose_0.hkx\n"
                    "meshes\\actors\\character\\animations\\PandoraPack\\Pose_1.hkx\n"
                ),
            )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertEqual(diagnostics["eventRowsMatchingSceneActorEvents"], 0)
            self.assertTrue(any("scene animation events do not match" in issue for issue in diagnostics["blockingIssues"]))

    def test_verify_rejects_pandora_payload_with_missing_hkx_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "MissingPandoraHkx.zip"
            write_pandora_ostim_zip(
                bad_zip,
                animationsetdata_text="meshes\\actors\\character\\animations\\PandoraPack\\Missing_0.hkx\n",
            )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)

            self.assertFalse(verification.ok)
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertEqual(diagnostics["missingHkxRowCount"], 2)
            self.assertTrue(any("missing HKX" in issue for issue in diagnostics["blockingIssues"]))

    def test_verify_reports_nested_pandora_module_layout_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested_zip = root / "NestedPandora.zip"
            write_pandora_ostim_zip(nested_zip, prefix="Wrapper/Data")

            verification = converter.verify_converted_zip(nested_zip, write_report=False)

            self.assertFalse(verification.ok)
            self.assertTrue(any("obsolete Pandora info.xml" in error for error in verification.report["errors"]))
            diagnostics = verification.report["pandoraModuleDiagnostics"]
            self.assertFalse(diagnostics["expectedPandoraGeneration"])
            self.assertTrue(diagnostics["checkboxOnlyRisk"])
            self.assertTrue(diagnostics["layoutWarnings"])
            self.assertTrue(any("Data folder is nested" in warning for warning in verification.report["warnings"]))

    def test_verify_rejects_ostim_tools_project_as_ostimsa_pandora_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_zip = root / "ToolsProject.zip"
            with ZipFile(project_zip, "w") as archive:
                archive.writestr(converter.OSTIM_TOOLS_PROJECT_FILE, json.dumps({"schema": converter.OSTIM_TOOLS_PROJECT_SCHEMA}))
                archive.writestr("scenes/Scene.json", json.dumps({"name": "Editable only"}))

            verification = converter.verify_converted_zip(project_zip, write_report=False)

            self.assertFalse(verification.ok)
            self.assertTrue(any("OStim Tools JSON project files were found" in error for error in verification.report["errors"]))

    def test_analyze_pandora_log_detects_expected_module_and_output(self):
        log_text = "\n".join(
            [
                "INFO: Pandora Mod Provider > PandoraPack > Load > info.xml > OK",
                "INFO: Pandora Assembler > PandoraPack > AnimData > Pose_0 > OK",
                "INFO: Dispatcher > PandoraPack > Export > meshes > OUTPUT GENERATED",
                "INFO: 2 total animations added",
            ]
        )

        report = converter.analyze_pandora_log_text(log_text, expected_modules=["PandoraPack"])

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["expectedModuleMentioned"])
        self.assertTrue(report["outputGenerated"])
        self.assertFalse(report["noAnimationsFound"])
        self.assertEqual(report["maxAnimationsAdded"], 2)
        text_report = converter.pandora_log_analysis_to_text(report)
        self.assertIn("Pandora Log Analysis", text_report)
        self.assertIn("Expected module mentioned: yes", text_report)

    def test_analyze_pandora_log_detects_current_merge_success_wording(self):
        log_text = "\n".join(
            [
                "INFO : Pandora Mod 1 : K4 1.5 SE - v.1.0.0",
                "INFO : Successfully merged OutputAnimSetData file",
                "INFO : Successfully merged OutputAnimData file",
            ]
        )

        report = converter.analyze_pandora_log_text(log_text, expected_modules=["K4 1.5 SE"])

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["expectedModuleMentioned"])
        self.assertTrue(report["expectedModuleLoadedEvidence"])
        self.assertTrue(report["outputGenerated"])

    def test_analyze_pandora_log_detects_auto_discovered_sexlab_fnis_list(self):
        log_text = "\n".join(
            [
                "INFO : Successfully merged OutputAnimSetData file",
                "INFO : Successfully merged OutputAnimData file",
                "INFO : FNIS Mod 9 : FNIS_Sanguine_Seduction_3_0_List",
            ]
        )

        report = converter.analyze_pandora_log_text(
            log_text,
            expected_fnis_lists=["meshes/actors/character/animations/Sanguine/FNIS_Sanguine_Seduction_3_0_List.txt"],
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["expectedFnisListMentioned"])
        self.assertFalse(report["expectedFnisListMissing"])
        self.assertEqual(report["discoveredFnisModNames"], ["FNIS_Sanguine_Seduction_3_0_List"])
        self.assertIn("No converted-pack checkbox is expected", report["recommendedUserAction"])

    def test_analyze_pandora_log_reports_missing_sexlab_fnis_list_without_requesting_checkbox(self):
        report = converter.analyze_pandora_log_text(
            "INFO : FNIS Mod 1 : FNIS_Other_List\nINFO : Successfully merged OutputAnimData file\n",
            expected_fnis_lists=["FNIS_Sanguine_Seduction_3_0_List.txt"],
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["expectedFnisListMissing"])
        self.assertIn("does not appear as a Pandora checkbox", report["recommendedUserAction"])

    def test_analyze_pandora_log_requires_every_expected_fnis_list(self):
        report = converter.analyze_pandora_log_text(
            "INFO : FNIS Mod 1 : FNIS_Human_List\nINFO : Successfully merged OutputAnimData file\n",
            expected_fnis_lists=["FNIS_Human_List.txt", "FNIS_Creature_List.txt"],
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["expectedFnisListMentioned"])
        self.assertEqual(report["missingExpectedFnisListNames"], ["FNIS_Creature_List"])
        self.assertIn("FNIS_Creature_List", report["recommendedUserAction"])

    def test_analyze_pandora_log_file_uses_observed_output_and_rejects_unrelated_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "Pandora Output"
            behavior = output / "meshes/actors/character/behaviors/0_master.hkx"
            behavior.parent.mkdir(parents=True, exist_ok=True)
            behavior.write_bytes(b"behavior")
            log_path = output / "Engine.log"
            log_path.write_text("INFO : Pandora Mod 1 : Test Pack - v.1.0.0\n", encoding="utf-8")
            settings_path = root / "Settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "SkyrimSE": {
                                "gameDataPath": str(root / "Live Data"),
                                "outputPath": str(root / "Live Data"),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = converter.pandora_user_settings_summary(settings_path=settings_path, log_path=log_path)
            report = converter.analyze_pandora_log_text(log_path.read_text(encoding="utf-8"), expected_modules=["Test Pack"])
            report["pandoraUserSettings"] = settings
            report["observedOutputArtifacts"] = converter.pandora_output_artifact_summary(log_path)
            report["outputGenerated"] = bool(report["observedOutputArtifacts"]["behaviorHkxCount"])

            self.assertFalse(settings["settingsApplicableToLog"])
            self.assertEqual(report["observedOutputArtifacts"]["behaviorHkxCount"], 1)
            self.assertNotIn("live Skyrim Data folder", converter.pandora_log_recommended_action(report))

    def test_analyze_pandora_log_flags_zero_animation_output(self):
        log_text = "\n".join(
            [
                "INFO: Pandora Mod Provider > PandoraPack > Load > info.xml > OK",
                "WARN: Pandora Assembler > PandoraPack > AnimData > DefaultMale > no animations found",
                "INFO: 0 animations added",
            ]
        )

        report = converter.analyze_pandora_log_text(log_text, expected_modules=["PandoraPack"])

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["expectedModuleMentioned"])
        self.assertTrue(report["noAnimationsFound"])
        self.assertTrue(report["expectedModuleSkipped"])
        self.assertIn("zero/no animations", report["recommendedUserAction"])

    def test_analyze_pandora_log_fails_when_expected_module_is_missing(self):
        log_text = "\n".join(
            [
                "INFO: Pandora Mod Provider > OtherPack > Load > info.xml > OK",
                "INFO: 4 total animations added",
            ]
        )

        report = converter.analyze_pandora_log_text(log_text, expected_modules=["PandoraPack"])

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["expectedModuleMissing"])
        self.assertIn("does not mention the expected converted module", report["recommendedUserAction"])

    def test_report_bundle_includes_pandora_log_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle_path = root / "Bundle.zip"
            report = converter.analyze_pandora_log_text(
                "INFO: Pandora Mod Provider > PandoraPack > Load > info.xml > OK\nINFO: 1 animations added\n",
                expected_modules=["PandoraPack"],
            )

            created = converter.create_nexus_safe_report_bundle(bundle_path, pandora_log_analysis=report)

            with _ZipFile(created, "r") as archive:
                names = set(archive.namelist())
                self.assertIn("pandora_log_analysis_public.json", names)
                self.assertIn("pandora_log_analysis_public.txt", names)
                bundled = json.loads(archive.read("pandora_log_analysis_public.json").decode("utf-8"))
            self.assertEqual(bundled["type"], "pandoraLogAnalysis")
            self.assertTrue(bundled["expectedModuleMentioned"])

    def test_verify_rejects_stale_converter_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "StaleMetadata.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "BadMeta",
                "length": 3,
                "speeds": [{"animation": "FakeEvent"}],
                "actors": [{}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            metadata = {
                "schema": converter.CONVERTER_METADATA_SCHEMA,
                "schemaVersion": converter.CONVERTER_METADATA_SCHEMA_VERSION,
                "pack": {"name": "BadMeta", "folder": "BadMeta", "behaviorCode": "BadMeta", "author": "Tester"},
                "deployment": {
                    "status": "PASS",
                    "sceneCount": 99,
                    "startableSceneCount": 1,
                    "categorizedStartableSceneCount": 1,
                    "hkxCount": 1,
                    "registeredAnimationEventCount": 1,
                    "missingAnimationEventCount": 0,
                    "missingSceneLinkCount": 0,
                },
                "checks": {
                    "hasScenes": True,
                    "hasStartableScenes": True,
                    "hasCategorizedStartableScenes": True,
                    "hasHkxAnimations": True,
                    "hasBehaviorList": True,
                    "hasPandoraFiles": True,
                    "hasMissingAnimationEvents": False,
                    "hasMissingSceneLinks": False,
                    "usesConverterSfxFallback": False,
                },
                "paths": {
                    "scenesFolder": "Data/SKSE/Plugins/OStim/scenes/BadMeta/",
                    "animationFolder": "Data/meshes/actors/character/animations/BadMeta/",
                    "behaviorList": "Data/meshes/actors/character/animations/BadMeta/Missing_List.txt",
                    "pandoraInfo": "Data/Pandora_Engine/mod/BadMeta/info.xml",
                    "pandoraAnimSetRoot": "Data/animationsetdatasinglefile/",
                    "pandoraNamedAnimSetRoot": "Data/Pandora_Engine/mod/BadMeta/animationsetdata/",
                    "converterMetadata": "Data/SKSE/Plugins/OStim/converter_metadata/BadMeta/metadata.json",
                    "actionFiles": [],
                },
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/BadMeta/BadMeta_Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/BadMeta/FakeEvent_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/BadMeta/FNIS_BadMeta_List.txt",
                    "b -Tn FakeEvent_0 FakeEvent_0.hkx\n",
                )
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", "meshes\\actors\\character\\animations\\BadMeta\\FakeEvent_0.hkx\n")
                archive.writestr(
                    "Data/Pandora_Engine/mod/BadMeta/info.xml",
                    "<mod><name>BadMeta</name><author>Tester</author></mod>",
                )
                archive.writestr(
                    "Data/Pandora_Engine/mod/BadMeta/animationsetdata/DefaultMale.txt",
                    "meshes\\actors\\character\\animations\\BadMeta\\FakeEvent_0.hkx\n",
                )
                archive.writestr(
                    "Data/SKSE/Plugins/OStim/converter_metadata/BadMeta/metadata.json",
                    json.dumps(metadata),
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["converterMetadataCount"], 1)
            self.assertTrue(any("metadata deployment.sceneCount is 99" in error for error in verification.report["errors"]))
            self.assertTrue(any("metadata path behaviorList does not exist" in error for error in verification.report["errors"]))

    def test_verify_rejects_fake_nemesis_checkbox_without_behavior_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "FakeNemesisCheckbox.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "FakeNemesis",
                "length": 3,
                "speeds": [{"animation": "FakeEvent"}],
                "actors": [{"animationIndex": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/FakeNemesis/FakeScene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/FakeNemesis/FakeEvent_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/FakeNemesis/FNIS_FakeNemesis_List.txt",
                    "b -Tn FakeEvent_0 FakeEvent_0.hkx\n",
                )
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", "meshes\\actors\\character\\animations\\FakeNemesis\\FakeEvent_0.hkx\n")
                archive.writestr("Data/Nemesis_Engine/mod/FakeNemesis/info.ini", "name=FakeNemesis\nauthor=Nobody\nsite=null\nhidden=false\n")
                archive.writestr(
                    "Data/Nemesis_Engine/mod/FakeNemesis/animationdata/DefaultMale.txt",
                    "meshes\\actors\\character\\animations\\FakeNemesis\\FakeEvent_0.hkx\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertTrue(any("Nemesis ERROR(2006)" in error and "File: animationdata" in error for error in verification.report["errors"]))
            self.assertTrue(any("contains a Nemesis checkbox but no real Nemesis behavior patch files" in error for error in verification.report["errors"]))

    def test_verify_rejects_searchable_ostim_scene_without_att_nemesis_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "SearchableButIdle.zip"
            scene_data = {
                "name": "Searchable Scene",
                "modpack": "SearchableButIdle",
                "length": 3,
                "speeds": [{"animation": "IdleRisk"}],
                "actors": [{}, {}],
                "actions": [{"type": "kissing", "actor": 0, "target": 1}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/SearchableButIdle/Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/SearchableButIdle/IdleRisk_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/SearchableButIdle/IdleRisk_1.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/SearchableButIdle/ATT_searchablebutidle_animlist.txt",
                    "b -Tn IdleRisk_0 IdleRisk_0.hkx SEARCHABLEBUTIDLE_AnimationSpeed: 1\n"
                    "b -Tn IdleRisk_1 IdleRisk_1.hkx SEARCHABLEBUTIDLE_AnimationSpeed: 1\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["behaviorOutputMode"], converter.OSTIMSA_BEHAVIOR_MODE_UNSUPPORTED)
            self.assertTrue(any("ATT animation list output is present" in error for error in verification.report["errors"]))

    def test_verify_rejects_behavior_registration_that_points_to_missing_hkx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "MissingRegisteredHkx.zip"
            scene_data = {
                "name": "Missing HKX Scene",
                "modpack": "MissingRegisteredHkx",
                "length": 3,
                "speeds": [{"animation": "MissingRegistered"}],
                "actors": [{"animationIndex": 0}],
                "actions": [{"type": "kissing", "actor": 0}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/MissingRegisteredHkx/Scene.json", json.dumps(scene_data))
                archive.writestr(
                    "Data/meshes/actors/character/animations/MissingRegisteredHkx/FNIS_MissingRegisteredHkx_List.txt",
                    "b -Tn MissingRegistered_0 MissingRegistered_0.hkx\n",
                )
                archive.writestr("Data/animdata/MissingRegisteredHkx_DefaultMale.txt", "MissingRegistered_0\n")
                archive.writestr(
                    "Data/animationsetdatasinglefile/MissingRegisteredHkx_DefaultMale.txt",
                    "meshes\\actors\\character\\animations\\MissingRegisteredHkx\\MissingRegistered_0.hkx\n",
                )
                archive.writestr(
                    "Data/Pandora_Engine/mod/MissingRegisteredHkx/info.xml",
                    "<mod><name>MissingRegisteredHkx</name><author>Tester</author></mod>",
                )
                archive.writestr("Data/Pandora_Engine/mod/MissingRegisteredHkx/animationdata/DefaultMale.txt", "MissingRegistered_0\n")
                archive.writestr(
                    "Data/Pandora_Engine/mod/MissingRegisteredHkx/animationsetdata/DefaultMale.txt",
                    "meshes\\actors\\character\\animations\\MissingRegisteredHkx\\MissingRegistered_0.hkx\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/missingregistered/info.ini",
                    "name=MissingRegisteredHkx\nauthor=Tester\nsite=null\nauto=null\nhidden=true\n",
                )
                archive.writestr(
                    "Data/Nemesis_Engine/mod/missingregistered/0_master/#0106.txt",
                    "<hkobject><hkcstring>MissingRegistered_0</hkcstring></hkobject>\n",
                )

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertGreaterEqual(verification.report["behaviorEventsPointingToMissingHkxCount"], 1)
            self.assertTrue(any("points to missing HKX file" in error for error in verification.report["errors"]))

    def test_source_nemesis_zero_master_patch_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "Source Nemesis Patch.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(
                    "Data/Meshes/0SA/mod/0Sex/scene/AA/Pos/HJ/Smoke.xml",
                    """\
<scene id="AA|Pos|HJ|Smoke" actors="2">
  <info name="Smoke" />
  <anim id="AA_HJ_Smoke" l="3" />
</scene>
""",
                )
                source.writestr("Data/meshes/actors/character/animations/Legacy/AA_HJ_Smoke_0.hkx", b"")
                source.writestr("Data/meshes/actors/character/animations/Legacy/AA_HJ_Smoke_1.hkx", b"")
                source.writestr(
                    "Data/Nemesis_Engine/mod/sourcepatch/info.ini",
                    "name=Source Patch\nauthor=Tester\nsite=null\nhidden=false\n",
                )
                source.writestr(
                    "Data/Nemesis_Engine/mod/sourcepatch/0_master/#0106.txt",
                    "<hkobject><hkcstring>AA_HJ_Smoke_0</hkcstring><hkcstring>AA_HJ_Smoke_1</hkcstring></hkobject>\n",
                )

            result = converter.convert_archive_to_ready_zip(archive)

            with ZipFile(result.zip_path) as converted:
                names = set(converted.namelist())
                report = json.loads(converted.read("conversion_report.json").decode("utf-8"))
                self.assertIn("Data/Nemesis_Engine/mod/sourcepatch/info.ini", names)
                self.assertIn("Data/Nemesis_Engine/mod/sourcepatch/0_master/#0106.txt", names)
                behavior_events = read_pandora_compatible_events(converted, names, "Source_Nemesis_Patch")
            self.assertIn("AA_HJ_Smoke_0", behavior_events)
            self.assertGreaterEqual(report["sourceEnginePatchFileCount"], 2)

            verification = converter.verify_converted_zip(result.zip_path, write_report=False)
            self.assertTrue(verification.ok, verification.report["errors"])

    def test_verify_rejects_behavior_hkx_packaged_as_animation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "BehaviorGraphAsAnimation.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "BehaviorGraphAsAnimation",
                "length": 3,
                "speeds": [{"animation": "PlayableEvent"}],
                "actors": [{}, {}],
                "actions": [{"type": "handjob", "actor": 0, "target": 1}],
            }
            animation_list = "\n".join(
                [
                    "b -Tn PlayableEvent_0 PlayableEvent_0.hkx",
                    "b -Tn PlayableEvent_1 PlayableEvent_1.hkx",
                    "b -Tn FNIS_Bad_Behavior FNIS_Bad_Behavior.hkx",
                    "",
                ]
            )
            animset = "\n".join(
                [
                    "meshes\\actors\\character\\animations\\Bad\\PlayableEvent_0.hkx",
                    "meshes\\actors\\character\\animations\\Bad\\PlayableEvent_1.hkx",
                    "meshes\\actors\\character\\animations\\Bad\\FNIS_Bad_Behavior.hkx",
                    "",
                ]
            )
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/Bad/BadScene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/Bad/PlayableEvent_0.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/Bad/PlayableEvent_1.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/Bad/FNIS_Bad_Behavior.hkx", b"")
                archive.writestr("Data/meshes/actors/character/animations/Bad/FNIS_Bad_List.txt", animation_list)
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", animset)
                archive.writestr(
                    "Data/Pandora_Engine/mod/Bad/info.xml",
                    "<mod><name>Bad</name><author>Tester</author></mod>",
                )
                archive.writestr("Data/Pandora_Engine/mod/Bad/animationsetdata/DefaultMale.txt", animset)

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["behaviorGraphHkxCount"], 0)
            self.assertTrue(any("legacy behavior graph HKX" in error for error in verification.report["errors"]))

    def test_verify_rejects_old_unresolved_relative_scene_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "UnresolvedRelative.zip"
            scene_data = {
                "name": "Scene",
                "modpack": "OldPack",
                "length": 3,
                "speeds": [{"animation": "FakeEvent"}],
                "actors": [{"animationIndex": 0}],
                "navigations": [{"destination": "OldPack_^+10A"}],
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/OldPack/Scene.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/OldPack/FakeEvent_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/OldPack/FNIS_OldPack_List.txt",
                    "b -Tn FakeEvent_0 FakeEvent_0.hkx\n",
                )
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", "meshes\\actors\\character\\animations\\OldPack\\FakeEvent_0.hkx\n")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["unresolvedRelativeSceneLinkCount"], 1)
            self.assertTrue(any("unresolved OSex relative scene reference" in error for error in verification.report["errors"]))

    def test_verify_rejects_zip_with_no_startable_scenes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_zip = root / "OnlyTransitions.zip"
            scene_data = {
                "name": "Transition",
                "modpack": "OnlyTransitions",
                "length": 3,
                "speeds": [],
                "actors": [{"animationIndex": 0}],
                "destination": "OnlyTransitions_Target",
            }
            with ZipFile(bad_zip, "w") as archive:
                archive.writestr("Data/SKSE/Plugins/OStim/scenes/OnlyTransitions/Transition.json", json.dumps(scene_data))
                archive.writestr("Data/meshes/actors/character/animations/OnlyTransitions/Dummy_0.hkx", b"")
                archive.writestr(
                    "Data/meshes/actors/character/animations/OnlyTransitions/FNIS_OnlyTransitions_List.txt",
                    "b -Tn Dummy_0 Dummy_0.hkx\n",
                )
                archive.writestr("Data/animationsetdatasinglefile/DefaultMale.txt", "meshes\\actors\\character\\animations\\OnlyTransitions\\Dummy_0.hkx\n")

            verification = converter.verify_converted_zip(bad_zip, write_report=False)
            self.assertFalse(verification.ok)
            self.assertEqual(verification.report["startableSceneCount"], 0)
            self.assertEqual(verification.report["deployableSceneCount"], 0)
            self.assertTrue(any("No deployable OStim scenes" in error for error in verification.report["errors"]))


if __name__ == "__main__":
    unittest.main()
