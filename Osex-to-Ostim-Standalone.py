#!/usr/bin/env python3
r"""
osex_to_ostim_sa_converter.py

GUI tool to convert OSex/OpenSex (OSA-based) XML scenes into OStim Standalone (SA)
JSON scenes, auto-generate Pandora inputs, and package a ready-to-install ZIP.

• Supports importing an OSex animation mod archive: .zip natively, .7z via 7z/7za/7zz in PATH
• Extracts to a temp folder, discovers XML/HKX, converts, and can build an install-ready ZIP
• Python 3.10+ · only stdlib (tkinter required for GUI)
"""
from __future__ import annotations

import json
import re
import sys
import shutil
import subprocess
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from zipfile import ZipFile, ZIP_DEFLATED

# -----------------------------
# GUI (stdlib)
# -----------------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    TAGMAP: Dict[str, str] = field(default_factory=lambda: {
        "scene": "Scene",
        "stage": "Stage",
        "anim":  "Animation",
        "sequence": "Sequence",
        "event": "Event",
    })

    ATTR: Dict[str, str] = field(default_factory=lambda: {
        # Scene
        "scene_id": "id",
        "scene_name": "name",
        "actors": "actors",
        # Stage
        "stage_id": "id",
        "stage_name": "name",
        "duration": "duration",
        # Animation
        "anim_id": "id",
        "anim_actor_index": "actorIndex",
        "anim_path": "file",
        # Optional offsets
        "pos_x": "x", "pos_y": "y", "pos_z": "z",
        "rot_x": "rx", "rot_y": "ry", "rot_z": "rz",
    })

    DEFAULT_ROLES: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    DEFAULT_STAGE_DURATION: float = 6.0


CONFIG = Config()


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Clip:
    actor_index: int
    hkx_path: str
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class Stage:
    stage_id: str
    name: str
    duration: float
    clips: List[Clip] = field(default_factory=list)

@dataclass
class Scene:
    scene_id: str
    name: str
    actor_count: int
    stages: List[Stage] = field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------
def norm_slashes(s: str) -> str:
    r"""Normalize path separators to forward slashes.
    Collapses any mix of '\' and '/' to a single '/'."""
    return re.sub(r"[\\/]+", "/", s)


# -----------------------------
# XML Parsing (tolerant)
# -----------------------------
def _get_attr(elem: ET.Element, key: str, default: Optional[str] = None) -> Optional[str]:
    attr_name = CONFIG.ATTR.get(key, key)
    return elem.attrib.get(attr_name, default)

def parse_osex_xml(xml_path: Path) -> List[Scene]:
    """Parse OSex/OpenSex XML into Scene objects with tolerant heuristics."""
    def tagset(*names: str) -> set[str]:
        return {n.lower() for n in names}

    SCENE_TAGS = tagset("Scene", "SceneData", "OSexScene", "OSex", "SexScene", "scene")
    STAGE_TAGS = tagset("Stage", "Pose", "Position", "stage", "pose")
    ANIM_TAGS  = tagset("Animation", "Anim", "Clip", "animation", "anim", "clip")

    # Alternative attribute names sometimes found
    ACTOR_ATTRS = ("actorIndex", "actor", "index", "slot")
    FILE_ATTRS = ("file", "path", "hkx", "animation", "anim")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def lname(e: ET.Element) -> str:
        return e.tag.split("}")[-1].lower()  # strip XMLNS if present

    def get_first_attr(e: ET.Element, names: tuple[str, ...]) -> Optional[str]:
        for n in names:
            if n in e.attrib:
                return e.attrib.get(n)
        return None

    scenes: List[Scene] = []

    # Strategy A: explicit <Scene> containers
    scene_els = [e for e in root.iter() if lname(e) in SCENE_TAGS]

    def collect_stage(stage_el: ET.Element, scene_id: str, scene_obj: Scene):
        sid = _get_attr(stage_el, "stage_id") or stage_el.attrib.get("id") or f"{scene_id}_stage{len(scene_obj.stages)+1}"
        sname = _get_attr(stage_el, "stage_name") or stage_el.attrib.get("name") or sid
        dur_s = _get_attr(stage_el, "duration") or stage_el.attrib.get("duration")
        try:
            duration = float(dur_s) if dur_s else CONFIG.DEFAULT_STAGE_DURATION
        except ValueError:
            duration = CONFIG.DEFAULT_STAGE_DURATION
        stage = Stage(stage_id=sid, name=sname, duration=duration)

        for anim_el in stage_el.iter():
            if lname(anim_el) not in ANIM_TAGS:
                continue
            actor_idx_s = _get_attr(anim_el, "anim_actor_index") or get_first_attr(anim_el, ACTOR_ATTRS) or "0"
            try:
                actor_index = int(actor_idx_s)
            except ValueError:
                actor_index = 0
            hkx = _get_attr(anim_el, "anim_path") or get_first_attr(anim_el, FILE_ATTRS) or ""
            if not hkx:
                continue
            def fget(tag_key: str) -> float:
                v = _get_attr(anim_el, tag_key)
                try:
                    return float(v) if v is not None else 0.0
                except ValueError:
                    return 0.0
            pos = (fget("pos_x"), fget("pos_y"), fget("pos_z"))
            rot = (fget("rot_x"), fget("rot_y"), fget("rot_z"))
            stage.clips.append(Clip(actor_index=actor_index, hkx_path=hkx, pos=pos, rot=rot))
        if stage.clips:
            scene_obj.stages.append(stage)

    if scene_els:
        for se in scene_els:
            scene_id = _get_attr(se, "scene_id") or se.attrib.get("id") or xml_path.stem
            scene_name = _get_attr(se, "scene_name") or se.attrib.get("name") or scene_id
            actor_count_attr = _get_attr(se, "actors") or se.attrib.get("actors")
            try:
                actor_count = int(actor_count_attr) if actor_count_attr else 2
            except ValueError:
                actor_count = 2
            scene = Scene(scene_id=scene_id, name=scene_name, actor_count=actor_count)

            stage_els = [e for e in se.iter() if lname(e) in STAGE_TAGS]
            if not stage_els:
                pseudo = ET.Element("Stage")
                for e in se.iter():
                    if lname(e) in ANIM_TAGS:
                        pseudo.append(e)
                if list(pseudo):
                    collect_stage(pseudo, scene.scene_id, scene)
            else:
                for st in stage_els:
                    collect_stage(st, scene.scene_id, scene)

            if scene.stages:
                if not actor_count_attr:
                    max_idx = 0
                    for st in scene.stages:
                        for c in st.clips:
                            if c.actor_index > max_idx:
                                max_idx = c.actor_index
                    scene.actor_count = max(scene.actor_count, max_idx + 1)
                scenes.append(scene)

    # Strategy B: no <Scene> wrapper — treat file as one scene with stages or just anims
    if not scenes:
        scene = Scene(scene_id=xml_path.stem, name=xml_path.stem, actor_count=2)
        stage_els = [e for e in root.iter() if lname(e) in STAGE_TAGS]
        if stage_els:
            for st in stage_els:
                collect_stage(st, scene.scene_id, scene)
        else:
            pseudo = ET.Element("Stage")
            for e in root.iter():
                if lname(e) in ANIM_TAGS:
                    pseudo.append(e)
            if list(pseudo):
                collect_stage(pseudo, scene.scene_id, scene)
        if scene.stages:
            max_idx = 0
            for st in scene.stages:
                for c in st.clips:
                    if c.actor_index > max_idx:
                        max_idx = c.actor_index
            scene.actor_count = max(2, max_idx + 1)
            scenes.append(scene)

    # Strategy C: last-resort scan for any attribute that ends with .hkx
    if not scenes:
        root_scene = Scene(scene_id=xml_path.stem, name=xml_path.stem, actor_count=2)
        stage = Stage(stage_id=root_scene.scene_id + "_auto", name="Auto", duration=CONFIG.DEFAULT_STAGE_DURATION)
        for el in root.iter():
            for _, v in el.attrib.items():
                if isinstance(v, str) and v.lower().endswith(".hkx"):
                    actor_idx_s = el.attrib.get("actor") or el.attrib.get("index") or el.attrib.get("actorIndex") or "0"
                    try:
                        actor_index = int(actor_idx_s)
                    except ValueError:
                        actor_index = 0
                    stage.clips.append(Clip(actor_index=actor_index, hkx_path=v))
                    break
        if stage.clips:
            max_idx = 0
            for c in stage.clips:
                if c.actor_index > max_idx:
                    max_idx = c.actor_index
            root_scene.actor_count = max(2, max_idx + 1)
            root_scene.stages.append(stage)
            scenes.append(root_scene)

    return scenes


# -----------------------------
# OStim SA JSON emit
# -----------------------------
def _role_name(i: int) -> str:
    try:
        return CONFIG.DEFAULT_ROLES[i]
    except IndexError:
        return f"Actor{i+1}"

def scene_to_ostim_json(scene: Scene, pack: str) -> Dict[str, Any]:
    actors = [{"name": _role_name(i), "index": i} for i in range(scene.actor_count)]
    poses: List[Dict[str, Any]] = []
    for st in scene.stages:
        pose = {
            "id": st.stage_id,
            "name": st.name,
            "duration": st.duration,
            "clips": [{"actor": c.actor_index, "file": norm_slashes(c.hkx_path)} for c in st.clips],
        }
        poses.append(pose)
    return {"id": scene.scene_id, "name": scene.name, "pack": pack, "actors": actors, "poses": poses}

def write_scene_json(scene: Scene, out_root: Path, pack: str) -> Path:
    data = scene_to_ostim_json(scene, pack)
    out_dir = out_root / pack
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene.scene_id}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# -----------------------------
# Alignment JSON support
# -----------------------------
def accumulate_alignment(scenes: List[Scene]) -> Dict[str, Any]:
    alignment: Dict[str, Any] = {"animations": {}}
    for sc in scenes:
        for st in sc.stages:
            for c in st.clips:
                key = f"{sc.scene_id}/{st.stage_id}/actor{c.actor_index}"
                alignment["animations"][key] = {
                    "pos": {"x": c.pos[0], "y": c.pos[1], "z": c.pos[2]},
                    "rot": {"x": c.rot[0], "y": c.rot[1], "z": c.rot[2]},
                }
    return alignment


# -----------------------------
# Pandora auto-generation
# -----------------------------
def collect_hkx_paths(scenes: List[Scene]) -> List[str]:
    seen = set()
    out: List[str] = []
    for sc in scenes:
        for st in sc.stages:
            for c in st.clips:
                p = norm_slashes(c.hkx_path)
                p = re.sub(r"^[\\/]+", "", p)  # strip leading separators
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out

def infer_project_from_path(hkx_rel: str) -> str:
    low = hkx_rel.lower()
    if "_1stperson" in low or "1stperson" in low:
        return "_1stperson"
    if "/horse" in low:
        return "horseproject"
    if "/draugr" in low:
        return "draugrproject"
    return "DefaultMale"

def generate_pandora_files(pandora_root: Path, hkx_paths: List[str]) -> List[Path]:
    """Write AnimSetData path lists for Pandora based on HKX paths.
    Uses print(file=...) so there are no literal newline strings to get split.
    """
    created: List[Path] = []
    groups: Dict[str, List[str]] = {}
    for rel in hkx_paths:
        proj = infer_project_from_path(rel)
        groups.setdefault(proj, []).append(norm_slashes(rel))

    for project, rels in groups.items():
        base = pandora_root / "animationsetdatasinglefile" / project
        base.mkdir(parents=True, exist_ok=True)

        # Project-wide list
        all_file = base / (project + ".txt")
        with all_file.open("w", encoding="utf-8") as f:
            for r in rels:
                path_norm = r if r.lower().startswith("meshes/") else ("meshes/" + r)
                print(norm_slashes(path_norm), file=f)
        created.append(all_file)

        # Common paired set
        set_file = base / "H2HDual.txt"
        with set_file.open("w", encoding="utf-8") as f:
            for r in rels:
                path_norm = r if r.lower().startswith("meshes/") else ("meshes/" + r)
                print(norm_slashes(path_norm), file=f)
        created.append(set_file)

    return created


# -----------------------------
# Packaging (.zip)
# -----------------------------
def make_ready_to_install_zip(
    zip_path: Path,
    generated_scenes_root: Path,
    pack: str,
    alignment_path: Optional[Path] = None,
    hkx_paths_for_auto: Optional[List[str]] = None,
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    data_prefix = Path("Data")
    ostim_scenes_dst = data_prefix / "SKSE/Plugins/OStim/scenes" / pack

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        pack_dir = generated_scenes_root / pack
        if not pack_dir.exists():
            raise FileNotFoundError(f"Expected scenes at {pack_dir}")
        for p in pack_dir.rglob("*.json"):
            rel_in_pack = p.relative_to(pack_dir)
            arc = ostim_scenes_dst / rel_in_pack
            zf.write(p, arcname=str(arc))

        # Auto Pandora
        if hkx_paths_for_auto:
            tmp = Path(tempfile.mkdtemp(prefix="pandora_auto_"))
            auto_root = tmp / "Pandora" / pack
            auto_root.mkdir(parents=True, exist_ok=True)
            _ = generate_pandora_files(auto_root, hkx_paths_for_auto)
            for f in auto_root.rglob("*"):
                if f.is_file():
                    arc = data_prefix / f.relative_to(tmp)
                    zf.write(f, arcname=str(arc))

        readme = textwrap.dedent(f"""
        OpenSex → OStim SA conversion (Pack: {pack})
        -------------------------------------------------
        Contents:
        - Data/SKSE/Plugins/OStim/scenes/{pack}/…  (converted scene JSON)
        - Data/Pandora/{pack}/…                     (auto-generated Pandora inputs)
        - alignment.json                            (copy to Documents path if desired)
        """)
        zf.writestr("README_OStim_SA.txt", readme.encode("utf-8"))

        if alignment_path and alignment_path.exists():
            zf.write(alignment_path, arcname="alignment.json")

    return zip_path


# -----------------------------
# Mod archive import (OSex animation mod)
# -----------------------------
def has_7z() -> bool:
    """True if 7z/7za/7zz is in PATH (for .7z extraction)."""
    for exe in ("7z", "7za", "7zz"):
        if shutil.which(exe):
            return True
    return False

def extract_mod_archive(archive: Path) -> Path:
    """Extract .zip natively or .7z via external 7z to a temp dir. Returns root."""
    tmp = Path(tempfile.mkdtemp(prefix="osex_mod_"))
    if archive.suffix.lower() == ".zip":
        with ZipFile(archive, "r") as zf:
            zf.extractall(tmp)
    elif archive.suffix.lower() == ".7z":
        if not has_7z():
            raise RuntimeError(".7z provided but 7z.exe/7za/7zz not found in PATH.")
        exe = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
        cmd = [exe, "x", str(archive), "-o" + str(tmp), "-y"]
        subprocess.check_call(cmd)
    else:
        raise RuntimeError("Unsupported archive type. Use .zip (recommended) or .7z (with 7z).")
    return tmp

def find_xml_root(extracted_root: Path) -> Path:
    """Find a folder under extracted_root that contains OSex/OpenSex XML files."""
    xmls = list(extracted_root.rglob("*.xml"))
    if not xmls:
        raise FileNotFoundError("No XML files found inside the mod archive.")
    parents = sorted({x.parent for x in xmls}, key=lambda p: len(p.parts))
    return parents[0]


# -----------------------------
# Conversion (used by GUI)
# -----------------------------
def discover_xml_files(input_root: Path) -> List[Path]:
    exts = {".xml", ".XML"}
    return [p for p in input_root.rglob("*") if p.suffix in exts]

def run_conversion(
    input_xml: Path,
    output_scenes: Path,
    pack: str,
    emit_alignment: Optional[Path] = None,
    merge_alignment_path: Optional[Path] = None,
    harvest_search_roots: Optional[List[Path]] = None,
) -> tuple[List[Scene], Optional[Path]]:
    xml_files = discover_xml_files(input_xml)
    if not xml_files:
        print(f"[warn] No XML files found under {input_xml}")
        xml_files = []

    all_scenes: List[Scene] = []
    for xml in xml_files:
        try:
            scenes = parse_osex_xml(xml)
        except ET.ParseError as e:
            print(f"[error] XML parse failed for {xml}: {e}")
            continue
        if not scenes:
            print(f"[warn] No scenes parsed from {xml}")
        all_scenes.extend(scenes)

    # If nothing parsed, brute HKX harvest
    if not all_scenes:
        def harvest_hkx(roots: List[Path]) -> List[Scene]:
            scenes: List[Scene] = []
            seen = set()
            for root_dir in roots:
                if not root_dir or not root_dir.exists():
                    continue
                for hkx in root_dir.rglob("*.hkx"):
                    try:
                        parts = list(hkx.parts)
                        if "meshes" in (p.lower() for p in parts):
                            idx = [i for i, p in enumerate(parts) if p.lower() == "meshes"][0]
                            rel = Path(*parts[idx:])
                        else:
                            rel = hkx.relative_to(root_dir)
                    except Exception:
                        rel = hkx.name
                    rel_s = norm_slashes(str(rel))
                    if rel_s in seen:
                        continue
                    seen.add(rel_s)
                    sid = hkx.stem
                    stage_id = sid + "_stage1"
                    sc = Scene(scene_id=sid, name=sid, actor_count=1,
                               stages=[Stage(stage_id=stage_id, name=sid, duration=CONFIG.DEFAULT_STAGE_DURATION,
                                             clips=[Clip(actor_index=0, hkx_path=rel_s)])])
                    scenes.append(sc)
            return scenes

        roots = harvest_search_roots or [input_xml]
        print("[info] Falling back to brute HKX harvest…")
        all_scenes = harvest_hkx(roots)

    if not all_scenes:
        raise RuntimeError("No scenes discovered; check your CONFIG and XML tag mapping.")

    for sc in all_scenes:
        out = write_scene_json(sc, output_scenes, pack)
        print(f"[ok] Wrote scene: {out}")

    alignment = accumulate_alignment(all_scenes)
    alignment_written: Optional[Path] = None

    if emit_alignment:
        emit_alignment.parent.mkdir(parents=True, exist_ok=True)
        emit_alignment.write_text(json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] Wrote alignment: {emit_alignment}")
        alignment_written = emit_alignment

    if merge_alignment_path:
        base = {"animations": {}}
        if merge_alignment_path.exists():
            try:
                base = json.loads(merge_alignment_path.read_text(encoding="utf-8"))
            except Exception:
                print(f"[warn] Could not parse existing alignment at {merge_alignment_path}; starting fresh.")
        base.setdefault("animations", {})
        base["animations"].update(alignment.get("animations", {}))
        merge_alignment_path.parent.mkdir(parents=True, exist_ok=True)
        merge_alignment_path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] Merged alignment into: {merge_alignment_path} ({len(base.get('animations', {}))} entries)")

    return all_scenes, alignment_written


# -----------------------------
# GUI
# -----------------------------
def launch_gui() -> int:
    if not TK_AVAILABLE:
        print("tkinter is not available in this Python environment.")
        return 2

    root = tk.Tk()
    root.title("OSex → OStim SA Converter")

    state = {
        "mod_archive": tk.StringVar(),
        "input_xml": tk.StringVar(),
        "output_scenes": tk.StringVar(),
        "pack": tk.StringVar(value="MyPack"),
        "emit_alignment": tk.StringVar(),
        "merge_alignment": tk.StringVar(),
        "zip_out": tk.StringVar(),
    }

    pad = {"padx": 8, "pady": 4}

    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    def browse_dir(var: tk.StringVar):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def browse_file_open_json(var: tk.StringVar):
        p = filedialog.askopenfilename(filetypes=(("JSON", "*.json"), ("All", "*.*")))
        if p:
            var.set(p)

    def browse_file_save_json(var: tk.StringVar):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON", "*.json"), ("All", "*.*")))
        if p:
            var.set(p)

    def browse_save_zip(var: tk.StringVar):
        p = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=(("ZIP file", "*.zip"),))
        if p:
            var.set(p)

    def browse_mod_archive():
        p = filedialog.askopenfilename(filetypes=(("Mod archives", "*.zip;*.7z"), ("ZIP", "*.zip"), ("7z", "*.7z"), ("All", "*.*")))
        if p:
            state["mod_archive"].set(p)

    r = 0
    ttk.Label(frm, text="OSex mod archive (.zip/.7z) [optional]:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["mod_archive"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=browse_mod_archive).grid(row=r, column=2, **pad)

    r += 1
    ttk.Label(frm, text="OR Input XML folder:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["input_xml"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=lambda: browse_dir(state["input_xml"])).grid(row=r, column=2, **pad)

    r += 1
    ttk.Label(frm, text="Output scenes root (…/OStim/scenes):").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["output_scenes"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=lambda: browse_dir(state["output_scenes"])).grid(row=r, column=2, **pad)

    r += 1
    ttk.Label(frm, text="Pack name:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["pack"], width=32).grid(row=r, column=1, sticky="w", **pad)

    r += 1
    ttk.Label(frm, text="Emit alignment.json to:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["emit_alignment"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=lambda: browse_file_save_json(state["emit_alignment"])).grid(row=r, column=2, **pad)

    r += 1
    ttk.Label(frm, text="Merge into existing alignment.json (Documents path):").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["merge_alignment"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=lambda: browse_file_open_json(state["merge_alignment"])).grid(row=r, column=2, **pad)

    r += 1
    ttk.Label(frm, text="Build ready-to-install ZIP here:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=state["zip_out"], width=56).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse", command=lambda: browse_save_zip(state["zip_out"])).grid(row=r, column=2, **pad)

    r += 1
    txt = tk.Text(frm, height=12, width=80)
    txt.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=8, pady=(8, 4))
    frm.rowconfigure(r, weight=1)

    def log(s: str):
        try:
            txt.insert("end", f"{s}\n")
        except Exception:
            # Failsafe if any editor split lines
            txt.insert("end", str(s))
            txt.insert("end", "\n")
        txt.see("end")
        root.update_idletasks()

    r += 1
    def on_run():
        try:
            mod_archive = Path(state["mod_archive"].get()).expanduser() if state["mod_archive"].get() else None
            input_xml = Path(state["input_xml"].get()).expanduser() if state["input_xml"].get() else None
            output_scenes = Path(state["output_scenes"].get()).expanduser()
            pack = state["pack"].get().strip()
            emit_alignment = Path(state["emit_alignment"].get()).expanduser() if state["emit_alignment"].get() else None
            merge_alignment = Path(state["merge_alignment"].get()).expanduser() if state["merge_alignment"].get() else None
            zip_out = Path(state["zip_out"].get()).expanduser() if state["zip_out"].get() else None

            if not output_scenes:
                messagebox.showerror("Missing output", "Select an output scenes folder"); return
            if not pack:
                messagebox.showerror("Missing pack name", "Enter a pack name"); return

            harvest_roots: List[Path] = []

            # If a mod archive is provided, it takes precedence
            if mod_archive:
                if not mod_archive.exists():
                    messagebox.showerror("Archive not found", str(mod_archive)); return
                log(f"Extracting mod archive: {mod_archive}")
                extracted = extract_mod_archive(mod_archive)
                work_xml_root = find_xml_root(extracted)
                harvest_roots = [work_xml_root, extracted]
                log(f"Found XML under: {work_xml_root}")
            else:
                if not input_xml or not input_xml.exists():
                    messagebox.showerror("Missing input", "Select a valid input XML folder or provide a mod archive"); return
                work_xml_root = input_xml
                harvest_roots = [input_xml]

            log("Converting…")
            scenes, alignment_written = run_conversion(
                input_xml=work_xml_root,
                output_scenes=output_scenes,
                pack=pack,
                emit_alignment=emit_alignment,
                merge_alignment_path=merge_alignment,
                harvest_search_roots=harvest_roots,
            )
            hkx_paths = collect_hkx_paths(scenes)
            log(f"OK: {len(scenes)} scenes written under {output_scenes / pack}")

            if zip_out:
                make_ready_to_install_zip(
                    zip_path=zip_out,
                    generated_scenes_root=output_scenes,
                    pack=pack,
                    alignment_path=alignment_written,
                    hkx_paths_for_auto=hkx_paths,
                )
                log(f"ZIP created: {zip_out}")

            messagebox.showinfo("Done", "Conversion complete.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            log(f"ERROR: {e}")

    ttk.Button(frm, text="Run Conversion", command=on_run)\
        .grid(row=r, column=0, columnspan=3, sticky="ew", padx=8, pady=8)

    root.mainloop()
    return 0


if __name__ == "__main__":
    # GUI-only entry point
    sys.argv = [sys.argv[0]]
    raise SystemExit(launch_gui())
