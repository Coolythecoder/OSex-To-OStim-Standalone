"""Friendly CLI wrapper for Adult Animation Converter.

This entry point provides the concise interface requested by testers:

    python convert.py --input "Source.zip" --output "Converted_OStimSA.zip"

The implementation delegates to Osex-to-OStim-Standalone.py so the GUI,
legacy CLI, deployment verifier, and this wrapper share the same conversion
and validation code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
CONVERTER_SCRIPT = ROOT / "Osex-to-OStim-Standalone.py"
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar"}
AUTO_REPORT = "__auto__"


def load_converter() -> Any:
    spec = importlib.util.spec_from_file_location("aac_converter", CONVERTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter module from {CONVERTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = load_converter()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a user-supplied legacy OSex/OSA/OpenSex-style archive or extracted folder "
            "into a verified OStim Standalone ZIP. The tool never writes into Skyrim's Data folder "
            "and never modifies the input path."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Source .zip/.7z/.rar archive, extracted source folder, or converted ZIP for --validate-only.")
    parser.add_argument("--output", type=Path, help="Output ZIP path, or an output folder where <Pack>_OStimSA.zip will be written.")
    parser.add_argument("--pack", help="Optional display/internal pack name. Defaults to the source name.")
    parser.add_argument("--mod-author", help="Optional original author name for generated metadata.")
    parser.add_argument(
        "--legacy-hkx-converter",
        type=Path,
        help=(
            "Optional HavokBehaviorPostProcess.exe, hkx32to64.exe, or hkxcmd.exe used to convert "
            "detected Skyrim LE animations to SE/AE automatically."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Diagnose the source and write reports without creating an installable ZIP.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing converted ZIP, or diagnose a source without building.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output ZIP.")
    parser.add_argument(
        "--pandora",
        action="store_true",
        help="Generate the Pandora-compatible Nemesis/ATT behavior patch. This is the default.",
    )
    parser.add_argument(
        "--nemesis",
        action="store_true",
        help="Compatibility alias for the same Pandora-compatible Nemesis/ATT patch.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra source/output details.")
    parser.add_argument(
        "--report-json",
        nargs="?",
        const=AUTO_REPORT,
        default=None,
        metavar="PATH",
        help="Write a public-safe JSON report. With no PATH, writes next to --output or --input.",
    )
    parser.add_argument(
        "--report-md",
        nargs="?",
        const=AUTO_REPORT,
        default=None,
        metavar="PATH",
        help="Write a Markdown report. With no PATH, writes next to --output or --input.",
    )
    return parser


def is_supported_archive(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ARCHIVE_SUFFIXES


def looks_like_converted_zip(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".zip":
        return False
    try:
        with converter.ZipFile(path, "r") as archive:
            entries = {converter.norm_slashes(name).strip("/").lower() for name in archive.namelist()}
    except Exception:
        return True
    marker_names = {
        converter.AAC_MANIFEST_FILE.lower(),
        "conversion_report.json",
        "sexlab_conversion_report.json",
    }
    if any(converter.PurePosixPath(entry).name in marker_names for entry in entries):
        return True
    return any("skse/plugins/ostim/converter_metadata/" in entry for entry in entries)


def output_zip_path(source: Path, output: Optional[Path], pack: Optional[str]) -> Path:
    pack_name = pack.strip() if pack and pack.strip() else source.stem if source.is_file() else source.name
    safe_pack = converter.safe_pack_folder(pack_name or "ConvertedPack")
    if output is None:
        base_dir = source.parent if source.parent else Path.cwd()
        return base_dir / f"{safe_pack}_OStimSA.zip"
    expanded = output.expanduser()
    if expanded.suffix.lower() == ".zip":
        return expanded
    return expanded / f"{safe_pack}_OStimSA.zip"


def report_base_path(source: Path, output: Optional[Path], pack: Optional[str]) -> Path:
    if output is None:
        if source.is_dir():
            pack_name = pack.strip() if pack and pack.strip() else source.name
            return source.parent / (converter.safe_pack_folder(pack_name or "SourceFolder"))
        return source
    expanded = output.expanduser()
    if expanded.suffix:
        return expanded
    return output_zip_path(source, expanded, pack)


def resolve_report_path(option_value: Optional[str], default_path: Path) -> Optional[Path]:
    if option_value is None:
        return None
    if option_value == AUTO_REPORT:
        return default_path
    requested = Path(option_value).expanduser()
    if requested.suffix:
        return requested
    requested.mkdir(parents=True, exist_ok=True)
    return requested / default_path.name


def public_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return converter.public_safe_report_value(report, debug_mode=converter.app_debug_mode())


def write_optional_reports(
    *,
    report: Dict[str, Any],
    text: str,
    report_json: Optional[str],
    report_md: Optional[str],
    default_json: Path,
    default_md: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    json_path = resolve_report_path(report_json, default_json)
    md_path = resolve_report_path(report_md, default_md)
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(public_report(report), indent=2, ensure_ascii=False), encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def prepare_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise RuntimeError(f"Output ZIP path is a directory: {path}")
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def diagnose_source(source: Path, args: argparse.Namespace) -> int:
    base = report_base_path(source, args.output, args.pack)
    report_dir = base.parent if base.suffix else base
    report_dir.mkdir(parents=True, exist_ok=True)
    if is_supported_archive(source):
        diagnosis = converter.diagnose_source_archive(
            source,
            report_dir=report_dir,
            legacy_hkx_converter=args.legacy_hkx_converter,
        )
    elif source.is_dir():
        diagnosis = diagnose_source_folder(
            source,
            report_dir=report_dir,
            legacy_hkx_converter=args.legacy_hkx_converter,
        )
    else:
        raise RuntimeError("Dry-run diagnosis needs a .zip/.7z/.rar source archive or an extracted source folder.")

    text = converter.source_diagnosis_report_to_text(public_report(diagnosis.report))
    default_json = base.with_name(f"{base.stem}_source_diagnosis.json")
    default_md = base.with_name(f"{base.stem}_source_diagnosis.md")
    json_path, md_path = write_optional_reports(
        report=diagnosis.report,
        text=text,
        report_json=args.report_json,
        report_md=args.report_md,
        default_json=default_json,
        default_md=default_md,
    )
    status = diagnosis.report.get("status") or ("PASS" if diagnosis.report.get("ok") else "FAIL")
    print(f"[{'ok' if diagnosis.report.get('ok') else 'error'}] Source diagnosis {status}: {source}")
    print(f"[info] Detected source type: {diagnosis.report.get('detectedSourceType')}")
    print(f"[info] Recommended action: {diagnosis.report.get('recommendedActionOneLine') or diagnosis.report.get('recommendedUserAction')}")
    if diagnosis.report_path:
        print(f"[info] JSON report: {diagnosis.report_path}")
    if diagnosis.text_report_path:
        print(f"[info] Text report: {diagnosis.text_report_path}")
    if json_path:
        print(f"[info] Requested JSON report: {json_path}")
    if md_path:
        print(f"[info] Requested Markdown report: {md_path}")
    return 0 if diagnosis.report.get("ok") else 1


def diagnose_source_folder(
    source: Path,
    report_dir: Path,
    legacy_hkx_converter: Optional[Path] = None,
) -> Any:
    source = source.expanduser()
    source_detection = converter.detect_source_pack_summary(source)
    source_detection = converter.add_compatibility_match_to_source_detection(source_detection, None, source)
    branch_layout = converter.archive_branch_layout_report(
        source,
        str(source_detection.get("selectedSourceType") or "unknown"),
        source_detection,
    )
    source_detection["archiveBranchLayout"] = branch_layout
    source_detection["sourceSelection"] = converter.source_selection_report(
        source_detection,
        source_detection.get("compatibilityMatch") if isinstance(source_detection.get("compatibilityMatch"), dict) else {},
        branch_layout,
    )
    temp_zip = report_dir / f"{converter.safe_pack_folder(source.name or 'SourceFolder')}_folder_diagnosis_source.zip"
    try:
        write_source_folder_archive(source, temp_zip)
        return converter.diagnose_source_archive(
            temp_zip,
            report_dir=report_dir,
            legacy_hkx_converter=legacy_hkx_converter,
        )
    finally:
        try:
            temp_zip.unlink()
        except OSError:
            pass


def write_source_folder_archive(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with converter.ZipFile(archive, "w", compression=converter.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*"), key=lambda item: str(item).lower()):
            if path.is_dir():
                continue
            if path.is_symlink():
                raise RuntimeError(f"Source folder contains unsupported symlink: {path}")
            rel = path.relative_to(source)
            name = converter.PurePosixPath(*rel.parts).as_posix()
            converter.validate_archive_member_name(name)
            zf.write(path, arcname=name)


def convert_source_folder_to_ready_zip(
    source: Path,
    zip_path: Path,
    *,
    pack: Optional[str],
    mod_author: Optional[str],
    nemesis_safe_output: bool,
    legacy_hkx_converter: Optional[Path] = None,
) -> Any:
    source = source.expanduser()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    pack_name = pack.strip() if pack and pack.strip() else source.name
    author = converter.normalized_mod_author(mod_author)
    source_detection: Dict[str, Any] = {}
    working_output: Optional[Path] = None
    try:
        source_detection = converter.detect_source_pack_summary(source)
        source_detection = converter.add_compatibility_match_to_source_detection(source_detection, None, source)
        branch_layout = converter.archive_branch_layout_report(
            source,
            str(source_detection.get("selectedSourceType") or "unknown"),
            source_detection,
        )
        source_detection["archiveBranchLayout"] = branch_layout
        source_detection["sourceSelection"] = converter.source_selection_report(
            source_detection,
            source_detection.get("compatibilityMatch") if isinstance(source_detection.get("compatibilityMatch"), dict) else {},
            branch_layout,
        )
        source_detection["sourceInputType"] = "folder"
        source_detection["sourceFolderName"] = source.name
        converter.enforce_adult_only_source(source_detection)

        effective_human_only = converter.default_human_only_ostim_from_source_detection(source_detection)
        input_root = converter.find_scene_source_root(source)
        working_output = converter.make_converter_temp_dir("ostim_sa_scenes_", zip_path)
        alignment_path = working_output / "alignment.json"
        result = converter.run_conversion(
            input_xml=input_root,
            output_scenes=working_output,
            pack=pack_name,
            emit_alignment=alignment_path,
            harvest_search_roots=[source, input_root],
            sfx_fallback_action=converter.DEFAULT_SFX_FALLBACK_ACTION_TYPE,
            ostim_menu_entry=True,
            ostim_menu_icon=converter.DEFAULT_OSTIM_MENU_ICON,
            human_only_ostim=effective_human_only,
            ocreatures_compatible_output=True,
            legacy_hkx_converter=legacy_hkx_converter,
        )
        converter.make_ready_to_install_zip(
            zip_path=zip_path,
            generated_scenes_root=working_output,
            pack=pack_name,
            alignment_path=result.alignment_written,
            hkx_assets=result.hkx_assets,
            scenes=result.scenes,
            source_archive=None,
            conversion_warnings=result.warnings,
            mod_author=author,
            sfx_fallback_action=converter.DEFAULT_SFX_FALLBACK_ACTION_TYPE,
            ostim_action_files=result.ostim_action_files,
            ostim_furniture_type_files=result.ostim_furniture_type_files,
            behavior_graphs=result.behavior_graphs,
            source_engine_mods=result.source_engine_mods,
            source_detection=source_detection,
            ostim_menu_entry_requested=True,
            conversion_diagnostics=result.diagnostics,
            nemesis_safe_output=nemesis_safe_output,
        )
        packaged_report = converter.read_packaged_conversion_report(zip_path) or converter.build_conversion_report(
            result.scenes,
            result.hkx_assets,
            pack_name,
            source_archive=None,
            conversion_warnings=result.warnings,
            mod_author=author,
            sfx_fallback_action=converter.DEFAULT_SFX_FALLBACK_ACTION_TYPE,
            source_detection=source_detection,
            output_type="OStim Standalone",
            ostim_menu_entry_requested=True,
            conversion_diagnostics=result.diagnostics,
        )
        verification = converter.verify_converted_zip(zip_path)
        final_zip = converter.mark_failed_zip_do_not_install(verification)
        packaged_report = converter.read_packaged_conversion_report(final_zip) or packaged_report
        packaged_report["postBuildVerification"] = converter.verification_summary_for_report(verification)
        converter.write_packaged_conversion_report_files(final_zip, packaged_report, verification)
        converter.write_external_conversion_report_files(final_zip, packaged_report, verification)
        return converter.ArchiveConversionResult(
            final_zip,
            pack_name,
            author,
            result.scenes,
            result.hkx_assets,
            packaged_report,
            result.warnings,
            result.ostim_action_files,
            result.ostim_furniture_type_files,
            verification,
        )
    except Exception as exc:
        converter.write_conversion_failure_report(zip_path, source, "OStim Standalone", exc, source_detection=source_detection)
        raise
    finally:
        converter.cleanup_converter_temp_dir(working_output)


def build_zip(source: Path, args: argparse.Namespace) -> int:
    out_zip = output_zip_path(source, args.output, args.pack)
    prepare_output_path(out_zip, args.overwrite)
    if args.verbose:
        behavior = "Pandora-compatible Nemesis/ATT patch"
        print(f"[info] Source: {source}")
        print(f"[info] Output ZIP: {out_zip}")
        print(f"[info] Behavior output: {behavior}")

    if is_supported_archive(source):
        result = converter.convert_archive_to_ready_zip(
            source,
            zip_path=out_zip,
            pack=args.pack,
            mod_author=args.mod_author,
            nemesis_safe_output=True,
            legacy_hkx_converter=args.legacy_hkx_converter,
        )
    elif source.is_dir():
        result = convert_source_folder_to_ready_zip(
            source,
            out_zip,
            pack=args.pack,
            mod_author=args.mod_author,
            nemesis_safe_output=True,
            legacy_hkx_converter=args.legacy_hkx_converter,
        )
    else:
        raise RuntimeError("Input must be a .zip/.7z/.rar archive or an extracted source folder.")

    verification = result.verification
    report = result.report
    text = converter.conversion_report_readme_text(report, verification, debug_mode=converter.app_debug_mode())
    default_json = result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report.json")
    default_md = result.zip_path.with_name(f"{result.zip_path.stem}_conversion_report.md")
    json_path, md_path = write_optional_reports(
        report=report,
        text=text,
        report_json=args.report_json,
        report_md=args.report_md,
        default_json=default_json,
        default_md=default_md,
    )
    print(f"[ok] ZIP created: {result.zip_path}")
    print(f"[ok] Scenes converted: {len(result.scenes)}")
    if verification:
        status = verification.report.get("status") or ("PASS" if verification.ok else "FAIL")
        print(f"[{'ok' if verification.ok else 'error'}] Post-build verification: {status}")
        if verification.text_report_path:
            print(f"[info] Verification report: {verification.text_report_path}")
    if json_path:
        print(f"[info] Requested JSON report: {json_path}")
    if md_path:
        print(f"[info] Requested Markdown report: {md_path}")
    return 0 if not verification or verification.ok else 1


def validate_only(source: Path, args: argparse.Namespace) -> int:
    if looks_like_converted_zip(source):
        result = converter.verify_converted_zip(source)
        text = (
            converter.sexlab_verification_report_to_text(public_report(result.report))
            if result.report.get("type") == "sexlabDeploymentVerification"
            else converter.verification_report_to_text(public_report(result.report))
        )
        base = report_base_path(source, args.output, args.pack)
        default_json = base.with_name(f"{base.stem}_deploy_verify.json")
        default_md = base.with_name(f"{base.stem}_deploy_verify.md")
        json_path, md_path = write_optional_reports(
            report=result.report,
            text=text,
            report_json=args.report_json,
            report_md=args.report_md,
            default_json=default_json,
            default_md=default_md,
        )
        status = result.report.get("status") or ("PASS" if result.ok else "FAIL")
        print(f"[{'ok' if result.ok else 'error'}] Deploy verification {status}: {source}")
        if result.report_path:
            print(f"[info] JSON report: {result.report_path}")
        if result.text_report_path:
            print(f"[info] Text report: {result.text_report_path}")
        if json_path:
            print(f"[info] Requested JSON report: {json_path}")
        if md_path:
            print(f"[info] Requested Markdown report: {md_path}")
        return 0 if result.ok else 1
    return diagnose_source(source, args)


def run_cli(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only cannot be used together.")
    source = args.input.expanduser()
    if not source.exists():
        print(f"[error] Input not found: {source}")
        return 1
    try:
        if args.validate_only:
            return validate_only(source, args)
        if args.dry_run:
            return diagnose_source(source, args)
        return build_zip(source, args)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
