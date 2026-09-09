from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .changelog_helper import changelog_draft

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Osex-to-OStim-Standalone.py"
COMPAT_DB = ROOT / "compatibility_db.json"
BUILD_SCRIPT = ROOT / "BUILD_RELEASE.ps1"
NEXUS_SOURCE_BUILD_SCRIPT = ROOT / "BUILD_NEXUS_SOURCE.ps1"
RELEASE_REQUIREMENTS = ROOT / "requirements-release.txt"
RUNTIME_REQUIREMENTS = ROOT / "requirements-runtime.txt"
REQUIRED_DOCS = [
    "README.md",
    "README_NEXUSMODS.md",
    "BEGINNER_GUIDE.md",
    "TROUBLESHOOTING.md",
    "COMPATIBILITY.md",
    "BUG_REPORT_TEMPLATE.md",
    "README_SECURITY.md",
    "THIRD_PARTY_LICENSES.md",
    "NEXUSMODS_CHANGELOG.txt",
]


def converter_version() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'^CONVERTER_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find CONVERTER_VERSION in Osex-to-OStim-Standalone.py")
    return match.group(1)


def release_zip_name(version: str, suffix: str = "") -> str:
    suffix = suffix.strip()
    label = f"{version} {suffix}" if suffix else version
    return f"Adult Animation Converter {label}.zip"


def nexus_source_zip_name(version: str, suffix: str = "") -> str:
    suffix = suffix.strip()
    label = f"{version} {suffix}" if suffix else version
    return f"Adult Animation Converter {label} Nexus Source.zip"


def run_step(name: str, command: list[str], *, shell: bool = False) -> None:
    print(f"[release-check] {name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, shell=shell)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def check_docs() -> None:
    missing = [doc for doc in REQUIRED_DOCS if not (ROOT / doc).is_file()]
    if missing:
        raise RuntimeError(f"Missing required docs: {', '.join(missing)}")


def check_compatibility_db() -> None:
    data = json.loads(COMPAT_DB.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("compatibility_db.json root must be an object")
    if not data.get("version"):
        raise RuntimeError("compatibility_db.json must include version")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("compatibility_db.json must include at least one entry")
    required = {"id", "packDisplayName", "sourceFramework", "status"}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"compatibility_db.json entry {index} is not an object")
        missing = sorted(required - set(entry))
        if missing:
            raise RuntimeError(f"compatibility_db.json entry {index} missing: {', '.join(missing)}")


def check_version_mentions(version: str) -> None:
    changelog = changelog_draft(version)
    if f"Version {version}" not in changelog:
        raise RuntimeError("Generated changelog draft did not include the current converter version")


def check_release_build_configuration() -> None:
    if not BUILD_SCRIPT.is_file():
        raise RuntimeError("Missing BUILD_RELEASE.ps1")
    if not RELEASE_REQUIREMENTS.is_file():
        raise RuntimeError("Missing requirements-release.txt")
    if not NEXUS_SOURCE_BUILD_SCRIPT.is_file():
        raise RuntimeError("Missing BUILD_NEXUS_SOURCE.ps1")
    if not RUNTIME_REQUIREMENTS.is_file():
        raise RuntimeError("Missing requirements-runtime.txt")
    script_text = BUILD_SCRIPT.read_text(encoding="utf-8")
    required_snippets = (
        ".release-venv",
        "requirements-release.txt",
        "--onedir",
        "--noupx",
        "--version-file",
        "Set-AuthenticodeSignature",
        "BUILD_PROVENANCE.txt",
        "forbiddenRuntimeRoots",
    )
    missing = [snippet for snippet in required_snippets if snippet not in script_text]
    if missing:
        raise RuntimeError(f"Release build hardening is incomplete: {', '.join(missing)}")
    forbidden_snippets = ("--onefile", "--debug noarchive", "\npyinstaller `")
    present = [snippet for snippet in forbidden_snippets if snippet in script_text]
    if present:
        raise RuntimeError(f"Release build uses a forbidden global/high-risk option: {', '.join(present)}")

    requirements = {
        line.split("==", 1)[0].strip().lower(): line.split("==", 1)[1].strip()
        for line in RELEASE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" in line
    }
    required_packages = {"pyinstaller", "pyinstaller-hooks-contrib", "customtkinter", "pillow"}
    missing_packages = sorted(required_packages - set(requirements))
    if missing_packages:
        raise RuntimeError(f"Release requirements are missing pinned packages: {', '.join(missing_packages)}")

    nexus_script_text = NEXUS_SOURCE_BUILD_SCRIPT.read_text(encoding="utf-8")
    nexus_required_snippets = (
        "Nexus Source.zip",
        "Get-BlockedFileReason",
        "nested ZIP signature",
        "nested 7z signature",
        "nested RAR signature",
        "PE executable signature",
        "SOURCE_RELEASE_MANIFEST.txt",
        "Compress-Archive",
    )
    nexus_missing = [snippet for snippet in nexus_required_snippets if snippet not in nexus_script_text]
    if nexus_missing:
        raise RuntimeError(f"Nexus source build audit is incomplete: {', '.join(nexus_missing)}")

    runtime_packages = {
        line.split("==", 1)[0].strip().lower(): line.split("==", 1)[1].strip()
        for line in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" in line
    }
    if set(runtime_packages) != {"customtkinter", "pillow"}:
        raise RuntimeError("Runtime requirements must contain only pinned customtkinter and Pillow dependencies")


def check_app_importable() -> None:
    code = (
        "import importlib.util, pathlib, sys; "
        "script = pathlib.Path('Osex-to-OStim-Standalone.py').resolve(); "
        "spec = importlib.util.spec_from_file_location('aac_release_import_check', script); "
        "module = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module); "
        "assert getattr(module, 'APP_NAME', '') == 'Adult Animation Converter'"
    )
    run_step(
        "compile",
        [sys.executable, "-m", "py_compile", "Osex-to-OStim-Standalone.py", "convert.py", "Adult Animation Converter.pyw"],
    )
    run_step("import", [sys.executable, "-c", code])
    run_step("app help", [sys.executable, "Osex-to-OStim-Standalone.py", "--help"])
    run_step("convert.py help", [sys.executable, "convert.py", "--help"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run release checks for Adult Animation Converter.")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run the regression test suite.")
    parser.add_argument("--skip-build", action="store_true", help="Do not run BUILD_RELEASE.ps1.")
    parser.add_argument("--print-changelog", action="store_true", help="Print a Nexus changelog draft.")
    args = parser.parse_args(argv)

    version = converter_version()
    print(f"[release-check] Converter version: {version}")
    check_docs()
    print("[release-check] Docs: OK")
    check_compatibility_db()
    print("[release-check] compatibility_db.json: OK")
    check_version_mentions(version)
    print("[release-check] changelog helper: OK")
    check_release_build_configuration()
    print("[release-check] isolated release build: OK")
    check_app_importable()
    print("[release-check] import check: OK")

    if not args.skip_tests:
        run_step("pytest", [sys.executable, "-m", "pytest", "-q"])
        run_step("legacy direct tests", [sys.executable, "tests\\test_converter.py"])

    if not args.skip_build and (ROOT / "BUILD_RELEASE.ps1").is_file():
        run_step("build", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "BUILD_RELEASE.ps1")])
        expected_zip = ROOT / release_zip_name(version)
        expected_hash = ROOT / f"{expected_zip.name}.sha256.txt"
        if not expected_zip.is_file():
            raise RuntimeError(f"Build did not create expected versioned ZIP: {expected_zip.name}")
        if not expected_hash.is_file():
            raise RuntimeError(f"Build did not create expected ZIP hash file: {expected_hash.name}")
        run_step(
            "Nexus source build",
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(NEXUS_SOURCE_BUILD_SCRIPT)],
        )
        expected_source_zip = ROOT / nexus_source_zip_name(version)
        expected_source_hash = ROOT / f"{expected_source_zip.name}.sha256.txt"
        if not expected_source_zip.is_file():
            raise RuntimeError(f"Build did not create expected Nexus source ZIP: {expected_source_zip.name}")
        if not expected_source_hash.is_file():
            raise RuntimeError(f"Build did not create expected Nexus source ZIP hash file: {expected_source_hash.name}")

    if args.print_changelog:
        print()
        print(changelog_draft(version), end="")

    print("[release-check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
