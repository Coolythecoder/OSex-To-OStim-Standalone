from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Osex-to-OStim-Standalone.py"


def converter_version() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'^CONVERTER_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def changelog_draft(version: str | None = None) -> str:
    version = version or converter_version()
    return "\n".join(
        [
            f"Version {version}",
            "",
            "* Added automatic Skyrim LE Havok animation detection and 32-bit-to-64-bit HKX conversion when a supported local helper is available.",
            "* Added discovery and explicit GUI/CLI selection for Creation Kit HavokBehaviorPostProcess.exe, Cathedral Assets Optimizer hkx32to64.exe, and hkxcmd.exe.",
            "* Kept source archives and existing SE/AE HKX files unchanged, validated every helper result, and blocked final packages that still contain unsupported 32-bit HKX files.",
            "* Added a transparent Nexus source release with no EXE, bundled runtime, DLL/PYD, batch/PowerShell launcher, or nested archive, plus payload and outer-ZIP SHA-256 hashes.",
            "* Kept the hardened one-folder Windows EXE as a separate distribution.",
            "",
        ]
    )


def main() -> int:
    print(changelog_draft(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
