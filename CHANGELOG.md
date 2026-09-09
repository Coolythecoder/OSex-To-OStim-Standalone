# Changelog

## Unreleased

## 6.0.0 - 2026-09-04

- Added a transparent Nexus source release containing no EXE, bundled runtime, DLL/PYD, batch/PowerShell launcher, or nested archive. The dedicated build performs extension and file-signature checks, writes a payload SHA-256 manifest, and creates a versioned outer-ZIP hash.
- Kept the standalone Windows EXE as a separate distribution because Nexus Mods may require manual moderation for executable uploads regardless of whether antivirus scans are clean.
- Added automatic Skyrim LE animation detection from Havok packfile headers and automatic 32-bit-to-64-bit HKX conversion when a supported local helper is available.
- Added discovery and explicit selection for Creation Kit `HavokBehaviorPostProcess.exe`, Cathedral Assets Optimizer's `hkx32to64.exe`, and `hkxcmd.exe`, including `AAC_HKX_CONVERTER`/`HKX_CONVERTER` environment overrides and a GUI/CLI setting.
- Kept source archives and already compatible SE/AE HKX files unchanged. Conversion runs on isolated working copies and validates every helper result before packaging.
- Added diagnosis, OStim Tools project, conversion, and deployment-report coverage for legacy HKX detection and conversion. Final ZIP verification now fails if a Skyrim LE or unsupported 32-bit HKX file remains.

## 5.9.0 - 2026-08-22

- Clarified classic SexLab/SLAL registration in Pandora. These exports are auto-discovered from `FNIS_*_List.txt`, do not create a selectable Pandora checkbox, and now report the exact `FNIS Mod` name to confirm in `Engine.log`.
- Added expected-FNIS-list validation to Pandora log analysis so missing auto-discovery is diagnosed separately from a missing OStim/Nemesis module, with profile and `--tesv` guidance when Pandora is scanning the wrong game data.
- Hardened Windows packaging with a fresh pinned build environment, PyInstaller one-folder mode with UPX disabled, embedded Windows version metadata, forbidden-dependency auditing, build provenance, SHA-256 files, and optional Authenticode signing.

## 5.8.0 - 2026-08-20

- Fixed a false adult-only safety block on OStim packs that bundle Nemesis `defaultmale` or `defaultfemale` baseline behavior catalogs. Vanilla behavior references in those inherited catalogs are ignored, while archive paths, scene metadata, ATT registrations, and custom animation paths remain subject to the blocker.
- Validated Sanguine Seductions - OStim Animation Pack 3.0 from the original Nexus archive: diagnosis passes with source warnings and SexLab export produces 17 animations, 34 registered HKX events, and no missing deployment events.

## 5.7.0 - 2026-08-13

- Fixed Build Recommended Package overriding the user's `Human-only OStim output` checkbox with a compatibility-profile default.
- Changed creature-runtime marker scans to `INCONCLUSIVE` when MO2/Vortex virtualization may hide OCreatures or Creature Framework files; genuinely missing expected actor-root behavior files still fail.
- Updated Pandora `Engine.log` analysis for Pandora 4.4 merge-success messages, observed output artifacts, and log-specific output settings so an unrelated global configuration no longer causes a false failure.
- Fixed SexLab P+/SLSB actor sex export. Ordinary male actors now export as `male: true`, `female: false`, `futa: false`, while explicit source futa flags survive SLSB round trips.
- Added a periodic GUI heartbeat for large solid archives and behavior jobs so a long-running worker no longer appears frozen.
- Added exact compatibility profiles for K4 Anims 1.5 SE and SLSB Anub 12.2025 using tested archive fingerprints and observed conversion limits.
- Validated K4 human/creature registration and SLSB Anub human-only/P+ output with real source archives and Pandora Behaviour Engine+ 4.4 in isolated test roots.
- Kept the minor-coded-content blocker strict; blocked archives remain non-convertible and include a redacted compatibility auto-fail candidate for maintainer review.

## 2.0.0 - 2026-07-10

- Added a typed neutral IR with source provenance, stable IDs, unknown-field retention, and explicit loss accounting.
- Added bidirectional OStim SA and OSA/OSex adapters plus current OStim sequence support.
- Added conservative legacy converter JSON, SLAL, and Flower Girls import adapters.
- Pinned current OStim loader semantics to commit `ad138cbd3bee2a736a422389d715b36b792aa671`.
- Replaced the old `id` / `pack` / `poses` / `clips` target shape with filename IDs and event-based speeds.
- Separated logical scenes, events, actor mappings, HKX assets, and behavior registration.
- Removed implicit HKX-to-scene fallback from normal conversion; added explicit non-installable salvage mode.
- Stopped generating alignment templates; source offsets now map to current scene/actor `{x,y,z,r}` fields.
- Added graph, event, actor, action, furniture, sequence, layout, and install-readiness validation.
- Added bounded traversal-safe ZIP/7z staging, hash-verified asset copies, deterministic ZIPs, and atomic finalization.
- Added structured JSON/text reports and a round-trip conversion manifest that preserves source IDs and source-only provenance.
- Added a shared CLI/Tkinter service layer and stable exit codes.
- Added synthetic fixtures, regression/security/round-trip tests, Ruff, and Python 3.10-3.13 CI.
