# Adult Animation Converter

Convert legacy OSex, OSex+, OpenSex OSA animation packs, SexLab/SLAL animation packs, FlowerGirls FNIS animation packs, existing OStim Standalone packs, and older OStim JSON scene packs into ready-to-install OStim Standalone packages. Supported archives can also be exported as SexLab/SLAL, SexLab P+/SLSB packs, or editable OStim Tools/AAC JSON projects.

Current release: Version 6.0.

This tool is a Windows desktop app. Pick an original OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim Standalone `.zip`, `.7z`, or `.rar`, and it creates a new `_OStimSA.zip` beside it. Install that generated ZIP with your mod manager, then run Pandora or Nemesis.

## What It Does

- Converts legacy OSex/OSex+/OpenSex XML scene files, SexLab/SLAL JSON animation data, FlowerGirls FNIS animation lists, existing OStim Standalone scene JSON, and older OStim scene JSON into fresh OStim Standalone scene JSON.
- Copies included HKX animation files into an OStim Standalone-friendly package.
- Generates a pack-specific Nemesis/ATT behavior patch that Pandora can assemble for human animation events.
- Generates FNIS-style creature lists and matching behavior hooks when creature output is enabled.
- Rejects obsolete `Pandora_Engine` checkbox-only packages that can appear in Pandora but leave actors idle.
- Generates OCreatures-compatible menu/index adapter entries for creature-focused OStim output when creature output is enabled.
- Lets you enter the original mod author for the generated metadata.
- Creates conversion and deployment verification reports.
- Diagnoses source archives before conversion.
- Adds Simple Mode and Advanced Mode.
- Adds Build Recommended Package to diagnose, choose the normal recommended output, build, and verify automatically.
- Matches known pack patterns through editable `compatibility_db.json`.
- Copies Nexus-ready diagnosis and bug-report summaries.
- Creates Nexus-safe report bundles and archive structure exports for support.
- Adds `AAC_README.txt` and `AAC_MANIFEST.json` inside generated packages.
- Marks failed verification outputs with `FAILED_DoNotInstall_`.
- Verifies any converted ZIP before you install it.
- Adds an OStim Tools JSON generator that writes editable project folders or scene JSON folders, validates them, and imports them back into the converter for normal OStimSA packaging.
- Adds an OStim Tools 3.3.1-style bridge config companion under `configs/bridge/ostimConfigs/` for full project exports.
- Fixes duplicate generated OStim menu destinations in two-page category/page menus before packaging.
- Adds a one-click SexLab/SLAL export that writes SLAL JSON, SLAL source, renamed HKX stage files, FNIS animation lists, source-provided behavior graph hooks when available, and SexLab sound metadata.
- Adds an optional `SexLab P+ export` checkbox that also writes SLSB source, a compiled `.slr` registry, prefixed P+ behavior events, and matching behavior graph hooks when the source pack provides one.
- Adds default SexLab discovery tags to SLAL JSON/source and SexLab P+ SLSB stage tags, with a checkbox to turn them off.
- Adds a configurable OStim SFX fallback action for converted scenes that have no reliable legacy action metadata.
- Adds OStim furniture metadata for built-in furniture scenes and copies OStim custom furniture type files when present.
- Adds Human-only OStim output for mixed SexLab/SLAL packs, while still allowing advanced creature output for users with a creature-capable setup.

## Requirements

- Windows.
- OStim Standalone and its normal requirements.
- Pandora or Nemesis.
- A legacy OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim animation pack archive, usually `.zip`, `.7z`, or `.rar`.

Two distributions are available:

- **Nexus Source** contains no EXE, bundled runtime, DLL, or nested archive. Install Python 3.11, run `py -3.11 -m pip install --user -r requirements-runtime.txt` once, then double-click `Adult Animation Converter.pyw`.
- **Standalone Windows** is self-contained and does not require Python. Keep `_internal` beside `Adult Animation Converter.exe`. Nexus may hold executable uploads for moderator review even when they are clean.

Do not install either converter distribution with a mod manager or place it in Skyrim's Data folder.

For `.7z` or `.rar` archives, install 7-Zip normally. The converter looks for `7z.exe`, `7za.exe`, or `7zz.exe` on PATH, in common Program Files install folders, Windows registry entries, Chocolatey, Scoop, beside the app, and beside the selected archive. Advanced users can also set `AAC_7ZIP` to the full `7z.exe` path.

Human-only OStim output is recommended for most users and is enabled by default for OStim Standalone builds. It skips creature/animal-root scenes from mixed packs and reports what was skipped. Build Recommended Package uses the checkbox value you explicitly selected even when a known-pack profile recommends a different setting. Disable it only if you have a creature-capable OStim setup. That may include OCreatures or another OStim creature extension, Creature Framework, matching creature assets such as More Nasty Critters or the pack's required assets, and creature-capable behavior generation with Pandora or the creature behavior tool required by your setup. The converter can package creature scenes when you opt in and generates OCreatures-style menu/index entries for creature-focused output; it does not install the creature runtime stack.

## First Time User Guide

If you are new to modding or unsure which file to install, read `BEGINNER_GUIDE.md` first.

Short version:

1. Open `Adult Animation Converter.pyw` from the Nexus Source edition, or `Adult Animation Converter.exe` from the standalone edition.
2. Stay in `Simple Mode` unless you need custom advanced output fields.
3. Drag in the original OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim Standalone archive, or click `Choose Archive`.
4. Click `Diagnose Archive` and read the recommendation.
5. Click `Build Recommended Package`.
6. Install the new generated `_OStimSA.zip`, `_SexLab.zip`, or `_SexLabPPlus.zip`, not the original archive.
7. Run Pandora or Nemesis.
8. Launch Skyrim through your mod manager.

The generated ZIP is MO2-ready: its archive root contains folders such as `SKSE`, `meshes`, and `Nemesis_Engine`. You should not need to right-click a nested `Data` folder and set it as the data directory. Each converted pack receives a unique behavior module/list code so multiple converted packs can be installed together.

## Basic Use

1. Run `Adult Animation Converter.pyw` from the Nexus Source edition, or `Adult Animation Converter.exe` from the standalone edition.
2. Use `Simple Mode` for the normal one-click workflow.
3. Drag a `.zip`, `.7z`, or `.rar` archive onto the app, or click `Choose Archive`.
4. Click `Diagnose Archive` if you want to see the detected source type and recommendation first.
5. Click `Build Recommended Package`.
6. Wait for automatic verification to finish.
7. Install the generated `_OStimSA.zip`, `_SexLab.zip`, or `_SexLabPPlus.zip` with your mod manager.
8. Run Pandora or Nemesis.
9. Start Skyrim and test the scenes in game.

The generated ZIP appears next to the source archive unless you choose a custom output path in the advanced fields.

Advanced users can switch to `Advanced Mode`, enter the original mod author, change `SFX fallback action`, choose custom output paths, or manually click `Build OStim Standalone ZIP` / `Build SexLab / SLAL ZIP`.

To make a SexLab/SLAL package manually, click `Build SexLab / SLAL ZIP`. Install the generated `_SexLab.zip`, run GenerateFNISforUsers, Nemesis, or Pandora, then open SexLab Animation Loader in game and register or enable the pack. If the source archive includes `FNIS_*_Behavior.hkx`, the export keeps that hook and adds a matching hook for the generated `FNIS_<PackName>_List.txt`.

For SexLab P+, tick `SexLab P+ export` beside the SexLab button before building. Install the generated `_SexLabPPlus.zip`. It includes the classic SLAL files plus `SKSE/Sexlab/Registry/<PackName>.slr`, editable `SKSE/Sexlab/Registry/Source/<PackName>.slsb.json`, P+ prefixed behavior entries in the generated FNIS lists, and the same behavior graph hook support when available. Ordinary male positions export as male only; explicit futa flags from SLSB source data are preserved.

Leave `SexLab discovery tags` checked unless you want minimal original tags only. The converter adds `AdultAnimationConverter`, `Converted`, and a safe pack-name tag to the SLAL JSON/source files; SexLab P+ exports carry matching tags into SLSB stage data so the pack is easier to identify in SexLab tools.

## OStim Tools JSON Generator

Use `Generate OStim Tools JSON` when you want editable project/scene JSON for manual finishing in OStim Tools. This creates an `_OStimToolsProject` folder with `ostim_tools_project.json`, OStim Tools-clean editable `scenes/`, a project-local HKX animation asset bundle when source assets were available, an OStim Tools 3.3.1-style bridge config under `configs/bridge/ostimConfigs/`, optional action and alignment scaffolds, reports, `AAC_README.txt`, and `AAC_MANIFEST.json`.

Use `Generate Scene JSON Folder` when you only want loose editable scene JSON files plus reports. Neither output is a ready-to-install mod. To use edited JSON in game, validate/import the project back into the converter and build a verified OStimSA ZIP through the normal packaging path. Use the full project export, not the loose scene folder, when you want the converter to carry source HKX assets and the bridge config companion into a later import/build step.

## Verify Before Installing

Use `Verify Converted ZIP` to check a converted package before deploying it.

The verifier checks for:

- OStim scene JSON files.
- HKX animation files.
- Behavior animation list files.
- Generated or source behavior engine patch files.
- Nemesis/ATT event registration for OStim scene events.
- FNIS or Pandora behavior entries that point at missing HKX files.
- Pandora AnimData files.
- Pandora AnimSetData files.
- Pandora metadata.
- OStim menu hub entries, when generated.
- Generated OStim return and Previous/Next navigation counts, when menu entries are generated.
- Duplicate generated OStim menu destinations.
- Creature scene counts and creature actor roots.
- Missing animation events.
- At least one deployable OStim scene, including furniture-bound scenes.
- Missing custom furniture type files.
- Broken or unresolved scene links.
- Legacy/non-Standalone JSON mistakes.
- SexLab/SLAL JSON and source files when verifying `_SexLab.zip`.
- SexLab P+ SLSB source and compiled `.slr` registry files when verifying `_SexLabPPlus.zip`.
- Source-provided SexLab behavior graph hooks and matching generated FNIS behavior graph aliases.
- FNIS list entries that point at missing HKX files.

If verification passes, the ZIP is structurally ready to install. You still need to run Pandora, Nemesis, or FNIS after installing it, depending on the target setup.

Verification can return `PASS`, `PASS WITH WARNINGS`, or `FAIL`. `PASS WITH WARNINGS` means the ZIP is structurally usable but the report contains notes worth reading before install or testing, such as creature runtime requirements, optional metadata warnings, or behavior-generator caveats. `FAIL` means do not install that output yet.

If a build fails verification, the app marks the output with `FAILED_DoNotInstall_` so it is not mistaken for a normal installable package.

If conversion fails before a usable ZIP is created, the app writes `<OutputName>_conversion_failed.json` and `<OutputName>_conversion_failed.txt` beside the intended output when it can. These failure reports include the converter version, source archive name, source archive hash, detected source type, fatal error, warning severity groups, and a recommended next action for Nexus troubleshooting. If an adult source archive is blocked by the minor-coded content safety check, the failure report includes a redacted compatibility auto-fail candidate with a `compatibility_db.json` entry template that can be sent to the maintainer for review without sharing the source pack.

## Diagnosis And Compatibility

`Diagnose Archive` checks an original archive without building a ZIP. It reports source type, confidence, known pack match, supported outputs, recommended behavior tool, creature/furniture requirements, missing HKX references, and whether conversion is likely safe. Blocked adult source archives include a redacted compatibility auto-fail candidate with the archive hash and a ready-to-copy fail-entry template for `compatibility_db.json`.

Known-pack matching comes from `compatibility_db.json`. The database is editable and can be opened from the app. It only adds notes, warnings, and recommendations; it cannot make a broken ZIP pass verification.

Normal reports are safe to paste publicly because they use archive names and archive-relative paths. Enable `Debug mode` only when you want full local paths and extra internal details for private troubleshooting.

Use `Create Report Bundle` to create `AAC_BugReport_<PackName>.zip`. It includes public-safe reports, diagnosis info, verification summary, Pandora Module Diagnostics, imported Pandora log analysis when available, archive structure, app version, selected settings, and visible log text. It does not include HKX animation assets or source pack contents.

Use `Export Archive Structure` to save a source archive file listing with archive-relative paths only. This is useful for support without asking users to upload copyrighted packs.

## Version 6.0 Update

- Added automatic Skyrim LE HKX detection and conversion to Skyrim SE/AE format when a supported local helper is available.
- Added GUI/CLI selection and discovery for Creation Kit `HavokBehaviorPostProcess.exe`, Cathedral Assets Optimizer's `hkx32to64.exe`, and `hkxcmd.exe`.
- Source archives remain unchanged, existing SE/AE HKX files remain unchanged, and verification blocks output containing an unsupported 32-bit HKX.
- Added a transparent Nexus source distribution with no bundled executable/runtime or nested archive, plus strict file-signature auditing and SHA-256 manifests.
- Kept the hardened standalone Windows EXE as a separate folder-style download.

## Version 5.9 Update

- Clarified classic SexLab/SLAL registration in Pandora. Generated `FNIS_*_List.txt` files are auto-discovered and do not create a selectable converted-pack checkbox; reports now give the exact `FNIS Mod` name to find in `Engine.log`.
- Added expected-FNIS-list validation to the Pandora log importer, including clearer mod-manager profile, virtual `Data`, and `--tesv` guidance when auto-discovery is missing.
- Rebuilt the Windows release pipeline around a fresh pinned environment, one-folder packaging with UPX disabled, embedded version metadata, dependency auditing, build provenance, SHA-256 files, and optional Authenticode signing.

## Version 5.8 Update

- Fixed a false adult-only safety block on OStim packs that include Nemesis `defaultmale` or `defaultfemale` baseline behavior catalogs. Vanilla Skyrim behavior references in those inherited catalogs are ignored, while source archive paths, scene metadata, ATT registrations, and custom animation paths remain checked.
- Tested the fix with the original Sanguine Seductions - OStim Animation Pack 3.0 archive. Its SexLab conversion contains 17 animations, 34 matching HKX registrations, and no missing deployment events.

## Version 5.7 Update

- Fixed Build Recommended Package overriding the user's Human-only OStim output checkbox with a profile default.
- Changed virtualized creature-runtime marker misses to `INCONCLUSIVE`; missing expected actor-root behavior files still fail.
- Updated Pandora 4.4 log parsing to recognize current merge-success lines, inspect observed output files, and ignore unrelated saved output settings.
- Fixed SexLab P+/SLSB actor sex flags so ordinary male positions are not marked as futa while explicit source futa flags survive round trips.
- Added periodic GUI heartbeat messages during long archive and behavior operations.
- Added exact tested profiles for K4 Anims 1.5 SE and SLSB Anub 12.2025, and validated their generated registrations with Pandora 4.4 in isolated roots.
- Corrected the Pandora checkbox/idle-actor failure. OStim Standalone builds now use the Nemesis/ATT module format Pandora actually assembles, and verification blocks the obsolete `Pandora_Engine/mod/<Pack>/animationdata` layout.
- Validated generated human ATT and creature FNIS registration with Pandora Behaviour Engine+ 4.4.0 in an isolated Skyrim test root.
- Fixed remaining GUI worker reads of Tk values, the advanced `zip_out` crash, solid 7z archives, multi-JSON SLAL naming, branch/actor-root selection, and behavior HKX files leaking into animation assets.
- Fixed the remaining Baka/HCOS `'str' object has no attribute 'get'` failure by handling privacy-safe truncated report examples and legacy list/string report fields consistently.
- Completed a real BakaFactory SLAL Animation 78 human-only build with 58 deployable scenes, 672 registered HKX events, complete menu reachability, and no missing deployment events.
- Existing converted ZIPs made with the checkbox-only Pandora layout must be rebuilt from the original source archive.
- Fixed a diagnosis/report popup that could show `'str' object has no attribute 'get'` when older or malformed compatibility/report fields were plain text instead of structured data.
- Hardened public report renderers, source-selection summaries, compatibility matching, packaged README/manifest generation, and Nexus-safe report bundles so malformed legacy JSON fields are reported as not recorded instead of crashing the GUI.
- Added safer JSON boundary validation for compatibility profiles, OStim Tools project manifests, SLAL/SLSB/OStim scene parsing, AAC manifests, and report sections so strings in object fields produce clear warnings or validation errors.
- Improved GUI internal-error handling: unexpected background-job crashes now write a traceback log with converter version, operation, selected archive name, and debug-mode state, plus Copy Error Summary and Open Error Log support actions.
- Added BakaFactory SLAL Animation 78 as a known mixed SLSB/SLAL profile.
- Mixed SLSB/SLAL diagnosis now explains when SLSB source JSON is selected over SLAL JSON, which branch root was selected, and which duplicate SLAL LE/SE branches were ignored.
- Diagnosis reports now split BakaFactory-style mixed pack guidance into normal Human-only OStim, creature-capable OStim, SexLab P+, and OStim Tools project paths.
- Duplicate HKX reporting now summarizes high duplicate counts with capped examples and a clear de-duplication policy instead of flooding reports with source paths.
- Added Pandora `Engine.log` analysis for checkbox/no-output cases, with a GUI import button, CLI support, public-safe reports, and report-bundle export.
- Improved behavior-output verification. The converter detects and blocks checkbox-only Pandora modules, validates ATT/FNIS events against packaged HKX files, and gives clearer guidance when Pandora creates no generated output.
- Added Delta of Venus - Sex Fantasies for OStim NG 1.0 as a known OSA/OStim NG profile. Converter 5.4 or newer repairs its case-only XML/HKX speed mismatches and reports the one source scene skipped because its HKX files are absent from the tested archive.
- Source diagnosis now separates missing generated-output HKX events from missing skipped-source HKX events, so usable builds are no longer marked unsafe just because a broken source speed was dropped before packaging.
- Bumped the public release line to 5.7.
- Release ZIP and ZIP hash filenames now include the converter version automatically, such as `Adult Animation Converter 5.7.zip`.
- Added `BUILD_TESTER.ps1` for tester/beta packages named like `Adult Animation Converter 5.7 beta.zip`, with optional labels such as `beta1` and `beta2`.
- Build Recommended Package now follows known-pack behavior-tool recommendations and emits Pandora-compatible behavior output for OStim Standalone.
- Tidied Simple Mode into clearer Recommendation, Project Tools, Next Steps, and Reports & Files sections so install steps no longer overlap the support buttons.
- Improved creature-only SLAL reports and verification metadata: final manifests record post-build verification, public JSON reports redact local paths, creature-runtime warnings are deduplicated, and scene counts distinguish JSON files from playable/menu helper scenes.
- Fixed false unsafe-path failures when 7-Zip reports the source RAR/7z archive header as a drive-qualified `Path = ...` line.
- Fixed duplicate generated OStim menu destinations in two-page category menus. Packs that previously failed verification with duplicate `Page_1`/`Page_2` Previous/Next links should be rebuilt from the original source archive with Version 5.7.
- Added a FlufyFox SLAL SE Creature 3.6 compatibility profile that identifies the pack as creature-only and warns that default human-only OStim output will skip every scene.
- Corrected the FlufyFox SLAL SE Creature 3.6 profile so the exact known creature-only archive recommends creature output instead of default human-only output.
- Added explicit OCreatures-compatible creature output. Creature SLAL builds now report OCreatures menu/index files, actor-stage mapping, HKX filename policy, `-Tn` handling, and runtime requirements.
- Added an OCreatures Validation report section that groups the creature-output checks for HKX references, duplicate event IDs, actor-slot mapping, behavior backend selection, duplicate behavior backend registration, menu coverage, `-Tn` handling, and runtime dependency check status.
- Added an OCreatures reference comparison helper for maintainers. It compares structure, paths, event names, actor roots, and behavior registration against a known working output without comparing or redistributing HKX binary contents.
- Direct CLI archive builds can use exact-profile human-only/creature-output defaults. Build Recommended Package shows that recommendation but respects the user's current checkbox; human-only builds still fail clearly if they would skip every scene.
- Added `--check-creature-runtime` to scan a tester's Data/mod staging folder for creature runtime markers and expected actor roots from a converted ZIP.
- Marked a known minor-coded Billyy Petite archive as unsupported so diagnosis reports are clearer and do not present it as a normal compatible Billyy-style pack.
- Added redacted compatibility auto-fail candidate blocks to blocked source diagnosis and conversion failure reports, including the archive hash and a ready-to-copy `compatibility_db.json` entry template for maintainer review.
- Added OStim Tools JSON/project support. Users can now generate editable OStim Tools-style project folders or plain scene JSON folders from supported source archives, validate those projects, and import them back into the converter for normal OStimSA packaging. Full project exports can carry a project-local HKX asset bundle and an OStim Tools 3.3.1-style bridge config companion for later import/build steps. Added reports, CLI options, GUI controls, and documentation for the new JSON generator workflow.
- Added Simple Mode and Advanced Mode.
- Added Build Recommended Package, which diagnoses the selected archive, chooses the recommended output, builds it, and verifies it automatically.
- Added clearer recommended-workflow guidance, result summaries, copyable install steps, Copy Log, and Open Last Report behavior.
- Added Nexus-safe report bundles and archive structure export so users can send useful support info without uploading animation assets.
- Added `AAC_README.txt` and `AAC_MANIFEST.json` inside every generated package.
- Added stale generated-ZIP warnings and failed-output protection. Failed verification outputs are marked `FAILED_DoNotInstall_`.
- Added Human-only OStim output for mixed SLAL packs. Creature/animal-root scenes are skipped by default for OStim Standalone builds and are reported clearly, while advanced users can still opt into creature output.
- Improved large SLAL-to-OStim conversion handling for creature-heavy packs such as Billyy Petite. Added actor-root-aware event/HKX validation, fixed duplicate page navigation in generated OStim menus, corrected failed-build install guidance, and fixed missing diagnosis summaries in Nexus-safe report bundles.
- Fixed SLAL-to-OStim conversions where the generated pack menu could appear in OStim but contain no animations. Reports now show written, skipped, linked, and unlinked scene visibility.
- Improved large-pack OStim navigation with category/page menu hubs, generated return navigation, and generated Previous/Next links where safe.
- Improved deployment verification for empty menu hubs, menu reachability, missing menu destinations, duplicate menu links, stale metadata, and behavior registration issues.
- Fixed Drago-style diagnosis false positives when SLAL JSON event names and HKX filenames only differ by letter casing.

## Advanced Fields

- `Source mod archive`: source `.zip`, `.7z`, or `.rar`.
- `Input XML folder`: use this only if converting from an extracted folder.
- `Output scenes root`: optional manual output folder for generated scene JSON.
- `Pack name`: display/folder name for the converted pack.
- `Original mod author`: author shown in the generated Pandora metadata.
- `Emit alignment.json to`: optional alignment export path.
- `Merge alignment.json`: optional merge target for existing alignment data.
- `Ready-to-install ZIP`: optional custom output ZIP path.
- `OStim Tools output folder`: optional output folder for editable OStim Tools/AAC project or scene JSON folder generation.
- `OStim Tools template JSON`: optional template merged into the AAC project scaffold for future schema experiments.

Most users only need the one-click conversion button.

## Command Line Use

Convert an archive:

```powershell
& ".\Adult Animation Converter.exe" --mod-archive "path\to\OriginalAnimationPack.rar" --mod-author "Original Author" --sfx-fallback-action ostimconvertermoan
```

Verify a converted ZIP:

```powershell
& ".\Adult Animation Converter.exe" --verify-zip "path\to\ConvertedPack.zip"
```

Build a SexLab P+/SLSB ZIP:

```powershell
& ".\Adult Animation Converter.exe" --target sexlabplus --mod-archive "path\to\OriginalAnimationPack.rar" --mod-author "Original Author"
```

Generate and validate editable OStim Tools JSON:

```powershell
& ".\Adult Animation Converter.exe" --target ostimtools --mod-archive "path\to\OriginalAnimationPack.rar" --ostim-tools-project-out "path\to\MyProject"
& ".\Adult Animation Converter.exe" --validate-ostim-tools-project "path\to\MyProject\ostim_tools_project.json"
& ".\Adult Animation Converter.exe" --import-ostim-tools-project "path\to\MyProject" --target ostim --zip-out "path\to\MyPack_OStimSA.zip"
```

Generate only scene JSON:

```powershell
& ".\Adult Animation Converter.exe" --target scene-json-folder --mod-archive "path\to\OriginalAnimationPack.rar" --scene-json-folder-out "path\to\SceneJSON"
```

Advanced conversion:

```powershell
& ".\Adult Animation Converter.exe" --input-xml "path\to\Data\Meshes\0SA\mod\0Sex\scene" --output-scenes "path\to\Data\SKSE\Plugins\OStim\scenes" --pack "MyConvertedPack" --mod-author "Original Author" --sfx-fallback-action kissing --zip-out "MyConvertedPack_OStimSA.zip"
```

## Generated Package Contents

The generated ZIP usually includes:

```text
SKSE/Plugins/OStim/scenes/<PackName>/
SKSE/Plugins/OStim/actions/ostimconvertermoan.json
meshes/actors/character/animations/<PackName>/
meshes/actors/<CreatureRoot>/animations/<PackName>/
meshes/actors/character/animations/<PackName>/ATT_<UniqueCode>_animlist.txt
meshes/actors/<CreatureRoot>/animations/<PackName>/FNIS_*_List.txt
meshes/actors/<CreatureRoot>/behaviors/FNIS_*_Behavior.hkx
Nemesis_Engine/mod/<UniqueCode>/info.ini
Nemesis_Engine/mod/<UniqueCode>/0_master/
SKSE/Plugins/OStim/converter_metadata/<PackName>/metadata.json
AAC_README.txt
AAC_MANIFEST.json
conversion_report.json
conversion_report.txt
README_OStim_SA.txt
```

The `ostimconvertermoan.json` action file is included when the default fallback is used by at least one scene. If you set the fallback to a built-in OStim action such as `kissing`, no extra action file is needed.

The converter metadata manifest is a self-check record for the generated package. It stores the pack name, author, generated paths, scene/action index, and deployment counts so the Verify button can catch stale or edited ZIPs before you install them.

`AAC_README.txt` and `AAC_MANIFEST.json` are written into every generated package. They record the converter version, original source archive name, output type, install steps, behavior recommendation, verification status, and package counts.

## Pandora And Nemesis Notes

Pandora consumes the generated Nemesis/ATT-compatible files:

```text
Nemesis_Engine/mod/<UniqueCode>/info.ini
Nemesis_Engine/mod/<UniqueCode>/0_master/
meshes/actors/character/animations/<PackName>/ATT_<UniqueCode>_animlist.txt
```

Creature output also includes actor-root-specific `FNIS_*_List.txt` files and matching `FNIS_*_Behavior.hkx` hooks. Fresh builds do not write root `animdata/`, `animationsetdatasinglefile/`, or pseudo-Pandora animationdata folders. Unique module and list codes prevent two converted packs from overwriting one another.

Pandora supports Nemesis-format patches and FNIS creature inputs. The converter binds each generated scene event to its packaged HKX through those supported formats. A visible converted-pack checkbox is not required; old `Pandora_Engine/mod/<Pack>/info.xml` packages could show a checkbox without supplying anything Pandora's assembler used.

Classic SexLab/SLAL exports are auto-discovered through their generated `FNIS_<Pack>_List.txt` files and are not expected to appear as selectable Pandora patches. After Pandora finishes, its `Engine.log` should contain an `INFO : FNIS Mod ... : FNIS_<Pack>_List` entry. If it does, continue in game and register or enable the pack in SexLab Animation Loader. If it does not, check that the generated `_SexLab.zip` is enabled in the same mod-manager profile and that Pandora is scanning the correct Skyrim `Data`/virtual filesystem.

When `Add OStim menu entry` is enabled, the generated OStim scene JSON also includes menu hub scenes linked from OStim's built-in idle scenes. Small packs link directly from the pack hub. Large packs are split into generated categories and numbered pages so OStim does not have to render hundreds of scenes from one flat menu node. This gives the pack a visible in-game doorway from OStim's main scene menu without adding every converted animation as a separate top-level entry. Linked converted scenes also get a generated return navigation back to the menu node that opened them, and related scenes get generated Previous/Next shortcuts when the converter can safely group them by actor setup, action, furniture, and creature race.

If Pandora lists the converted pack but actors stand idle in game, do not keep testing the same old converted ZIP. Reconvert the original archive with the latest converter, install the new `_OStimSA.zip`, rerun Pandora, and make sure Pandora's generated output mod is enabled/deployed in your mod manager. A pack can be visible in Pandora while the actual motion events are not registered. If Pandora shows the converted pack as a checkbox but creates no files, use `Import Pandora Log` on Pandora's `Engine.log` and then create a report bundle.

Nemesis does not need a manually selected converted-pack checkbox for these packs. Install the generated ZIP, run `Update Engine` if Nemesis asks for it, then run `Launch Nemesis Behavior Engine`. Human events use the ATT list and generated patch:

```text
meshes/actors/character/animations/<PackName>/ATT_<UniqueCode>_animlist.txt
Nemesis_Engine/mod/<UniqueCode>/0_master/
```

If an older converted ZIP shows a converted-pack checkbox in Nemesis with `(null)` beside it, do not tick it. Reconvert the original archive with the latest converter.

If Nemesis reports `ERROR(2006)` with `File: animationdata`, remove the older converted ZIP from your mod manager, rebuild the original archive with this version or newer, install the fresh ZIP, and rerun Nemesis. Older builds wrote Pandora-style metadata under `Nemesis_Engine`, which Nemesis can mistake for a missing behavior XML patch.

## Third-Party License Notes

This release bundles GPL-3.0 Animlist Transition Tool template resources so converted OStim Standalone packages can include hidden Nemesis/ATT behavior patches without requiring users to install a separate compiler. The license and upstream notes are included in the app folder as `THIRD_PARTY_LICENSES.md` and under `_internal/assets/animlist_transition_tool/`.

## OStim SFX Notes

OStim sound effects come from action metadata, not from editing the HKX files. The converter writes normal OStim actions whenever it can infer them from OSex XML, SLAL tags, or OStim scene/action JSON. For old pose-style scenes with no reliable action type, it uses the `SFX fallback action` setting. The default `ostimconvertermoan` writes a small generated action file and references it in scene JSON so OStim voice/moan SFX can still play. SexLab/SLAL exports use the same inferred actions and fallback choice to write SexLab `sound` and stage sound metadata such as `Sucking`, `Squishing`, or `NoSound`. You can change the fallback to a built-in OStim action such as `kissing`, use another custom action name, or enter `none` to disable fallback actions.

SexLab exports do not use the OStim main-menu icon system. Their discoverability is handled through SexLab tags: generated SLAL and P+ output includes `AdultAnimationConverter`, `Converted`, and the safe pack folder name by default. Use `--no-sexlab-discovery-tags` or uncheck `SexLab discovery tags` if you want only the source pack's original tags.

## Furniture Notes

When the source clearly describes a built-in OStim furniture type, the converter writes OStim furniture metadata for the scene. Supported built-in targets include beds, chairs, benches, tables, alchemy tables, and enchanting tables. These scenes are counted as furniture-bound scenes in the report and should be started from matching furniture in OStim, not from the normal free-space list.

Legacy OStim packs that already include custom furniture type JSON under `Data/SKSE/Plugins/OStim/furniture types/` have those files copied into the generated ZIP. SexLab spawned-prop and anim-object scenes such as toy chairs, pillories, crosses, tilted wheels, glory holes, or other pack-specific objects are kept as normal selectable scenes unless they safely map to a real OStim furniture type, so they do not disappear from the in-game menu.

## Creature Notes

Human-only OStim output is enabled by default for OStim Standalone builds. The converter detects creature actor types, creature race metadata, creature/bestiality tags, and non-character HKX roots such as `horse`, `canine`, `draugr`, and `werewolfbeast`. In human-only mode, creature and mixed human/creature scenes are skipped before OStim menus and Pandora behavior registration are generated.

Advanced users can disable Human-only OStim output to include creature scenes. In that mode the converter enables OCreatures-compatible creature output by default, remaps SLAL creature actor-stage events using the working reference pattern `_A1_ -> _S#_1`, `_A2_ -> _S#_0`, `_A3_ -> _S#_2`, `_A4_ -> _S#_3`, and `_A5_ -> _S#_4`, preserves non-character actor roots, creates OCreatures `OCr<CreatureRoot>` menu/index adapter entries, and creates matching Pandora behavior registration for OStim output. SexLab and P+ exports keep FNIS lists aligned with the packaged actor-root HKX files.

Verification reports whether Human-only OStim output was enabled, how many creature scenes were detected or skipped, which actor roots were involved, whether the OCreatures menu/index entry was generated, and whether retained creature scenes are reachable from that OCreatures path. The OCreatures Validation section groups the hard checks for HKX references, duplicate event IDs after sanitisation, actor-slot mapping, one selected behavior path, duplicate behavior backend registration, OCreatures menu coverage, group-scene menu actor coverage, and `-Tn` handling. A pass means the converted ZIP contains matching scene or registry data, HKX files, and behavior registration for the target output. You still need the matching OStim or SexLab creature runtime, creature assets, and creature-capable behavior generation before included creature scenes can show and play in game.

## Missing Animation Warnings

Some legacy OSex XML files reference animations that are not actually included in the source archive. When this happens, the converter removes only the missing speed. If a scene has no valid speeds left, it is dropped instead of being packaged as broken. If a transition points at an old external OSex scene that is not inside the archive, the converter removes that dead destination and keeps the animation as an internal reachable scene. The report records these repairs as warnings.

OSex+ and OpenSex-style archives are supported when they still include OSA-style XML scene files and HKX animations. SexLab/SLAL packs are supported when they include `SLAnims/json/*.json` plus matching HKX animations; the converter normalizes SLAL actor/stage HKX names into OStim event names, infers OStim actions from SLAL tags for SFX, and can export native SexLab P+ `.slr` registry files when `SexLab P+ export` is checked. FlowerGirls packs are supported when they include `FNIS_FlowerGirlsSE_List.txt` or another FlowerGirls-marked FNIS list plus matching HKX animations; the converter reads the FNIS event/file pairs, maps actor/stage events into OStim speeds, and preserves behavior graph hooks when the source provides them. OStim Standalone packs are supported when they include `SKSE/Plugins/OStim/scenes/*.json` plus HKX animations, with or without a top-level `Data` wrapper; custom action files under `SKSE/Plugins/OStim/actions/` are copied into the generated package and upgraded with OStim voice/moan SFX metadata. The converter preserves modern OStim scene fields such as speed metadata, actor metadata, navigation metadata, furniture, tags, and source behavior engine patches while rebuilding behavior/Pandora files and reports. The scanner recognizes `0SA`, `OSA`, `OSA+`, `0Sex`, `OSex+`, OpenSex, `SLAnims/json`, FlowerGirls FNIS lists, and OStim scene paths, and ignores installer XML such as `fomod/ModuleConfig.xml`.

FlowerGirls support converts animation content into OStim scenes. It does not convert FlowerGirls quests, dialogue, spells, scripts, or framework features. If a FlowerGirls archive only contains files such as `.esp`, `.pex`, `.psc`, or `.seq`, install it normally with FlowerGirls instead of converting it.

If verification reports missing animation events after conversion, the source archive may be incomplete or may not be a compatible OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim animation pack.

## Troubleshooting

### The app does not open

Try running the `.exe` from a terminal so Windows can show the error:

```powershell
& ".\Adult Animation Converter.exe"
```

If Windows shows `Windows protected your PC` or `not commonly downloaded`, read `README_SECURITY.md` and verify the published hashes. That is a SmartScreen reputation warning, not a named Defender Antivirus detection.

### Windows Defender says the app is a virus

Use the folder-style release ZIP instead of an older single-file EXE. Extract the whole `Adult Animation Converter` folder and run the EXE from inside that folder; do not copy the EXE away from `_internal`. Current releases are built in a fresh pinned environment with UPX disabled, embedded version metadata, a forbidden-dependency audit, and `BUILD_PROVENANCE.txt`.

The release includes `README_SECURITY.md`, `BUILD_PROVENANCE.txt`, and `SHA256SUMS.txt` so users can compare the published SHA256 hash with the file they downloaded. If Defender gives a named detection, do not disable protection or add a broad exclusion. Keep the file quarantined, report the exact detection name and hash, and submit an incorrectly detected clean release at `https://www.microsoft.com/en-us/wdsi/filesubmission`.

### The converter cannot open a `.7z` or `.rar`

Install 7-Zip normally, then try again. The converter can find normal 7-Zip installs without you editing PATH. If you use a portable copy, put `7z.exe` beside the converter or set `AAC_7ZIP` to the full `7z.exe` path.

### OStim scenes appear but animations do not play

Install the generated `_OStimSA.zip` with your mod manager, then run Pandora or Nemesis. Behavior generation is still required.

### Verification fails

Open the generated `_deploy_verify.txt` report beside the ZIP. It lists the exact missing files or missing animation events.

## Credits

This tool only converts and packages existing source archives. Credit for the original animation content belongs to the original mod authors. Use the `Original mod author` field so the generated Pandora metadata reflects the correct author.
