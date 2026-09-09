# Adult Animation Converter

Current public desktop release: Version 6.0. The packaged Windows app and the full GUI/CLI entry point are built from `Osex-to-OStim-Standalone.py`.

The `animation_converter` package is the in-progress modular conversion core. It has its own 2.0 schema and command line, but it is not the public 6.0 Windows release.

## Modular Core Development

```powershell
python -m pip install -e ".[dev]"
python -m animation_converter inspect "path\to\pack.zip"
python -m animation_converter validate "path\to\pack.zip"
python -m animation_converter convert "path\to\pack.zip" --to ostim-sa --output "ConvertedPack" --package directory --behavior pandora
python -m animation_converter gui
```

Use `--strict`, `--best-effort`, or explicit `--salvage` to select conversion policy. Outputs contain `conversion-report.json`, `conversion-report.txt`, and `conversion-manifest.json`.

The OStim schema is pinned to VersuchDrei/OStimNG commit `ad138cbd3bee2a736a422389d715b36b792aa671`. See [format support](docs/FORMAT_SUPPORT.md), [schema notes](docs/OSTIM_SA_SCHEMA_NOTES.md), [design](docs/DESIGN.md), [losses](docs/CONVERSION_LOSSES.md), [behavior generators](docs/BEHAVIOR_GENERATORS.md), and the [migration guide](docs/MIGRATION_1_TO_2.md).

## Version 6.0 One-Click Usage

New to this? Start with `BEGINNER_GUIDE.md`. It explains which ZIP to choose, which ZIP to install, and why Pandora or Nemesis still needs to be run afterward.

Run the program, optionally enter the original mod author, leave or change `SFX fallback action` in the `Quick Build` section, leave `Add OStim menu entry` checked if you want the pack to show from OStim's main scene menu, leave `Human-only OStim output` checked for normal OStim Standalone builds, click `Build OStim Standalone ZIP From Archive`, and choose the downloaded OSex/OSex+/OpenSex/SexLab/FlowerGirls/legacy FNIS/OStim Standalone `.zip`, `.7z`, or `.rar` archive. The converter creates a ready-to-install `_OStimSA.zip` next to the archive.

For `.7z` or `.rar` archives, install 7-Zip normally. The converter finds common 7-Zip installs automatically, including Program Files, registry entries, Chocolatey, Scoop, portable copies beside the app, portable copies beside the selected archive, and `AAC_7ZIP` if you set it.

The generated OStim Standalone ZIP includes OStim scene JSON, copied HKX animations, an optional OStim main-menu entry, a pack-specific Pandora-compatible Nemesis/ATT patch for human events, FNIS-style creature lists and behavior hooks when required, converter metadata, and reports. Its archive root contains normal Skyrim Data folders such as `SKSE`, `meshes`, and `Nemesis_Engine`, so MO2 should not ask you to select a nested `Data` directory. Install it with a mod manager, then run Pandora.

Use `Diagnose Source Archive` before converting messy packs. Diagnosis reports the detected source type, known compatibility profile, source parser selection, archive branch layout, duplicate HKX handling, recommended output, recommended behavior tool, HKX counts, missing HKX references, creature/furniture requirements, and a one-line recommended action. If an adult source archive is blocked by the minor-coded content safety check, the JSON and TXT reports also include a redacted `compatibilityAutoFailCandidate` section with the archive hash and a ready-to-copy `compatibility_db.json` fail-entry template. The `Copy Diagnosis Summary` and `Copy Nexus Bug Report` buttons create paste-ready support text without exposing full local paths in normal mode.

Known-pack support is data-driven through `compatibility_db.json`. The app loads the bundled file and an optional user-editable copy from the app settings folder, then includes the database version and match result in reports. Compatibility profiles add warnings and recommendations only; they never bypass deployment verification.

To build a SexLab/SLAL pack instead, click `Build SexLab/SLAL ZIP From Archive`. The generated `_SexLab.zip` includes `SLAnims/json/<PackName>.json`, `SLAnims/source/<PackName>.txt`, renamed HKX files using SexLab stage names such as `<Animation>_A1_S1.hkx`, SLAL-style `FNIS_<PackName>_List.txt` files, source-provided `FNIS_*_Behavior.hkx` hooks plus a matching generated hook when available, SexLab sound metadata inferred from the converted OStim actions, and pack-level discovery tags. Install it with a mod manager, run GenerateFNISforUsers, Nemesis, or Pandora, then open SexLab Animation Loader in game and register or enable the generated pack. Pandora auto-discovers this output from its `FNIS_*_List.txt`; the converted SexLab pack is not expected to appear as a selectable Pandora checkbox. Confirm recognition in `Engine.log` by finding an `INFO : FNIS Mod ... : FNIS_<PackName>_List` line.

For SexLab P+, tick `SexLab P+ export` beside the SexLab button before building. The generated `_SexLabPPlus.zip` keeps the SLAL files for compatibility and also includes `SKSE/Sexlab/Registry/<PackName>.slr`, editable `SKSE/Sexlab/Registry/Source/<PackName>.slsb.json`, prefixed P+ FNIS behavior events that point at the packaged HKX files, and the same source-provided behavior graph hooks when available. Ordinary generated male positions use independent SLSB flags (`male: true`, `female: false`, `futa: false`); explicit source futa flags are preserved when importing SLSB data. The app detects the local SexLab Scene Builder repo before enabling this checkbox. If the checkbox is greyed out, place the `SexLab-Scene-Builder` repo beside this app/project or set `SEXLAB_SCENE_BUILDER_REPO` to the repo folder.

Leave `SexLab discovery tags` checked for SexLab exports unless you specifically want a minimal tag list. The converter adds `AdultAnimationConverter`, `Converted`, and a safe pack-name tag to SLAL JSON and `common_tags(...)` in the SLAL source text. SexLab P+ exports carry the same discovery information into SLSB stage tags so the pack is easier to search, filter, and identify in SexLab tooling.

## OStim Tools JSON Generator

OStim Tools JSON mode creates editable project/scene JSON files for users who want to finish a conversion manually in OStim Tools. It is not the same as a ready-to-install OStimSA ZIP. To install in game, build a verified OStimSA package after editing.

Click `Generate OStim Tools JSON` to parse a supported source archive through the normal detection and scene-conversion pipeline, then write `<SafePackName>_OStimToolsProject/`. The project contains `ostim_tools_project.json`, OStim Tools-clean editable scene JSON under `scenes/`, a project-local HKX animation asset bundle under `animations/` when source assets were available, an OStim Tools 3.3.1-style bridge config under `configs/bridge/ostimConfigs/`, optional inferred action scaffolds under `actions/`, optional `alignment/alignment.json`, reports, `AAC_README.txt`, and `AAC_MANIFEST.json`. The wrapper manifest uses AAC's own schema, `AdultAnimationPackConverter.OStimToolsProject`, because OStim Tools 3.3.1 provides scene/config schemas but not a standalone project-manifest schema.

Click `Generate Scene JSON Folder` or use the `scene-json-folder` CLI target if you only want `scenes/` plus reports and manifests. This lighter mode does not write a project manifest, action files, alignment files, behavior files, or a game-installable package.

Use `Validate OStim Tools Project` before importing edited JSON. Validation checks the manifest, listed scene files, duplicate scene IDs, OStim-style scene structure, animation strings, actors, actions, tags, furniture fields, declared OStim Tools bridge config files, declared alignment files, unsupported fields, and full local path leaks. It reports `PASS`, `PASS WITH WARNINGS`, or `FAIL`.

Use `Import OStim Tools Project` to load an AAC/OStim Tools project, a direct `ostim_tools_project.json`, or a folder of scene JSON files back into the converter's normal scene model. Full AAC project exports can carry HKX assets for a later ZIP build; plain scene folders may not. Imported projects are still not installable by themselves: behavior registration and deployable packaging belong to the normal OStimSA ZIP build and verification path.

Pandora reads the generated `Nemesis_Engine/mod/<unique code>/` ATT patch and the matching `ATT_*_animlist.txt`. Creature builds additionally use `FNIS_*_List.txt` and matching creature behavior hooks. The converter gives each pack a unique module/list code so two converted packs can be enabled together without sharing `DefaultMale.txt` or root AnimData files.

When `Add OStim menu entry` is enabled, the converter adds one small menu hub scene per supported actor setup and links it from OStim's built-in idle scenes, such as MF, MM, FF, and group starts. Small packs link directly from that hub. Large packs are split into generated categories such as All, Oral, Vaginal, Anal, Furniture, Group, and Other, with a Creature category only when creature output is intentionally included. Numbered pages are used when a category is too large. This avoids dumping hundreds of converted scenes into one brittle top menu while still giving the pack a visible in-game doorway. Linked converted scenes also get a generated return navigation back to the generated pack menu node that opened them, and related scenes get generated Previous/Next shortcuts when the converter can safely group them by actor setup, action, furniture, and creature race. Furniture-bound scenes still start from matching furniture, and creature scenes still need a creature-capable OStim setup.

The converter also writes `SKSE/Plugins/OStim/converter_metadata/<PackName>/metadata.json`. This manifest records the pack name, author, generated paths, scene/action index, and deployment check counts. The Verify button reads it back and fails stale metadata if the ZIP no longer matches what was generated.

Furniture-aware scenes are marked with OStim furniture metadata when the source clearly points at a built-in furniture type such as `bed`, `chair`, `bench`, `table`, `alchemytable`, or `enchantingtable`. These scenes are counted separately as furniture-bound scenes and should be started from matching furniture in OStim, not from the normal free-space scene list. OStim packs that already include custom furniture type JSON under `Data/SKSE/Plugins/OStim/furniture types/` have those files copied into the converted ZIP. SexLab spawned-prop and anim-object scenes such as toy chairs, pillories, crosses, tilted wheels, glory holes, or other pack-specific objects are kept as normal selectable scenes unless they safely map to a real OStim furniture type, so they do not disappear from the in-game menu.

`Human-only OStim output` is enabled by default for OStim Standalone builds. For mixed SexLab/SLAL packs, the converter detects creature actor metadata and non-character HKX roots such as `horse`, `canine`, `draugr`, or `werewolfbeast`, skips creature and mixed human/creature scenes, removes their behavior/Pandora registrations, and reports the skipped scene, HKX, root, and event counts. This lets normal OStim users build the human part of a mixed pack without needing a creature runtime. Build Recommended Package displays known-pack advice but uses the checkbox value you explicitly selected.

Advanced users can disable `Human-only OStim output` to include creature scenes. In that mode the `OCreatures-compatible creature output` adapter is enabled by default. The adapter preserves creature actor roots, writes FNIS-style lists and matching behavior hooks for each creature root, generates OCreatures-style `OCr<CreatureRoot>` menu/index entries, and records the SLAL-to-OCreatures actor-stage mapping used by working reference tools: `_A1_ -> _S#_1`, `_A2_ -> _S#_0`, and `_A3_ -> _S#_2`. Creature scenes still need a working in-game creature stack: OCreatures or another OStim creature extension, Creature Framework, matching creature assets such as More Nasty Critters or the pack's required assets, and creature-capable behavior generation with Pandora or the behavior tool required by your setup.

The OCreatures report section shows whether the adapter was enabled, which menu/index files were generated, how many creature scenes are reachable from the OCreatures menu path, the actor mapping result, the HKX filename policy, and `-Tn` handling for FNIS-style creature rows. The converter verifies structure and registration only; it does not prove that OCreatures, Creature Framework, ABC/MNC-style assets, or Pandora output are installed in the user's game.

To help testers check their local setup, run `python Osex-to-OStim-Standalone.py --check-creature-runtime "<Skyrim Data or enabled mods folder>" --creature-runtime-zip "<converted OStimSA zip>"`. The checker scans for OStim, OStim creature extension, Creature Framework, creature asset, behavior-generator, and expected actor-root markers, then writes `*_creature_runtime_check.json` and `.txt`. It is a marker scan, not an in-game playback guarantee. If MO2 or Vortex keeps runtime files outside the scanned folder, absent framework markers produce `INCONCLUSIVE`, not a false claim that the mods are uninstalled; genuinely absent expected actor-root behavior files still fail.

Converted OStim scenes use OStim action metadata for sounds. When the converter can infer a normal OStim action such as `handjob`, `blowjob`, `footjob`, `boobjob`, `kissing`, or `vaginalsex`, OStim's built-in action sounds and moans are used. When old XML/JSON/SLAL data has no reliable action type, the converter uses the `SFX fallback action` field. The default is `ostimconvertermoan`, which adds `SKSE/Plugins/OStim/actions/ostimconvertermoan.json`, a small fallback action that enables OStim voice/moan SFX without adding sex-specific actor requirements that could hide the scene. SexLab/SLAL exports use the same inferred actions and fallback choice to write SexLab `sound` and stage sound metadata such as `Sucking`, `Squishing`, or `NoSound`. Advanced users can change the field to a built-in OStim action such as `kissing`, a custom action name, or `none` to disable fallback actions.

SexLab does not use an OStim-style icon scene menu. For SexLab exports, the converter's discoverability support is tag-based: generated SLAL and P+ data includes `AdultAnimationConverter`, `Converted`, and the safe pack folder name by default. Use `--no-sexlab-discovery-tags` or uncheck `SexLab discovery tags` if you need the output to preserve only the original animation tags.

Pandora uses the generated Nemesis/ATT-compatible module for human events and FNIS-compatible lists for creature roots. A converted pack does not need a visible checkbox to be processed. If an older conversion shows a checkbox but Pandora creates no useful output, remove that conversion and rebuild the original archive with the latest converter.

OSex+ and OpenSex-style archives are supported when they still include OSA-style XML scene files and HKX animations. SexLab/SLAL packs are supported when they include `SLAnims/json/*.json` plus matching HKX animations; the converter normalizes SLAL actor/stage HKX names into OStim event names, infers OStim actions from SLAL tags for SFX, and can export native SexLab P+ `.slr` registry files when `SexLab P+ export` is checked. FlowerGirls packs are supported when they include `FNIS_FlowerGirlsSE_List.txt` or another FlowerGirls-marked FNIS list plus matching HKX animations. Older non-SLAL SexLab packs, including S.A.P.-style sources, are supported when a generic `FNIS_*_List.txt` contains recognizable paired `A1/A2/...` scene events. The converter rebuilds those actor groups and stages instead of importing every HKX as an unrelated one-actor scene, then links unknown-sex scenes from every compatible OStim actor-count menu. Because legacy FNIS lists do not carry the original SexLab role restrictions, actor roles and alignment still need an in-game check. OStim Standalone packs are supported when they include `SKSE/Plugins/OStim/scenes/*.json` plus HKX animations, with or without a top-level `Data` wrapper; the converter preserves modern scene fields such as speed metadata, actor metadata, navigation metadata, furniture, tags, custom action files, custom furniture type files, and source behavior engine patches while rebuilding behavior/Pandora files and reports. The scanner recognizes `0SA`, `OSA`, `OSA+`, `0Sex`, `OSex+`, OpenSex, `SLAnims/json`, FlowerGirls and grouped legacy FNIS lists, and OStim scene paths, and ignores installer XML such as `fomod/ModuleConfig.xml`.

Legacy FNIS and other supported sources are checked for Skyrim LE 32-bit Havok animation files automatically. When an LE HKX is found, the converter uses a configured or locally discovered `HavokBehaviorPostProcess.exe`, `hkx32to64.exe`, or `hkxcmd.exe` helper to create an SE/AE 64-bit working copy before packaging. Source archives are never modified, already compatible SE/AE HKX files are copied unchanged, and the build stops if no helper is available or a helper produces an invalid result. Select a helper under `Advanced Options`, pass `--legacy-hkx-converter`, or set `AAC_HKX_CONVERTER`. Conversion helpers are not bundled with AAC; suitable sources include the Skyrim Creation Kit and Cathedral Assets Optimizer.

FlowerGirls support converts animation content into OStim scenes. It does not convert FlowerGirls quests, dialogue, spells, scripts, or framework features. If a FlowerGirls archive only contains files such as `.esp`, `.pex`, `.psc`, or `.seq`, install it normally with FlowerGirls instead of converting it.

If a legacy OSex or OSex+ scene points at animation files that are not included in the source archive, the converter removes only those missing speeds. If a scene has no valid speeds left, it is dropped instead of being packaged as a broken scene. If a transition points at an old external OSex scene that is not in the archive, the converter removes that dead destination and keeps the animation as an internal reachable scene so the generated pack stays usable.

To check a converted package before deploying it, click `Verify Converted ZIP...` and choose any generated `.zip`. OStim ZIPs are checked for scene JSON, HKX files, selected behavior output mode, Pandora registration, named Pandora Module Diagnostics, legacy FNIS/ATT lists when present, generated/source behavior engine patch files when present, behavior entries that point at missing HKX files, OStim menu hub counts, generated return/Previous/Next navigation counts, duplicate generated menu destinations, converter metadata mismatches, invalid legacy JSON, old fake Nemesis checkbox metadata, packs with no deployable OStim scenes, human-only creature filtering leaks, missing custom furniture type files, broken scene links, and missing animation events. SexLab/SLAL and SexLab P+ ZIPs are checked for SLAL JSON/source files, SLSB source, compiled `.slr` registry files, HKX files, FNIS animation lists, source-provided behavior graph hooks, missing stage events, and FNIS lines that point at missing HKX files.

Verification now reports `PASS`, `PASS WITH WARNINGS`, or `FAIL`. `PASS WITH WARNINGS` means the ZIP is structurally usable, but the report has notes you should read before installing or testing, such as missing optional metadata, creature runtime requirements, or behavior-generator caveats. Errors still mean do not install that output yet.

For human-only OStim builds, a passing verification means skipped creature scenes were not left behind in OStim scene JSON, menu links, creature actor-root HKX files, or behavior registration. For OStim creature builds, a passing verification means the ZIP contains deployable OStim scene JSON, creature-root HKX files, matching ATT/FNIS registration, metadata, and an OCreatures menu entry path when the pack is creature-only. For SexLab creature exports, a passing verification means the SLAL/P+ data, creature-root HKX files, and FNIS lists agree. It does not install OCreatures, Creature Framework, SexLab creature requirements, creature assets, or creature behavior generation for you.

If the pack appears in Pandora but actors stay idle in game, reconvert the original archive with the latest converter, install the new `_OStimSA.zip`, rerun Pandora, and make sure Pandora's generated output mod is enabled/deployed in your mod manager. Old converted ZIPs that only had scene metadata or AnimSetData can appear in menus without registering playable motion events. If Pandora shows the checkbox but creates no output, click `Import Pandora Log`, choose Pandora's `Engine.log`, then use `Create Report Bundle` so the bundle includes public-safe Pandora log analysis beside the converter verification.

If conversion fails before a ZIP is created, the app now writes `<OutputName>_conversion_failed.json` and `<OutputName>_conversion_failed.txt` beside the intended output when possible. These reports include the converter version, source archive name, source archive hash, detected source type, fatal error, grouped warnings, and a recommended next action. Blocked adult source archives also include a redacted compatibility auto-fail candidate that can be sent to the maintainer for review without sharing the source pack. They are designed to be safe to paste into a Nexus bug report.

## Version 6.0 Changelog

- Added automatic Skyrim LE HKX detection and conversion to Skyrim SE/AE format when a supported local conversion helper is available.
- Added local discovery and explicit GUI/CLI selection for Creation Kit `HavokBehaviorPostProcess.exe`, Cathedral Assets Optimizer's `hkx32to64.exe`, and `hkxcmd.exe`.
- Added strict result validation: source archives remain unchanged, compatible SE/AE files pass through unchanged, and packaging fails if an unsupported 32-bit HKX remains.
- Added a transparent Nexus source release with no EXE, bundled runtime, DLL/PYD, batch/PowerShell launcher, or nested archive, plus payload and outer-ZIP SHA-256 hashes.
- Kept the hardened folder-style Windows EXE as a separate distribution for users who do not have Python installed.

## Version 5.9 Changelog

- Clarified and validated classic SexLab/SLAL registration in Pandora. SexLab exports are auto-discovered from `FNIS_*_List.txt`, do not create a selectable Pandora checkbox, and now report the exact `FNIS Mod` name to confirm in `Engine.log`.
- Added FNIS-list-aware Pandora log analysis so missing auto-discovery is reported separately from a missing OStim/Nemesis module.
- Hardened Windows packaging with a fresh pinned build environment, PyInstaller 6.22.2 one-folder output, no UPX, embedded version metadata, forbidden-dependency auditing, build provenance, and optional Authenticode signing.

## Version 5.8 Changelog

- Fixed a false adult-only safety block on OStim packs that bundle Nemesis `defaultmale` or `defaultfemale` baseline behavior catalogs. Vanilla behavior references in those inherited catalogs no longer block conversion, while archive paths, scene metadata, ATT registrations, and custom animation paths remain checked.
- Validated Sanguine Seductions - OStim Animation Pack 3.0 from the original Nexus archive. Diagnosis passes with source warnings, and SexLab export produces 17 animations with 34 registered HKX events and no missing deployment events.

## Version 5.7 Changelog

- Corrected the idle-actor/Pandora checkbox problem. New OStim Standalone packages use the Nemesis/ATT module format that Pandora actually assembles; the obsolete `Pandora_Engine/mod/<Pack>/animationdata` layout is rejected during verification.
- Validated the generated human ATT patch and creature FNIS lists with Pandora Behaviour Engine+ 4.4.0 in an isolated Skyrim test root.
- Fixed GUI worker access to Tk variables and the advanced-build `zip_out` error.
- Fixed the remaining Baka/HCOS `'str' object has no attribute 'get'` crash by teaching report and verification renderers to read privacy-safe truncated example containers as well as legacy list/string data.
- Completed a real BakaFactory SLAL Animation 78 human-only build: 58 deployable scenes, 672 registered HKX events, complete menu reachability, and zero missing deployment events.
- Fixed solid 7z archive ratio handling, multi-JSON SLAL pack naming, SLAL branch/actor-root selection, and accidental packaging of behavior graph HKX files as animation clips.
- Kept each generated behavior module/list pack-specific so multiple converted packs can be installed together without shared root AnimData collisions.
- Existing `_OStimSA.zip` files made with the checkbox-only Pandora layout must be rebuilt from the original source archive.
- Fixed a diagnosis/report popup that could show `'str' object has no attribute 'get'` when older or malformed compatibility/report fields were plain text instead of structured data.
- Hardened public report renderers, source-selection summaries, compatibility matching, packaged README/manifest generation, and Nexus-safe report bundles so malformed legacy JSON fields are reported as not recorded instead of crashing the GUI.
- Added safer JSON boundary validation for compatibility profiles, OStim Tools project manifests, SLAL/SLSB/OStim scene parsing, AAC manifests, and report sections so strings in object fields produce clear warnings or validation errors.
- Improved GUI internal-error handling: unexpected background-job crashes now write a traceback log with converter version, operation, selected archive name, and debug-mode state, plus Copy Error Summary and Open Error Log support actions.
- Added BakaFactory SLAL Animation 78 as a known mixed SLSB/SLAL profile.
- Mixed SLSB/SLAL diagnosis now explains when SLSB source JSON is selected over SLAL JSON, which branch root was selected, and which duplicate SLAL LE/SE branches were ignored.
- Diagnosis reports now split BakaFactory-style mixed pack guidance into normal Human-only OStim, creature-capable OStim, SexLab P+, and OStim Tools project paths.
- Duplicate HKX reporting now summarizes high duplicate counts with capped examples and a clear de-duplication policy instead of flooding reports with source paths.
- Added Pandora `Engine.log` analysis for checkbox/no-output cases, with a GUI import button, CLI support, public-safe reports, and report-bundle export.
- Improved behavior-output verification. The converter detects and blocks obsolete checkbox-only Pandora modules, validates ATT/FNIS rows against packaged HKX files, and gives clearer guidance when Pandora creates no generated output.
- Added Delta of Venus - Sex Fantasies for OStim NG 1.0 as a known OSA/OStim NG profile. Converter 5.4 or newer repairs its case-only XML/HKX speed mismatches and reports the one source scene skipped because its HKX files are absent from the tested archive.
- Source diagnosis now separates missing generated-output HKX events from missing skipped-source HKX events, so usable builds are no longer marked unsafe just because a broken source speed was dropped before packaging.
- Bumped the public release line to 5.7.
- Release ZIP and ZIP hash filenames now include the converter version automatically, such as `Adult Animation Converter 5.7.zip`.
- Added `BUILD_TESTER.ps1` for tester/beta packages named like `Adult Animation Converter 5.7 beta.zip`, with optional labels such as `beta1` and `beta2`.
- Build Recommended Package now follows known-pack behavior-tool recommendations and always emits the Pandora-compatible behavior patch for OStim Standalone.
- Tidied Simple Mode into clearer Recommendation, Project Tools, Next Steps, and Reports & Files sections so install steps no longer overlap the support buttons.
- Improved creature-only SLAL reports and verification metadata: final manifests record post-build verification, public JSON reports redact local paths, creature-runtime warnings are deduplicated, and scene counts distinguish JSON files from playable/menu helper scenes.
- Fixed false unsafe-path failures when 7-Zip reports the source RAR/7z archive header as a drive-qualified `Path = ...` line.
- Fixed duplicate generated OStim menu destinations in two-page category menus. Packs that previously failed verification with duplicate `Page_1`/`Page_2` Previous/Next links should be rebuilt from the original source archive with Version 5.7.
- Added a FlufyFox SLAL SE Creature 3.6 compatibility profile that identifies the pack as creature-only and warns that default human-only OStim output will skip every scene.
- Corrected the FlufyFox SLAL SE Creature 3.6 profile so the exact known creature-only archive recommends creature output instead of default human-only output.
- Added an explicit OCreatures-compatible creature output adapter. Creature SLAL builds now report OCreatures modes, menu/index files, actor-stage mapping, HKX filename policy, `-Tn` handling, and runtime requirements.
- Added `--compare-ocreatures-reference` for maintainer/debug comparison against known-good OCreatures output structures without comparing or redistributing HKX binary contents.
- Direct CLI archive builds and Build Recommended Package now honor exact-profile human-only/creature-output defaults automatically; explicit human-only builds still fail clearly if they would skip every scene.
- Added `--check-creature-runtime` to scan a tester's Data/mod staging folder for creature runtime markers and expected actor roots from a converted ZIP.
- Marked a known minor-coded Billyy Petite archive as unsupported so diagnosis reports are clearer and do not present it as a normal compatible Billyy-style pack.
- Added redacted compatibility auto-fail candidate blocks to blocked source diagnosis and conversion failure reports, including the archive hash and a ready-to-copy `compatibility_db.json` entry template for maintainer review.
- Added OStim Tools JSON/project support. Users can now generate editable OStim Tools-style project folders or plain scene JSON folders from supported source archives, validate those projects, and import them back into the converter for normal OStimSA packaging. Added reports, CLI options, GUI controls, and documentation for the new JSON generator workflow.
- Added an OStim Tools 3.3.1-style bridge config companion at `configs/bridge/ostimConfigs/<PackName>ScenesConfig.json` and stripped AAC-only metadata from exported scene JSON so editor-facing files stay closer to OStim Tools' official scene schema.
- Added Human-only OStim output for mixed SLAL packs. Creature/animal-root scenes are skipped by default for OStim Standalone builds, reports and Verify now show detected/skipped creature scenes, roots, HKX files, and behavior events, and advanced users can still disable the option for creature-capable setups.
- Improved large SLAL-to-OStim conversion handling for creature-heavy packs such as Billyy Petite. Added actor-root-aware event/HKX validation, fixed duplicate page navigation in generated OStim menus, corrected failed-build install guidance, and fixed missing diagnosis summaries in Nexus-safe report bundles.
- Fixed SLAL-to-OStim conversions where the generated pack menu could appear in OStim but contain no animations. The converter now tracks each SLAL source animation through parsing, scene writing, HKX/behavior matching, category assignment, and menu reachability; reports and Verify show written, skipped, linked, and unlinked scene visibility.
- Improved OStim menu fallback categories and pagination. Sparse/unknown SLAL tags now fall back to `Other`, generated SFX fallback actions no longer force scenes into a generic `Sex` category, and two-page categories no longer create duplicate Previous/Next destinations.
- Added Simple/Advanced modes, drag-and-drop archive selection, Build Recommended Package, clearer next-step guidance, copyable install steps, Nexus-safe report bundles, archive structure export, generated README/manifest files inside converted ZIPs, stale ZIP warnings, improved failed-build protection, tooltips, and a cleaner beginner workflow.
- Fixed large-pack OStim menu navigation. Generated OStim menu entries now use category/page hubs for large SLAL-style packs, deployment verification checks root-to-category/page scene reachability, and reports show linked/unlinked menu coverage plus missing or duplicate menu navigation issues.
- Fixed a source diagnosis false warning for Drago-style SLAL packs where the JSON and HKX filename only differ by letter casing, such as `Feetonface` versus `FeetonFace`.
- Improved OStim-to-SexLab export source quality. Fixed double-prefix risks with `anim_id_prefix`, removed duplicate actor declaration risks, cleaned common/per-animation tags, collapsed repeated animvars/SOS params, skipped OStim transition/internal scenes that are not valid SexLab animations, and improved FNIS/SLAnim verification and reports.
- Fixed UI scaling issue where the bottom of the app could be cut off on smaller screens or high-DPI displays. The main window is now scrollable and resizable.

## Third-Party License Notes

The converter bundles GPL-3.0 Animlist Transition Tool template resources to generate hidden Nemesis/ATT behavior patches for converted OStim Standalone packs. See `THIRD_PARTY_LICENSES.md` and `assets/animlist_transition_tool/LICENSE-GPL-3.0.txt`.

## Windows Defender Notes

The release ZIP is built as a folder-style Windows app instead of a single self-extracting EXE. Extract the whole `Adult Animation Converter` folder and keep `_internal` beside `Adult Animation Converter.exe`. The release script builds from a fresh pinned virtual environment, disables UPX, embeds Windows version metadata, rejects known unrelated runtime modules, and includes `BUILD_PROVENANCE.txt`.

Older one-file PyInstaller builds could trigger Defender heuristics because they unpacked themselves into a temporary folder at launch. The folder-style release avoids that behavior and includes `README_SECURITY.md`, `SHA256SUMS.txt`, and a separate versioned ZIP hash file such as `Adult Animation Converter 6.0.zip.sha256.txt`.

Do not tell users to turn off antivirus globally. `Windows protected your PC` is a SmartScreen reputation warning, not a Defender Antivirus detection. If Defender gives a named detection, keep the file quarantined, record the exact detection and SHA-256 hash, and submit an incorrectly detected clean release through Microsoft's file-submission portal. See `README_SECURITY.md` for the supported process and signing limitations.

For Nexus Mods, use `BUILD_NEXUS_SOURCE.ps1` to create the separate versioned `Nexus Source.zip`. That archive contains no EXE, bundled Python runtime, DLL/PYD, batch/PowerShell launcher, or nested archive; it includes inspectable Python source and payload hashes. Users must install Python 3.11 plus the two dependencies in `requirements-runtime.txt`. Nexus controls final moderation, so this format removes common structural quarantine triggers but cannot promise approval.

## Advanced usage

For tester-friendly command-line builds, use the short wrapper:

```powershell
python .\convert.py --input "path\to\OriginalAnimationPack.zip" --output "path\to\Converted_OStimSA.zip"
```

`--input` accepts `.zip`, `.7z`, `.rar`, or an extracted source folder. `--output` can be an exact `.zip` path or a folder where `<PackName>_OStimSA.zip` will be created. The wrapper generates Pandora-compatible Nemesis/ATT behavior registration by default; `--pandora` and `--nemesis` are compatibility aliases for that same output. It does not write into Skyrim's `Data` folder and does not modify the input archive or folder.

Useful tester flags:

```powershell
python .\convert.py --input "path\to\ExtractedPack" --output "path\to\Converted_OStimSA.zip" --overwrite --report-json --report-md
python .\convert.py --input "path\to\OriginalAnimationPack.7z" --output "path\to\Reports" --dry-run --report-json --report-md
python .\convert.py --input "path\to\Converted_OStimSA.zip" --validate-only --report-md
```

`--dry-run` writes source diagnosis reports without making an installable ZIP. `--validate-only` checks an existing converted ZIP, or diagnoses a source input if the input is not a converted ZIP. `--report-json` and `--report-md` may be used as simple flags or followed by a specific output path.

Run the Python file with no arguments to open the GUI:

```powershell
python .\Osex-to-Ostim-Standalone.py
```

Or run it from the command line:

```powershell
python .\Osex-to-Ostim-Standalone.py --mod-archive "path\to\OriginalAnimationPack.rar" --mod-author "Original Author" --sfx-fallback-action ostimconvertermoan
```

Human-only OStim output is enabled by default. To intentionally include creature OStim scenes from a creature-capable setup:

```powershell
python .\Osex-to-Ostim-Standalone.py --mod-archive "path\to\MixedCreaturePack.rar" --include-creature-ostim
```

Compare OCreatures structure against a known working output folder, ZIP, or path listing without comparing animation binaries:

```powershell
python .\Osex-to-Ostim-Standalone.py --compare-ocreatures-reference "path\to\AAC_Output.zip" "path\to\KnownGood_OCreatures_Output"
```

Build a SexLab/SLAL ZIP from the command line:

```powershell
python .\Osex-to-Ostim-Standalone.py --target sexlab --mod-archive "path\to\OriginalAnimationPack.rar" --mod-author "Original Author"
```

Disable generated SexLab discovery tags:

```powershell
python .\Osex-to-Ostim-Standalone.py --target sexlab --mod-archive "path\to\OriginalAnimationPack.rar" --no-sexlab-discovery-tags
```

Build a SexLab P+/SLSB ZIP from the command line:

```powershell
python .\Osex-to-Ostim-Standalone.py --target sexlabplus --mod-archive "path\to\OriginalAnimationPack.rar" --mod-author "Original Author"
```

Generate an editable OStim Tools/AAC JSON project:

```powershell
python .\Osex-to-Ostim-Standalone.py --mod-archive "path\to\OriginalAnimationPack.rar" --target ostimtools --ostim-tools-project-out "path\to\MyProject"
```

Generate only editable scene JSON files:

```powershell
python .\Osex-to-Ostim-Standalone.py --mod-archive "path\to\OriginalAnimationPack.rar" --target scene-json-folder --scene-json-folder-out "path\to\SceneJSON"
```

Validate or import an edited OStim Tools/AAC project:

```powershell
python .\Osex-to-Ostim-Standalone.py --validate-ostim-tools-project "path\to\MyProject\ostim_tools_project.json"
python .\Osex-to-Ostim-Standalone.py --import-ostim-tools-project "path\to\MyProject" --output-scenes "path\to\Data\SKSE\Plugins\OStim\scenes"
python .\Osex-to-Ostim-Standalone.py --import-ostim-tools-project "path\to\MyProject" --target ostim --zip-out "path\to\MyPack_OStimSA.zip"
```

Project JSON output is for authoring. Build and verify a normal OStimSA ZIP before installing anything in game. Full AAC project exports can carry the HKX assets needed for that ZIP; plain scene JSON folders may need matching animation assets before verification can pass.

Verify an already-converted ZIP:

```powershell
python .\Osex-to-Ostim-Standalone.py --verify-zip "path\to\ConvertedPack.zip"
```

```powershell
python .\Osex-to-Ostim-Standalone.py --input-xml "path\to\Data\Meshes\0SA\mod\0Sex\scene" --output-scenes "path\to\Data\SKSE\Plugins\OStim\scenes" --pack "MyConvertedPack" --mod-author "Original Author" --sfx-fallback-action kissing --zip-out "MyConvertedPack.zip"
```
