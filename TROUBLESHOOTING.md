# Troubleshooting

This app creates a new mod ZIP. It does not edit Skyrim directly. Install the generated ZIP with MO2 or Vortex, then rerun Pandora or Nemesis before testing in game.

## PASS, PASS WITH WARNINGS, FAIL

- `PASS`: the generated ZIP has the required scene/animation metadata, selected behavior registration, and packaged HKX files.
- `PASS WITH WARNINGS`: the ZIP is structurally usable, but the report found limitations such as skipped creature scenes, included creature scenes, furniture scenes, or compatibility notes.
- `FAIL`: do not install the ZIP. Fix the source archive issue or attach the report in a Nexus support post.

## Scenes Appear In Search But Actors Idle

This usually means the scene JSON exists, but the playback chain is broken. Check the human report for:

- scene events missing behavior registration
- behavior events pointing to missing HKX
- missing HKX references from SLAL/SLSB metadata
- stale converted ZIPs made by an older converter

Reconvert from the original source archive, install only the new ZIP, and rerun Pandora or Nemesis.

## Internal Error: `'str' object has no attribute 'get'`

This means the app received JSON/config/report/profile data in a shape it did not expect, usually a plain string where an object was expected. Current builds validate the common compatibility database, report, manifest, OStim Tools project, and source JSON boundaries so malformed data should produce a clear validation warning or error instead of only a raw AttributeError.

If an unexpected internal bug still reaches the GUI, the app writes a traceback log under the app settings folder and the dialog shows that log path, converter version, current operation, selected archive name, and debug-mode state. Use `Copy Error Summary` or `Open Error Log` in the Reports & Files section, then attach the error log and a Nexus-safe report bundle when reporting the issue.

Public-safe reports redact full local paths unless Debug mode is enabled. Include the source archive name and converter version in support posts.

## OStim Tools JSON Project Output

`Generate OStim Tools JSON` and `Generate Scene JSON Folder` create authoring files, not deployable mods. Do not install those folders with MO2 or Vortex. Open or edit the JSON first, then import the project back into the converter and build a verified `_OStimSA.zip`.

Full OStim Tools project exports also include an OStim Tools 3.3.1-style bridge config under `configs/bridge/ostimConfigs/`. Scene JSON files are the main editable scene files; the bridge config is a companion for OStim Tools-style metadata workflows.

If validation fails, open `reports/ostim_tools_project_validation_README.txt`. Common causes are missing files listed in `ostim_tools_project.json`, duplicate scene IDs, malformed scene JSON, actions that point at actors that do not exist, declared bridge config or alignment files that were moved or deleted, or full local paths accidentally pasted into public project JSON.

If import succeeds but a later OStimSA package fails verification, treat that as a normal packaging failure. Full AAC project exports can include a project-local HKX asset bundle, but edited projects or plain scene folders may not. The final installable ZIP still needs matching animation assets, behavior registration, and deployment verification.

## Duplicate Page_1/Page_2 Menu Destinations

If Verify fails with duplicate generated OStim menu destinations such as `Page_1` or `Page_2`, rebuild the original source archive with Version 5 or newer. Older generated ZIPs could write two different Previous/Next labels to the same page destination in two-page categories; Version 5 deduplicates those destinations before packaging.

## Pandora vs Nemesis

Pandora is recommended for most converted OStim Standalone packages. Fresh builds generate a pack-specific `Nemesis_Engine/mod/<UniqueCode>/` ATT patch and a matching `ATT_*_animlist.txt`. Pandora supports this format, and a visible converted-pack checkbox is not required.

Classic SexLab/SLAL exports use a different registration path. Pandora automatically scans their `meshes/actors/*/animations/<Pack>/FNIS_<Pack>_List.txt` files, so a converted `_SexLab.zip` is not supposed to add a Pandora checkbox. After running Pandora, open `Engine.log` and look for `INFO : FNIS Mod ... : FNIS_<Pack>_List`. If that exact generated list name is present, Pandora recognized the pack. The next step is to register or enable it in SexLab Animation Loader in game.

If the expected `FNIS Mod` line is absent, make sure the generated `_SexLab.zip` is enabled in the same MO2/Vortex profile Pandora scans and that the list is visible in the virtual `Data` folder. For a stock-game/Wabbajack setup or more than one Skyrim installation, correct Pandora's game path or launch it with `--tesv` as documented by Pandora. Looking for a checkbox will not diagnose this type of export.

Creature builds also generate actor-root `FNIS_*_List.txt` files and matching `FNIS_*_Behavior.hkx` hooks. Run Pandora after installing the converted ZIP and make sure its generated output is enabled in the same mod-manager profile.

If Pandora shows an old converted-pack checkbox but creates no output files, that ZIP may use the obsolete `Pandora_Engine/mod/<Pack>/info.xml` plus animationdata/animationsetdata layout. Current verification fails that layout because Pandora displays it but does not assemble those folders. Rebuild from the original archive, then use `Import Pandora Log` and create a report bundle if the new output still fails.

Check Pandora's settings too. If its output path is the live Skyrim `Data` folder, change it to a dedicated `Pandora Output` folder or pass `-o "path"` from your mod manager. That makes generated files visible, removable, and easier to attach to support reports.

Imported Pandora logs are checked against files beside that specific `Engine.log`. The analyzer recognizes current Pandora 4.4 `OutputAnimData` and `OutputAnimSetData` merge messages and only applies saved output-path warnings when the saved folder matches the imported log. This avoids treating settings from another Pandora copy as evidence about the tested run.

## A Large Archive Looks Frozen

Archive extraction and packaging run on a background worker. Version 5.7 and newer write a `Still working...` heartbeat every 15 seconds while a long operation is active. Solid 7z archives and packs with thousands of HKX files can take several minutes; wait for a success/error result rather than starting the same build again.

## SexLab Export Looks Different From The Source

SexLab export uses SexLab-native source style by default. For normal MF animations, actor1 is usually Female and actor2 is usually Male. Converter/discovery tags are only added when `SexLab discovery tags` is enabled. Repeated stage sounds and default timers are omitted unless a stage really needs its own value.

In SexLab P+/SLSB output, `male`, `female`, and `futa` are independent flags. Generated ordinary male positions use `male: true`, `female: false`, `futa: false`; an explicit futa flag is preserved only when it was present in the imported SLSB source. Rebuild older P+ output if every male actor was marked as futa.

If a SexLab ZIP verifies with source/FNIS warnings, open the `SexLab Source/FNIS Quality` section in the report. Double-prefix risks, duplicate actor declarations, repeated common tags, invalid animvars, duplicate FNIS events, and bad FNIS `s`/`+` ordering should be fixed by reconverting with the latest app. Skipped invalid scenes are usually OStim menus, transitions, idles, or one-HKX internal scenes that are not playable SexLab animations.

## Missing HKX Files

If diagnosis reports missing HKX references, the selected archive is probably incomplete. Use the full original animation archive, not an add-on, patch, installer-only archive, ESP-only archive, or script-only archive.

## Skyrim LE 32-bit HKX Files

AAC identifies Skyrim LE and SE/AE Havok packfiles from their binary headers. LE animation files are converted automatically on isolated working copies when one of these supported local helpers is available: Creation Kit `HavokBehaviorPostProcess.exe`, Cathedral Assets Optimizer's `hkx32to64.exe`, or `hkxcmd.exe`. Compatible SE/AE files pass through unchanged, and the original archive is never edited.

Choose a helper in `Advanced Options`, use `--legacy-hkx-converter <path>`, or set `AAC_HKX_CONVERTER`. AAC also searches its application folder, `PATH`, common Cathedral Assets Optimizer locations, and installed Skyrim Creation Kit tool folders. It deliberately does not execute converter programs found inside the selected source archive.

AAC does not bundle these third-party/Havok tools. If a diagnosis finds LE HKX but no helper, install a trusted converter from its official source and select its executable. The build is stopped instead of packaging incompatible animations. If conversion runs but its output is still 32-bit, unreadable, or missing, attach the source diagnosis and failure report when asking for help.

Open `Legacy HKX Conversion` in the report to see how many files were inspected, already compatible, converted, unknown, or blocked. `Verify Converted ZIP` independently rejects any confirmed LE or unsupported 32-bit HKX left in the final package.

## Mixed SLSB/SLAL Archives And Duplicate HKX

Some packs include multiple source branches in one archive, for example `SLSB SE`, `SLAL SE`, and `SLAL LE`. When the diagnosis has both `SexLab P+/SLSB source JSON` and `SexLab/SLAL animation JSON`, the converter should normally select SLSB source JSON and report that SLAL JSON was detected but not used.

For known mixed archives such as BakaFactory SLAL Animation 78, the report splits the recommendation into normal OStim, creature-capable OStim, SexLab P+, and OStim Tools paths. Normal OStim users should follow `Normal User Recommendation` and keep `Human-only OStim output` enabled. Creature-capable users should only include creature scenes after confirming their creature runtime and behavior output.

Large duplicate HKX counts can be expected when the same event exists in several branches. Check `Duplicate HKX Handling`: duplicate package paths should be de-duplicated, example rows should be capped, and `Created behavior conflicts` should be `no`. If missing HKX count is nonzero or the selected content root is wrong, attach the diagnosis report.

## Human-only OStim Output And Creature Scenes

`Human-only OStim output` is recommended for most users. It skips creature/animal-root animations from mixed SLAL packs so you do not need OStim creature support. Disable it only if you have the required creature runtime, creature assets, and behavior generation.

Build Recommended Package uses the checkbox value currently shown in the GUI. A known compatibility profile can recommend human-only or creature output, but it does not override your explicit selection.

If the report says creature scenes were skipped, that is expected for normal OStim Standalone output. The retained human scenes should still be installed, registered, and tested normally.

If you disabled `Human-only OStim output`, creature scenes can be packaged and reported, but the converter cannot verify your installed creature runtime. Creature playback still needs the matching creature framework/assets and creature-capable behavior generation.

## Creature-only SLAL Packs

Some SLAL packs are creature-only or creature-focused. These can be structurally converted when the archive contains valid JSON, HKX files, menu links, and ATT/FNIS behavior registration, but they still require OStim creature runtime support in game.

For known creature-only profiles, the report may recommend OStim Standalone with creature output enabled. `PASS WITH WARNINGS` means the ZIP structure passed verification; it does not mean OCreatures or another OStim creature extension, Creature Framework, matching creature assets, and creature-capable Pandora output are installed in your load order.

If `Human-only OStim output` is enabled for a creature-only pack, the converter may skip every playable scene and refuse to create an empty installable menu. Use the report's creature roots, retained/skipped scene counts, and runtime requirements to decide whether the pack is usable for your setup.

For OCreatures builds, quote the `OCreatures Validation` section when reporting bugs. It collects the important checks in one place: HKX references, duplicate event IDs after sanitisation, actor-slot mapping, selected behavior path, duplicate behavior backend registration, OCreatures menu coverage, creature `-Tn` handling for FNIS-style rows, and whether a local creature runtime scan was attached.

Use the runtime marker check when a tester says the converted ZIP installs but actors still idle or no creature scene starts:

```powershell
python Osex-to-OStim-Standalone.py --check-creature-runtime "E:\Path\To\Skyrim\Data-or-enabled-mods" --creature-runtime-zip "E:\Path\To\Converted_OStimSA.zip"
```

Scan the actual enabled mods/output folder, not only the generated converter ZIP. The report is a heuristic marker scan. MO2 and Vortex can virtualize enabled files outside the physical Skyrim `Data` folder, so missing OCreatures, Creature Framework, or asset markers are reported as `INCONCLUSIVE` rather than proof that those mods are uninstalled. Scan the relevant staging/mod folders or run the check through the same mod-manager environment. Missing expected creature actor-root behavior files is still a real failure.

When the report says `Behavior output mode: pandora_compatible+creature_fnis`, install the generated ZIP and run Pandora with the required creature runtime enabled.

## Creature Pack Does Not Appear In OCreatures

For creature-focused OStim output, a normal OStim menu hub is not enough. The generated ZIP must also include an OCreatures menu/index path under `SKSE/Plugins/OStim/scenes/OCreatures/OCr<CreatureRoot>/`.

Open the deploy verification report and check:

- `OCreatures Output`
- `OCreatures Menu Integration`
- `OCreatures menu entry generated`
- `Creature scenes reachable from OCreatures menu`
- `Actor mapping result`
- `-Tn flag handling`

If creature scenes exist but the OCreatures menu entry is missing, verification should fail. If the entry exists and all scenes are reachable, the next checks are runtime-side: OCreatures or another OStim creature extension, Creature Framework, matching creature assets, installed converted ZIP, and enabled/deployed Pandora output.

Maintainers can compare an AAC output against a known working OCreatures output folder, ZIP, or path listing without comparing HKX binary contents:

```powershell
python Osex-to-OStim-Standalone.py --compare-ocreatures-reference "E:\Path\To\AAC_Output.zip" "E:\Path\To\KnownGood_OCreatures_Output"
```

## Furniture Scenes

Furniture or object scenes may only appear when started from matching in-game furniture or when the target runtime supports the furniture metadata. Use diagnosis and reports to confirm the converter detected furniture scenes.

## FlowerGirls

FlowerGirls quest, script, and plugin mods are not converted. Only animation content with FNIS animation lists and matching HKX files can be converted.

## What To Attach

For Nexus support posts, attach or paste:

- the generated `conversion_report_README.txt`
- the deploy verification text report
- `reports/ostim_tools_project_report_README.txt` or `reports/ostim_tools_project_validation_README.txt` for OStim Tools JSON workflows
- the OCreatures reference comparison report, if one was run
- the copied Nexus bug report
- the copied diagnosis summary

Do not post screenshots alone; they rarely include enough technical detail.
