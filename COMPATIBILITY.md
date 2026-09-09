# Compatibility

Adult Animation Converter supports structural conversion for OSex, OSex+, OpenSex/OSA, SexLab/SLAL, FlowerGirls FNIS animation lists, existing OStim Standalone packs, and older OStim JSON packs.

## Compatibility Database

The app loads optional known-pack profiles from:

- `compatibility_db.json` beside the EXE or script
- `%LOCALAPPDATA%\Adult Animation Converter\compatibility_db.json`
- a path set with `AAC_COMPATIBILITY_DB`

Invalid entries are ignored with warnings. The database cannot make a broken conversion pass verification; it only adds recommendations, warnings, and support notes.

Each entry can include:

- `packDisplayName`
- `detectedSafePackName`
- `sourceFramework`
- `sourceTypeCodes`
- `preferredSourceParser`
- `fallbackSourceParsers`
- `sourceBranchPatterns`
- `ignoredBranchPatterns`
- `expectedMixedSourceTypes`
- `expectedDuplicateHkxHighCount`
- `archiveFingerprints`
- `namePatterns`
- `filePatterns`
- `folderPatterns`
- `supportedOutputTypes`
- `recommendedOutputType`
- `recommendedBehaviorTool`
- `defaultHumanOnlyOStim`
- `creatureOutputRecommended`
- `creatureRuntimeRequired`
- `creatureNotes`
- `ocreaturesOutputRecommended`
- `ocreaturesMenuIntegrationRequired`
- `ocreaturesActorMapping`
- `ocreaturesPreserveHkxNames`
- `ocreaturesHkxFilenamePolicy`
- `ocreaturesRequiresTnFlag`
- `ostimToolsSupported`
- `recommendedOStimToolsMode`
- `ostimToolsNotes`
- `templateRecommended`
- `sourceMetadataRecommended`
- `alignmentRecommended`
- `recommendedNormalUserWorkflow`
- `recommendedCreatureUserWorkflow`
- `recommendedPPlusWorkflow`
- `recommendedOstimToolsWorkflow`
- `duplicateHkxPolicy`
- `customFurniturePolicy`
- `status`
- `knownWarnings`
- `knownIssues`
- `minimumConverterVersion`
- `notes`

When an adult source archive is blocked by the minor-coded content safety check, source diagnosis and conversion failure reports include `compatibilityAutoFailCandidate`. That block contains the archive hash, redacted hit counts, and a `compatibilityDbEntryTemplate` object. Maintainers can copy the template into `compatibility_db.json` after confirming the report came from the original source archive. Prefer full SHA256 fingerprints; partial hashes are only a fallback for very large archives.

## Status Meanings

- `Working`: expected to convert when the source archive is complete.
- `Working with warnings`: expected to convert, but read the limitations.
- `Partially supported`: some scenes or metadata may need manual checking.
- `Needs creature runtime`: creature assets/frameworks are required in game.
- `Needs furniture handling`: furniture/object scenes need matching runtime furniture support.
- `Unsupported archive type`: install normally; do not convert.
- `Known broken source archive`: use a different source archive or updated pack.
- `Needs latest converter`: reconvert with the current release.
- `Unsupported`: do not build, install, export, or recommend this archive through Adult Animation Converter.

## SexLab Export Style

SexLab/SLAL and SexLab P+/SLSB exports use SexLab-native source style by default. For normal MF animations, actor1 is usually Female and actor2 is usually Male. Converter/discovery tags are only added when `SexLab discovery tags` is enabled. Stage timers and repeated stage sounds are omitted unless the converter detects a real stage-specific value, and actor-specific metadata is written through actor-specific stage params where supported.

Generated `SLAnims/source` files also use build-tool-friendly prefixes. When the final SLAL JSON IDs share a prefix, the source emits `anim_id_prefix(...)` and strips that prefix from each `Animation(id=...)` so rebuilding with SLAnimGenerate does not double-prefix events. Pack-level discovery tags stay in `common_tags(...)`, repeated SOS/animvar actor stage params are collapsed, and invalid OStim menu/transition/internal scenes are skipped instead of becoming dead SexLab entries.

## Human-only OStim Defaults

Profiles can set `defaultHumanOnlyOStim` to choose whether OStim Standalone builds should skip creature and mixed human/creature scenes by default. This is recommended for most mixed SLAL packs because normal OStim users often do not have a creature-capable runtime.

Set `creatureOutputRecommended` only when creature OStim output is known to be the better default for that profile. Set `creatureRuntimeRequired` and `creatureNotes` to explain which creature extension, assets, or behavior generation users need if they disable human-only output.

For creature-only SLAL packs, set `defaultHumanOnlyOStim` to `false`, `creatureOutputRecommended` to `true`, and `status` to `needs creature runtime`. Add expected counts such as `expectedOStimSceneJsonFiles`, `expectedDeployableCreatureScenes`, `expectedHkxPackaged`, `expectedBehaviorEvents`, and `expectedCreatureRoots` when a tester provides a verified report. Exact archive fingerprints should take precedence over broad creature-root or K4-style patterns.

Reports for these packs should remain `PASS WITH WARNINGS` when structurally valid. The warning is the runtime requirement: users still need OCreatures or another OStim creature extension, Creature Framework, matching creature assets, and creature-capable Pandora behavior generation.

Creature-only OStim packs must also have `OCreatures Output` and `OCreatures Menu Integration` sections. A valid FlufyFox-style build should report an OCreatures menu entry under `SKSE/Plugins/OStim/scenes/OCreatures/OCr<CreatureRoot>/` and show all retained creature scenes reachable from that OCreatures path. Normal OStim human menu hub reachability alone is not sufficient for creature-only packs.

## Mixed SLSB/SLAL Branch Archives

Some modern SexLab packs ship several top-level branches in one archive, such as `SLSB SE`, `SLAL SE`, and `SLAL LE`. When SLSB source JSON and SLAL JSON are both present, the converter should prefer SLSB source JSON because it carries richer SexLab P+/SLSB metadata. Use profile fields such as `preferredSourceParser`, `fallbackSourceParsers`, `sourceBranchPatterns`, and `ignoredBranchPatterns` to make that choice visible in diagnosis reports.

For BakaFactory-style mixed archives, duplicate HKX counts can be very high because the same animation files appear in multiple edition/source branches. Set `expectedDuplicateHkxHighCount` and `duplicateHkxPolicy` so reports explain that duplicate HKX package paths are expected, capped, and de-duplicated before behavior registration. The report should still fail if referenced HKX events are missing or if the selected parser cannot produce deployable scenes.

Profiles for mixed human/creature SLSB packs should split recommendations by user path: normal OStim users keep `Human-only OStim output` enabled, creature-capable users disable it only with the full creature runtime, SexLab P+ users can preserve the SLSB structure, and OStim Tools users should treat project output as editable authoring data rather than an installable mod.

## OCreatures Profile Fields

Use `ocreaturesOutputRecommended` when the known pack should use the OCreatures adapter for OStim output. Use `ocreaturesMenuIntegrationRequired` when creature scenes should fail verification if no OCreatures-facing menu/index entry is generated.

The current SLAL creature mapping rule is:

- `ocreaturesActorMapping`: `slal_creature_a1_to_s1_actor1_a2_to_s1_actor0_a3_plus_identity`
- SLAL `_A1_` maps to OStim/OCreatures `_S#_1`
- SLAL `_A2_` maps to OStim/OCreatures `_S#_0`
- SLAL `_A3_` maps to OStim/OCreatures `_S#_2`
- Later SLAL actor slots continue as source slot minus one, such as `_A4_ -> _S#_3` and `_A5_ -> _S#_4`

Set `ocreaturesRequiresTnFlag` to `true` for FNIS-style creature list compatibility. Current creature output writes `-Tn` rows in actor-root FNIS lists consumed by Pandora. Human events use the generated ATT/Nemesis-compatible patch.

`ocreaturesPreserveHkxNames` should describe the selected AAC packaging policy, not a third-party reference tool's implementation. AAC's full OStimSA ZIP currently packages HKX files with deterministic generated event names so scene JSON and behavior registration share one contract. The reference FNIS-list converters may preserve source HKX filenames because they rewrite list rows rather than rebuilding the full OStim scene package. Record that distinction in `ocreaturesHkxFilenamePolicy` and the profile notes when it matters.

## OStim Tools JSON Recommendations

Profiles can set `ostimToolsSupported` to `true` or `false`. Diagnosis reports this as `OStim Tools JSON project: supported`, `supported with warnings`, or `not recommended`.

Use `recommendedOStimToolsMode` when a known pack works better as a full `project`, a lighter `scene-json-folder`, or should be treated as `unsupported`. Use `ostimToolsNotes` for pack-specific manual-editing caveats.

Set `templateRecommended`, `sourceMetadataRecommended`, and `alignmentRecommended` when a profile should encourage a template file, public-safe source metadata, or alignment export. These fields are advisory only; they do not bypass project validation or final OStimSA ZIP verification.

## Public Reports

Normal reports use archive names and archive-relative paths so they are safe to paste publicly. Debug mode can include full local paths and more internal detail.
