# Conversion Losses

Every conversion is classified as exact, structurally valid with known losses, asset-only salvage, or unsupported. The manifest separates generated, discarded, unsupported, and inferred values.

## OSA / OSex to OStim SA

Representable fields include scene names and IDs, actor counts and indices, speed events, duration, direct navigation links, transition endpoints, named furniture, tags, actions, and x/y/z/r offsets.

Known limits:

- OSA UI hue, gaze, motif, drive, and theme elements are retained as unknown XML provenance unless a current OStim field has an explicit mapping.
- OSA `rx` and `ry` Euler rotations cannot be represented by OStim's single `r` value. They are retained in provenance and reported.
- Framework scripts, plugin records, quests, and menus are not scene metadata.
- Missing scene metadata is an error. HKX presence is not evidence for a complete scene.

## OStim SA to OSA / OSex

The exporter writes the representable scene subset and records unsupported values. In particular, current expression controls, `fadeOnEntry`, `scaleOffsetWithFurniture`, some auto-transition semantics, and OStim-only action behavior may not have a validated destination equivalent.

Strict mode fails when these values are present. Best-effort mode may write XML with a loss report. It never calls the result lossless.

## SLAL and Flower Girls

SLAL actor stages are represented as OStim speed variants only when every actor has a stage event. Registry behavior, sounds, anim objects, creature runtime dependencies, and framework state may require manual work.

Flower Girls conversion uses explicit FNIS rows only. Quests, dialogue, spells, scripts, and framework menus are not converted.

## Reverse conversion and manifests

When `conversion-manifest.json` is present, the service restores original source scene identifiers and preserved source-only provenance, including unknown XML retained by the source adapter. It does not fabricate fields that the manifest did not preserve. Strict reverse conversion fails on required unrecoverable fields; best effort records them.

## Salvage

Salvage mode creates visibly named `SALVAGED ASSET` placeholders and tags them `salvaged` and `incomplete`. A salvage output is never install ready and is intended only to organize assets while real metadata is recovered.
