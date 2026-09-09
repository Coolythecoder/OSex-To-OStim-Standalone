# OStim Standalone Schema Notes

## Pinned source

The 2.0.0 adapter is pinned to VersuchDrei/OStimNG commit [`ad138cbd3bee2a736a422389d715b36b792aa671`](https://github.com/VersuchDrei/OStimNG/tree/ad138cbd3bee2a736a422389d715b36b792aa671), inspected on 2026-07-10.

Authoritative files inspected:

- `skse/src/Graph/GraphTable/GraphTableSetupNodes.cpp`
- `skse/src/Graph/Node.h`
- `skse/src/GameAPI/GamePosition.cpp`
- `skse/src/Graph/GraphTable/GraphTableSequences.cpp`
- `skse/src/Alignment/Alignments.cpp`
- `data/SKSE/Plugins/OStim/scenes README.txt`
- `data/SKSE/Plugins/OStim/sequences README.txt`
- representative JSON under `data/SKSE/Plugins/OStim/scenes`

The repository's local Ayasato Animations and Mike24 OStim Standalone archives were also inspected as third-party examples. They agree on filename IDs, `modpack`, event-based speeds, actor arrays, and `{x,y,z,r}` offsets. Loader source wins if an example disagrees.

## Confirmed rules

- A scene ID is the JSON filename without `.json`.
- Subdirectories do not namespace scene IDs. Duplicate filenames collide globally.
- Third-party folder and file names must not begin with the reserved `OStim` prefix.
- Canonical pack display key casing is `modpack`, not `modPack`.
- A speed names an animation event through `animation`; it does not contain an HKX file path.
- `playbackSpeed` and `displaySpeed` are optional numeric speed fields.
- `defaultSpeed` is a zero-based index into `speeds`.
- Actor `animationIndex` selects the actor suffix sent with the speed event.
- Actor bend aliases `sosBend` and `tngBend` are both consumed by the pinned loader.
- Current offsets contain `x`, `y`, `z`, and one rotational field, `r`.
- A scene with `destination` is a transition. Its ordinary `navigations` are ignored.
- Navigation metadata includes origin/destination, priority, description, icon, border, and `noWarnings`.
- Sequence IDs come from filenames. Sequence entries use `id` and optional `duration`.

## Canonical emitted fields

Scene fields include `name`, `modpack`, `length`, transition fields, `navigations`, `speeds`, `defaultSpeed`, `noRandomSelection`, `fadeOnEntry`, `furniture`, `offset`, `scaleOffsetWithFurniture`, `tags`, `autoTransitions`, `actors`, and `actions`.

Actor fields include `type`, `intendedSex`, bend, scale, scale height, animation index, expression fields, look fields, `noStrip`, `feetOnGround`, offset, requirements, tags, and auto transitions.

Action fields include `type`, `actor`, `target`, `performer`, `muted`, `doPeaks`, and `peaksAnnotated`.

Run `python -m animation_converter schema-dump ostim-sa` for the machine-readable field list and pinned commit.

## Alignment decision

`Data/SKSE/Plugins/OStim/alignment.json` is loaded and serialized by `Alignments.cpp` as runtime/user thread alignment state. It is keyed by thread, scene, and actor and contains `offsetX`, `offsetY`, `offsetZ`, `scale`, `rotation`, and `sosBend`.

Pack-authored static offsets are loader scene or actor fields. Therefore 2.0 implements outcome B from the task specification:

- source offsets are emitted in scene/actor JSON;
- the converter does not generate `alignment.json`;
- absent source offsets stay absent;
- explicit neutral zeros remain valid with provenance;
- all-zero values caused by a parse failure are blocking.
