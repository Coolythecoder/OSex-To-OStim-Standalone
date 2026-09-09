# Converter 2.0 Design

## Audit of the previous implementation

The repository contains two relevant historical states:

1. The tagged 1.2.1 script was a small one-file prototype. It mixed Tkinter, XML parsing, conversion, file copying, behavior text generation, ZIP creation, and command-line handling.
2. The working tree also contains a substantially expanded 5.7 compatibility implementation in `Osex-to-OStim-Standalone.py`. It added many useful source handlers and deployment checks, but still places almost all behavior in one module.

The 1.x prototype behaved as follows:

- Input detection relied primarily on archive extension, directory names, and tolerant XML tag/attribute searches.
- ZIP extraction used the standard library; 7z was delegated to an external executable.
- XML parsing searched broad sets of scene, stage, animation, actor, and file aliases. Its final fallback searched arbitrary XML attributes for `.hkx` values.
- JSON generation used a converter-owned `id` / `pack` / `poses` / `clips` shape. That shape is not consumed by the current OStim loader.
- HKX files were copied and referenced directly from clips, conflating binary assets with logical animation events.
- Alignment generation defaulted absent values to zero and could write an all-zero alignment file without proving those values came from the source.
- Behavior generation treated path/event text as sufficient registration without a separately validated event-to-actor-to-HKX model.
- The GUI called conversion functions directly and conversion code knew about GUI state.
- Packaging wrote generated JSON, copied HKX files, behavior text, and alignment data into a ZIP without an independent install-readiness decision.

The expanded 5.7 module corrected several output keys and added valuable source support, reports, archive checks, behavior tooling, and a large regression suite. Those functions remain importable for compatibility, but the 2.0 entry points do not call that conversion pipeline.

## Encoded regressions

The 2.0 tests make the original failure mode explicit:

- `tests/test_regressions.py` rejects the old `id` / `pack` / `poses` / `clips` object as current OStim scene data.
- Normal conversion with zero parsed scenes fails.
- Normal conversion never falls back to one scene per HKX.
- Only explicit salvage mode creates visibly incomplete asset placeholders.
- A neutral zero offset explicitly present in source is accepted.
- An all-zero offset result carrying parse-failure or missing-source provenance is blocking.
- A menu/scene graph must contain resolvable in-pack destinations and an entry point.
- Event mapping, asset mapping, behavior registration, and scene JSON are validated as separate layers.

## Architecture

The root package is used instead of a `src/` directory so `python -m animation_converter` works directly from a source checkout without installation. The ownership boundaries match the requested architecture:

- `models.py`: typed neutral IR, provenance, stable IDs, manifests, losses, and semantic normalization.
- `diagnostics.py`: INFO, WARNING, ERROR, and FATAL diagnostics.
- `adapters/`: source and target format boundaries.
- `registry.py` and `detection.py`: capability registration and evidence-based format selection.
- `graph.py`: reference remapping and graph integrity.
- `archive.py`: bounded secure source staging.
- `assets.py`: HKX discovery, event mapping, byte copying, and hash verification.
- `validation.py`: cross-format semantics and install-readiness inputs.
- `behavior/`: behavior-writer plugins and explicit external process support.
- `reporting.py`: deterministic manifest and report serialization.
- `packaging.py`: validated atomic directory finalization and deterministic ZIP creation.
- `service.py`: the application transaction shared by CLI and GUI.
- `cli.py` and `gui.py`: presentation only.

The conversion flow is:

1. Stage the user-supplied source without modifying it.
2. Detect format from path layout and metadata signatures.
3. Parse with an explicit adapter into the neutral IR.
4. Restore source IDs and preserved source-only provenance from a conversion manifest when reversing a prior conversion.
5. Validate source semantics and graph integrity.
6. Assign safe target IDs and event names, then emit target metadata.
7. Copy HKX files byte-for-byte and verify SHA-256 hashes.
8. Write a behavior registration manifest or validated generator output.
9. Validate target semantics and calculate install readiness from named checks.
10. Write reports and atomically finalize a directory or deterministic ZIP.

## Conversion modes

- Normal allows warnings but blocks errors.
- Strict blocks warnings and any recorded conversion loss.
- Best effort allows non-fatal incomplete output, preserves diagnostics, and cannot hide install-readiness blockers.
- Salvage is explicit. It may create one marked placeholder per HKX, always reports asset-only salvage, and is never install ready.

## Determinism

JSON is UTF-8 with two-space indentation and stable field construction. ZIP entries use sorted POSIX paths, fixed permissions, and the 1980 ZIP epoch. `SOURCE_DATE_EPOCH` can supply a deterministic manifest timestamp. Absolute local paths are not serialized.

## Non-goals

The service does not convert plugins, quests, dialogue, Papyrus, framework menus, skeletons, or Skyrim LE binaries. It does not edit HKX files or invoke a behavior generator without a separate explicit user action.
