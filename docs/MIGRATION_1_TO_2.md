# Migration from 1.x / Legacy 5.7 to 2.0

## Entry points

Install development/runtime metadata once with:

```powershell
python -m pip install -e .
```

Use:

```powershell
python -m animation_converter inspect INPUT
python -m animation_converter validate INPUT
python -m animation_converter convert INPUT --to ostim-sa --output OUTPUT --package directory
python -m animation_converter gui
```

The historical top-level launchers delegate to the 2.0 command layer. Their old option sets are not a second conversion implementation.

## Behavioral changes

- Current OStim scene IDs come from filenames; an `id` property is not emitted.
- `modpack` is canonical. Historical `modPack` is accepted with a warning.
- Speeds reference animation event names. Direct HKX paths in `clips` are not current scene JSON.
- HKX-only fallback is removed from normal conversion.
- Salvage requires `--salvage` and cannot produce an install-ready package.
- Static source offsets are scene/actor `{x,y,z,r}` fields. The converter no longer manufactures `alignment.json`.
- Broken graph links and missing event/asset mappings are found before finalization.
- A ZIP existing on disk is not proof of install readiness.
- Every output includes `conversion-report.json`, `conversion-report.txt`, and `conversion-manifest.json`.

## Mode mapping

- Previous permissive conversion corresponds most closely to `--best-effort`, but 2.0 still reports blocking validity failures.
- Use normal mode for ordinary conversions.
- Use `--strict` for authoring and CI.
- Use `--salvage` only when scene metadata cannot be recovered.

## Legacy Python API

The large historical module remains importable during migration because the repository's existing compatibility tests and release helpers use its functions. New conversion work must use `ConverterService` and adapters. The compatibility module is not imported by the 2.0 package.
