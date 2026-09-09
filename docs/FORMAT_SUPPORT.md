# Format Support

Support classifications describe the 2.0 package, not the historical 5.7 compatibility module.

| Direction | Classification | Notes |
|---|---|---|
| OStim SA -> neutral IR | exact | All pinned loader fields, sequences, unknown JSON, provenance, events, and assets are represented. |
| OStim SA -> OStim SA | exact for loader semantics | Unknown fields are retained for same-format round trips. Runtime `alignment.json` state is not pack metadata. |
| OSA / OSex / OpenSex -> OStim SA | supported with known losses | Explicit scene/module/stage dialects, speeds, actors, links, offsets, tags, furniture, and actions are handled. Scripts and framework UI are excluded. |
| OStim SA -> OSA / OSex | supported with known losses | Representable scene fields are exported. OStim expressions, `fadeOnEntry`, and some graph/runtime features have no validated equivalent. |
| OSA / OSex -> OStim SA -> OSA / OSex | supported with known losses | The manifest restores source identifiers and retained unknown XML. Unpreserved values are reported, never reconstructed. |
| Legacy converter JSON -> OStim SA | supported with known losses | The old `id` / `pack` / `poses` / `clips` object is imported only by the legacy adapter. |
| Legacy OStim/NG loader-like JSON -> OStim SA | supported | Current-like fields and historical `modPack` casing are accepted by the OStim adapter with normalization diagnostics. |
| SLAL -> OStim SA | import only | Complete per-actor stages become OStim speed variants. Registry, sound, object, and framework semantics may be lossy. |
| Flower Girls / FNIS list -> OStim SA | import only | Only explicit FNIS event-to-HKX rows are imported. Quests, dialogue, spells, scripts, and menus are unsupported. |
| OStim SA -> SLAL | unsupported in 2.0.0 | The legacy 5.7 compatibility API is separate and is not claimed as a 2.0 reverse adapter. |
| OStim SA -> Flower Girls | experimental / unsupported in 2.0.0 | No output is emitted. |
| HKX-only input -> OStim SA | salvage only | Requires `--salvage`; placeholders are visibly marked and never install ready. |
| ESP / ESL / ESM or Papyrus conversion | unsupported | Outside animation metadata conversion. |
| HKX retargeting or LE-to-SE conversion | unsupported | Assets are copied byte-for-byte only. |

## Supported source layouts

- OStim: `Data/SKSE/Plugins/OStim/scenes/**/*.json` or a direct scenes folder; optional `sequences/*.json`.
- OSA/OSex: XML below `meshes/0SA`, `meshes/OSA`, `0Sex`, `OSex`, or OpenSex scene directories; explicit module XML is also recognized.
- SLAL: `SLAnims/json/*.json` with actor stage arrays.
- Flower Girls: Flower Girls/FNIS animation list text with explicit event and HKX columns.

Detection uses these layouts and file contents. Archive filename and extension do not select a format.
