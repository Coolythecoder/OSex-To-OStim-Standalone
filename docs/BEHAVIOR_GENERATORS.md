# Behavior Generators

Scene JSON, animation events, actor-to-event mappings, HKX assets, and behavior registration are separate layers. A package may contain valid JSON and byte-identical HKX files while still leaving actors idle because no behavior graph dispatches the event.

## Current 2.0 behavior

`--behavior pandora` and `--behavior nemesis` write a validated `behavior-registration.json` under converter metadata. The file records event, actor index, target HKX path, and hash. It is deliberately a registration manifest, not an invented generator patch.

The converter does not invoke Pandora or Nemesis during conversion. The user must install the output, run the selected generator explicitly, inspect its return status/output, and deploy the generated behavior output.

`--behavior none` still records event mappings but marks behavior registration incomplete.

Until generator-native fixture syntax is implemented and verified, all three modes make `behaviorRegistrationComplete` false and therefore prevent an install-ready label. This is conservative by design.

## Evidence inspected

- Ayasato Animations includes a real `Nemesis_Engine/mod/ayasat` patch with many numbered fragments, demonstrating that a text file containing HKX paths is not a complete Nemesis patch.
- Current local Pandora-oriented converter outputs include named animation data and animation set data, but those generated files are not treated as an authoritative general-purpose writer specification.

## Explicit external execution API

`animation_converter.behavior.run_external_generator` accepts an executable and argument list, uses `shell=False`, captures stdout/stderr, and returns the exit code. Front ends must call it only after an explicit user action. Conversion never calls it automatically.

## Annotations

The converter does not inject or modify binary HKX annotations. `doPeaks` and `peaksAnnotated` scene metadata is preserved where the destination supports it. Unsupported engine-specific annotations are reported.
