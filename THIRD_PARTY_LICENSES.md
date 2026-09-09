# Third-Party Licenses

Adult Animation Converter bundles a small set of behavior patch template resources from Animlist Transition Tool so converted OStim Standalone packages can include a real hidden Nemesis/ATT patch instead of only scene JSON and animation list metadata.

## Animlist Transition Tool

- Project: Animlist Transition Tool
- Repository: https://github.com/VersuchDrei/AnimlistTransitionTool
- Purpose in this app: GPL-licensed template resources used to generate hidden Nemesis behavior patch text files for converted FNIS-style animation lists.
- License: GPL-3.0
- Bundled license file: `assets/animlist_transition_tool/LICENSE-GPL-3.0.txt`
- Bundled upstream notes: `assets/animlist_transition_tool/README-AnimlistTransitionTool.md`

The bundled release ZIP keeps these files under `_internal/assets/animlist_transition_tool/`.

## Optional External HKX Converters

Adult Animation Converter can call a user-installed `HavokBehaviorPostProcess.exe`, `hkx32to64.exe`, or `hkxcmd.exe` to convert confirmed Skyrim LE animation files for SE/AE. These executables and their code are not bundled, copied, or redistributed by this project. They remain subject to the licenses and distribution terms of their respective providers.
