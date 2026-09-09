# Bug Report Template

Use the app buttons first:

1. `Diagnose Source Archive`
2. `Copy Diagnosis Summary`
3. Convert the archive if diagnosis says it is safe
4. `Verify Converted ZIP`
5. `Copy Nexus Bug Report`

Paste this with the generated report files:

```text
Source pack:
Converter version:
Output type:
OStim Tools mode used, if any:
OStim Tools validation result, if any:
Behavior tool used:
Behavior output mode shown in report:
Mod manager:
Verification result:
Installability reason:
Creature runtime check attached, if this is creature output: yes/no
Detected source type:
Known pack match:
Selected source parser:
Source selection reason:
Fallback source parsers detected:
Selected content root:
Ignored content roots:
Scene JSON files written:
Deployable/playable scenes:
Menu/helper scenes:
HKX packaged:
Behavior events generated:
Scene events missing behavior registration:
Behavior events missing HKX:
Duplicate HKX source count:
Duplicate HKX policy/result:
Human-only OStim output enabled:
Creature scenes detected:
Creature scenes skipped:
Mixed human/creature scenes skipped:
Creature actor roots:
OCreatures-compatible output enabled:
OCreatures menu entry generated:
Creature scenes reachable from OCreatures menu:
OCreatures actor mapping result:
OCreatures -Tn flag result, if shown:
OCreatures reference comparison attached, if run: yes/no
What happened in game:
Did scenes appear in OStim/SexLab search:
Did creature scenes appear in the OCreatures creature menu:
Did actors move:
Did Pandora/Nemesis finish without errors:
Pandora Engine.log imported, if relevant: yes/no
Pandora log analysis status:
Pandora expected module mentioned/skipped:
Pandora output generated evidence:
Pandora no/zero animations evidence:
Report summary:
Report file attached: yes/no
OStim Tools project report attached, if relevant: yes/no
```

Attach `conversion_report_README.txt` and the verification text report whenever possible.
For OStim Tools JSON workflows, also attach `reports/ostim_tools_project_report_README.txt` or `reports/ostim_tools_project_validation_README.txt`.
For OCreatures issues, attach the `OCreatures Output` and `OCreatures Menu Integration` sections, plus the reference comparison report if a maintainer asked you to run one.
For Pandora checkbox/no-output issues, use `Import Pandora Log` on Pandora's `Engine.log` before creating the report bundle.
