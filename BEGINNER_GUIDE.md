# Beginner Guide: Using Adult Animation Converter

This guide is written for people who are still new to Skyrim modding. You do not need to understand XML files, animation files, or Python. The program does the conversion work for you.

## What This Program Does

Old OSex, OSex+, OpenSex, SexLab, FlowerGirls, and some older OStim animation packs were made for older systems. Existing OStim Standalone packs can also be rebuilt when you want fresh behavior files, Pandora metadata, reports, or SexLab exports.

This program takes the original mod archive and makes a new ZIP file that is easier to install with OStim Standalone. It can also make a SexLab/SLAL ZIP, or a SexLab P+ ZIP, for people who want to use SexLab instead.

Important: the program does not edit your Skyrim folder directly. It creates a new converted ZIP file. You install that new ZIP with your mod manager.

It can also make an OStim Tools/AAC JSON project or a plain scene JSON folder for manual editing. Those folders are not installable mods. Use them only if you want to review or finish JSON manually, then import the project back into the converter and build a verified OStim Standalone ZIP before installing.

## What You Need First

Before using the converter, make sure you have:

1. Skyrim Special Edition or Anniversary Edition set up for modding.
2. OStim Standalone installed.
3. Pandora or Nemesis installed.
4. A mod manager, such as Mod Organizer 2 or Vortex.
5. The original OSex, OSex+, OpenSex, SexLab/SLAL, FlowerGirls, or OStim Standalone animation pack as a `.zip`, `.7z`, or `.rar` file.
6. This program's extracted `Adult Animation Converter` folder.

If your source mod is a `.7z` or `.rar` file and the converter cannot open it, install 7-Zip normally. You do not need to edit Windows PATH; the converter looks for normal 7-Zip installs automatically. If you are unsure, a `.zip` source file is usually easiest.

For normal OStim use, leave `Human-only OStim output` enabled. This skips creature/animal-root animations from mixed SexLab/SLAL packs so you can use the human scenes without installing a creature setup. Disable it only if you intentionally want creature OStim scenes and already have OCreatures or another OStim creature extension, Creature Framework, matching creature assets such as More Nasty Critters or whatever the pack says it requires, and a behavior generator/setup that supports creature animations.

## Simple Version

The whole process is:

1. Open the converter.
2. Optional but helpful: click `Diagnose Source Archive` and choose the original archive.
3. Choose the OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim Standalone archive for conversion.
4. Wait for the converter to make a new `_OStimSA.zip`.
5. Install that new `_OStimSA.zip` in your mod manager.
6. Run Pandora or Nemesis.
7. Launch Skyrim and test in OStim Standalone.

If you want classic SexLab/SLAL instead of OStim, click `Build SexLab/SLAL ZIP From Archive`. The new file usually ends with `_SexLab.zip`. Install that file, run FNIS/Nemesis/Pandora, then open SexLab Animation Loader in game and register or enable the new pack. When you use Pandora, the converted SexLab pack will not normally appear as a checkbox. Pandora reads its generated `FNIS_*_List.txt` automatically; `Engine.log` should show that name on an `FNIS Mod` line. The converter also writes SexLab sound settings into that export, using the same `SFX fallback action` choice when an old scene does not clearly say what kind of sound it should use. If the original pack had FNIS behavior hook files, the converter keeps them and adds a matching hook for the new generated FNIS list.

Leave `SexLab discovery tags` unchecked for normal SexLab exports. SexLab export now uses a SexLab-native source style by default: for normal MF animations, actor1 is usually Female and actor2 is usually Male, repeated stage sounds and default timers are left out, and actor-specific settings are written only where needed.

Turn `SexLab discovery tags` on only if you specifically want extra finder tags such as `AdultAnimationConverter`, `Converted`, and the pack name in SexLab tools.

If you use SexLab P+, tick `SexLab P+ export` next to the SexLab button before clicking it. The new file usually ends with `_SexLabPPlus.zip`. Install that ZIP and run FNIS/Nemesis/Pandora. The converter includes the P+ registry file for you, so you do not need to use the SLSB compiler separately for normal use. It also uses the same FNIS behavior hook support as the classic SexLab export when the source pack provides those files.

If you want to edit scenes in OStim Tools first, click `Generate OStim Tools JSON` instead of building an installable ZIP. This creates a folder ending in `_OStimToolsProject` with editable scene JSON, a project manifest, a project-local animation asset bundle when source HKX files were available, an OStim Tools-style bridge config companion, reports, and an `AAC_README.txt`. Use `Generate Scene JSON Folder` if you only want loose scene JSON files. Neither output includes behavior registration, and neither output should be installed in your mod manager.

After editing an OStim Tools project, use `Validate OStim Tools Project` to check for missing scene files, duplicate scene IDs, invalid JSON, and unsupported fields. Use `Import OStim Tools Project` to load it back into the converter. To play the edited scenes in game, build a normal verified OStim Standalone ZIP afterward.

## Step By Step

### Step 1: Put The Files Somewhere Easy To Find

Extract the whole `Adult Animation Converter` folder somewhere simple, such as your Downloads folder or a folder on your Desktop.

Keep the `_internal` folder beside `Adult Animation Converter.exe`. Do not copy the EXE out by itself.

Keep the original OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim Standalone mod archive somewhere you can find it. Do not extract it first unless you already know why you need to.

### Step 2: Open The Converter

Double-click `Adult Animation Converter.exe`.

If Windows shows `Windows protected your PC` or `not commonly downloaded`, that is a SmartScreen reputation warning rather than a Defender Antivirus detection. Check the ZIP and EXE hashes against the published values and read `README_SECURITY.md` before deciding whether you trust the download. If Defender gives a named malware or potentially unwanted application detection, do not bypass it; keep the file quarantined and report the exact detection name and SHA-256 hash.

### Step 3: Add The Original Author Name

This step is optional, but recommended.

If you know who made the original animation pack, type their name into `Original mod author`.

This does not change the animations. It only adds the author name to the generated Pandora metadata so credit is clearer.

### Step 4: Leave The Sound Fallback Alone Unless You Need To Change It

In the `Quick Build` area, most people should leave `SFX fallback action` set to:

```text
ostimconvertermoan
```

This helps older OSex/SexLab scenes use OStim voice/moan sound effects when the old files do not clearly say what kind of action the animation is.

If someone helping you with troubleshooting tells you to change it, you can type a built-in OStim action such as `kissing`, or type `none` to turn the fallback off.

### Step 5: Leave The OStim Menu Entry On

In the `Quick Build` area, leave `Add OStim menu entry` checked.

This gives the converted pack a doorway from OStim's main scene menu, so you do not have to rely only on searching for the pack in OStim. Converted scenes linked from that doorway also get a return option back to the pack menu, and some related scene groups get Previous/Next options.

Also leave `Human-only OStim output` checked unless someone specifically tells you to disable it. Many large SexLab/SLAL packs mix human and creature animations. With this option on, the converter keeps the human scenes and clearly reports any skipped creature or mixed human/creature scenes.

If you are making a SexLab package instead, leave `SexLab discovery tags` checked. SexLab does not use the OStim menu icon system, so these tags are the SexLab-friendly way to make the converted pack easier to identify after it is installed.

### Step 6: Convert The Archive

If this is a pack people have reported problems with, click `Diagnose Source Archive` first. The diagnosis tells you what kind of pack it looks like, whether a known compatibility profile matched it, which output type is recommended, whether Pandora or Nemesis is recommended, and whether missing HKX files were found. You can click `Copy Diagnosis Summary` if you need to ask for help on Nexus.

Some SexLab packs contain several branches in one download, such as `SLSB SE`, `SLAL SE`, and `SLAL LE`. If the report says SLSB source JSON was selected and SLAL JSON was ignored, that is usually intentional: the SLSB branch has richer metadata. Read the `Normal User Recommendation` line. For mixed human/creature packs, normal OStim users should usually keep `Human-only OStim output` enabled and install the retained human scenes only.

Click `Build OStim Standalone ZIP From Archive`.

Choose the original OSex/OSex+/OpenSex/SexLab/FlowerGirls/OStim Standalone archive you downloaded. This should be the original mod file, not a file that this converter already made.

The converter will create a new ZIP beside the original file. The new file usually ends with:

```text
_OStimSA.zip
```

That new ZIP is the one you install.

### Step 7: Check The Converted ZIP

After conversion, click `Verify Converted ZIP`.

Choose the new converted ZIP file. For OStim this usually ends with `_OStimSA.zip`. For SexLab it usually ends with `_SexLab.zip`, and for SexLab P+ it usually ends with `_SexLabPPlus.zip`.

If the result says the ZIP passed, it is structurally ready to install. This does not mean every animation is guaranteed to look perfect in game, but it means the package has the files its target system expects. OStim ZIPs are checked for scene JSON, Pandora files, behavior lists, and scene links. SexLab and P+ ZIPs are checked for SLAL or SLSB data, registry files when needed, HKX files, and FNIS list entries that point at the right animation files.

OStim converted ZIPs also include a converter metadata file. You do not need to open it. It helps the Verify button confirm that the ZIP still contains the same scenes, animation files, Pandora files, and behavior list that the converter generated.

The converter may also add a small OStim sound helper called `ostimconvertermoan.json`. That is normal. It comes from the `SFX fallback action` field.

For OStim packs, the converter also preserves modern scene metadata and updates copied custom action files so OStim voice/moan sound effects can work with them.

Some converted scenes may be furniture scenes. That means OStim expects you to start them from a matching object, such as a bed, chair, bench, table, alchemy table, or enchanting table. If a pack has special spawned props or animation objects, such as toy chairs, pillories, crosses, tilted wheels, or glory holes, and OStim does not have a matching furniture type for them, the converter keeps those scenes in the normal selectable list so they are not hidden.

If `Human-only OStim output` was enabled, the Verify report may say that creature scenes were skipped. That is expected for mixed packs and means the generated OStim ZIP is meant for normal human OStim playback. If you disabled the option, the Verify report will say `Creature scenes` and list `Creature actor roots` such as `horse`, `canine`, or `draugr`. If those numbers are above zero, make sure your OStim or SexLab creature requirements and creature behavior generation are installed before testing in game.

For OStim creature output, also check the `OCreatures Output` and `OCreatures Menu Integration` sections. A creature-focused pack should say `OCreatures menu entry generated: yes` and show all retained creature scenes reachable from the OCreatures menu. Test those scenes from the OCreatures creature menu, not only from the normal human OStim scene menu.

Some packs are creature-only. For those, the report may say the ZIP passed with warnings and that creature runtime is required. That means the files and behavior registration look correct, but the converter cannot prove your in-game creature setup is installed. If the report says `pandora_compatible+creature_fnis`, install the ZIP and run Pandora with your creature runtime enabled.

If you leave `Human-only OStim output` enabled on a creature-only pack, there may be no playable scenes left to install. In that case, the converter should warn clearly instead of giving you an empty menu.

### Step 8: Install The New ZIP In Your Mod Manager

Install the new `_OStimSA.zip` with your mod manager.

Do not install an `_OStimToolsProject` folder or `_SceneJSON` folder. Those are editing folders only. They become useful in game only after you import/build them through the normal verified OStim Standalone ZIP workflow.

In Mod Organizer 2, use `Install a new mod from an archive`, then choose the `_OStimSA.zip`.

The converted ZIP should already look valid to MO2. Its archive root contains folders like `SKSE`, `meshes`, and `Nemesis_Engine`, so you should not need to manually set a nested `Data` folder as the data directory. New conversions use a unique module/list code for each pack, so multiple converted packs can be installed without sharing root `DefaultMale.txt` files.

In Vortex, drag the `_OStimSA.zip` into the downloads/mods area, then install and enable it.

Do not install the old OSex/SexLab archive as the final mod for OStim Standalone. The converted `_OStimSA.zip` is the one you want. If your source was already an OStim Standalone pack, use the converted ZIP when you want the rebuilt behavior/Pandora files and reports.

If you made a SexLab export instead, install the generated `_SexLab.zip`. If you checked `SexLab P+ export`, install the generated `_SexLabPPlus.zip`. The default discovery tags do not replace SexLab Animation Loader registration; they just make the pack easier to identify after it is installed.

For FlowerGirls packs, the converter is only taking the animation files and turning them into OStim scenes. It does not convert FlowerGirls quests, dialogue, spells, or framework scripts. If the FlowerGirls file only has plugin or script files such as `.esp`, `.pex`, `.psc`, or `.seq`, install that mod normally with FlowerGirls instead.

### Step 9: Run Pandora Or Nemesis

After installing the converted ZIP, run Pandora or Nemesis.

This step is required. Skyrim animation packs usually need behavior generation before the animations can play correctly.

If you use Pandora, a separate converted-pack checkbox is not required. Pandora reads the generated Nemesis/ATT patch for human animation events. Creature builds also include FNIS-style lists and matching creature behavior hooks.

Some modern OStim packs already include source behavior patch folders such as `Nemesis_Engine/mod/<name>/0_master/`. The converter preserves valid source patches; otherwise it generates a pack-specific one. The Verify button checks the selected path against scene events and HKX files.

If you use Nemesis, you usually will not see a named checkbox for the converted pack. That is normal. The converter creates an FNIS-style animation list that Nemesis can read. If an older converted ZIP shows a converted-pack checkbox with `(null)` beside it, do not tick it; rebuild the original archive with the latest converter.

For converted OStim packs made with `Human-only OStim output` enabled, run Pandora or Nemesis normally for the retained human scenes. For creature output, use the creature-capable behavior generator your setup requires. Many creature setups use Pandora or FNIS-style creature generation. Plain base OStim plus a normal human-only behavior run is not enough for creature scenes.

If Nemesis stops with `ERROR(2006)` and says `File: animationdata`, you are probably using an older converted ZIP that wrote Pandora-style folders into `Nemesis_Engine`. Remove that old converted mod, rebuild the original archive with the latest converter, install the new ZIP, then run Nemesis again.

### Step 10: Start Skyrim And Test

Launch Skyrim through your mod manager.

Open OStim Standalone in game and check whether the converted scenes appear and play.

If you left `Add OStim menu entry` checked, the pack should have a visible entry from OStim's normal human scene menu for supported actor setups. Scenes opened from that generated pack menu should have a return option back to the pack menu. Some related scene groups may also have Previous/Next options. You can still use OStim search if you want to jump straight to a specific scene.

If the report says a pack has furniture-bound scenes, look for those by starting OStim from matching furniture in game. For example, bed scenes are tested from a bed, chair scenes from a chair, and workbench scenes from the matching workbench.

If the report says creature scenes were skipped by `Human-only OStim output`, test the retained human scenes from the normal OStim menu. If you intentionally included creature scenes, test those only after your creature mods and creature behavior generation are working. If human scenes show but creature scenes do not, first check that your creature framework, creature assets, OStim creature extension, and Pandora output are enabled.

If a creature pack installs but does not appear beside other OCreatures packs, open the deploy verification report. The OCreatures menu/index line should say yes. If it says no or shows unreachable creature scenes, do not keep testing in game yet; attach the report bundle so the generated structure can be compared with a known working OCreatures output.

## Which ZIP Should I Install?

Install this one:

```text
OriginalModName_OStimSA.zip
```

Do not install:

```text
OriginalModName.zip
```

The original file is only the source. The `_OStimSA.zip` file is the converted version.

## What If The Converter Shows A Warning?

Some old packs mention animations or old OSex scene links that are not actually included in the original download. When that happens, the converter removes missing speeds, drops scenes that have no valid speeds left, or removes the dead external scene destination so the converted pack stays usable by itself.

If verification still passes, you can usually install the converted ZIP.

If verification fails, open the `_deploy_verify.txt` report beside the ZIP. It tells you what is missing.

If the diagnosis shows a large `Duplicate HKX Handling` section, do not panic by itself. Some archives repeat the same HKX files under multiple source branches. The important checks are whether the report selected the right source branch, whether missing HKX count is zero, and whether the normal or creature-capable recommendation matches your setup.

## Common Problems

### The app will not open

Right-click the EXE, choose `Properties`, and click `Unblock` if Windows shows that option. Then try again.

### The converter cannot open my `.7z` or `.rar`

Install 7-Zip normally, then try again. You do not need to edit Windows PATH. If you use a portable 7-Zip copy, put `7z.exe` beside the converter. You can also use a `.zip` version of the mod archive if one is available.

### I installed the converted ZIP but animations do not play

Run Pandora or Nemesis after installing the converted ZIP. This is the most commonly missed step.

If you use Pandora and the pack appears there but actors stand idle in game, reconvert the original archive with the latest converter, install the new `_OStimSA.zip`, rerun Pandora, and make sure Pandora's output mod is enabled in your mod manager. An old converted ZIP can be visible in Pandora without having the newer AnimData files needed for motion registration.

### I do not know what to choose

For normal use, only use these two buttons:

1. `Build OStim Standalone ZIP From Archive`
2. `Verify Converted ZIP`

You can ignore the advanced fields unless someone specifically tells you to use them.

### Windows says the app is a virus

Make sure you downloaded the newest folder-style release ZIP, not an older single-file EXE. Extract the whole folder and run `Adult Animation Converter.exe` from inside it.

The download includes `README_SECURITY.md` and `SHA256SUMS.txt`. These let you compare the published SHA256 hash with the file you downloaded.

Do not turn off Windows Defender for your whole computer. If Defender still warns on the current clean release, the file can be submitted to Microsoft as a false positive:

```text
https://www.microsoft.com/en-us/wdsi/filesubmission
```

## Quick Checklist

Before starting Skyrim, make sure:

1. You converted the old archive.
2. You installed the new `_OStimSA.zip`.
3. The converted mod is enabled in your mod manager.
4. You ran Pandora or Nemesis after installing it.
5. If you use Pandora, its generated output mod is enabled/deployed in your mod manager.
6. You are launching Skyrim through your mod manager.
7. `Human-only OStim output` was left enabled unless you intentionally wanted creature scenes.
8. If you intentionally included creature scenes, your OStim creature extension, Creature Framework, creature assets, and creature behavior generation are installed and enabled.
9. For OCreatures output, the report says the OCreatures menu entry was generated and the creature scenes are reachable from that menu path.
