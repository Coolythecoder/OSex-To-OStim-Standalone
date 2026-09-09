ADULT ANIMATION CONVERTER - TRANSPARENT NEXUS SOURCE RELEASE
============================================================

This edition intentionally contains no EXE, installer, DLL, bundled Python
runtime, batch/PowerShell launcher, or nested archive. Every program file is
plain Python source so Nexus Mods and users can inspect it directly.

REQUIREMENTS
------------

- Windows 10 or 11
- 64-bit Python 3.11 from https://www.python.org/downloads/windows/
- Python's optional Tcl/Tk component (enabled by default in python.org builds)
- 7-Zip for opening source .7z and .rar animation archives

ONE-TIME SETUP
--------------

1. Extract this entire folder somewhere outside Skyrim's Data folder.
2. Open Windows Terminal in this folder.
3. Install the two pinned GUI dependencies:

   py -3.11 -m pip install --user -r requirements-runtime.txt

RUN THE APP
-----------

Double-click:

   Adult Animation Converter.pyw

You can also launch the same GUI from Windows Terminal:

   py -3.11 Osex-to-OStim-Standalone.py

The command-line wrapper remains available:

   py -3.11 convert.py --input "SourcePack.7z" --output "ConvertedPack.zip"

IMPORTANT
---------

- Do not install this source folder with a mod manager.
- Do not put this source folder in Skyrim's Data folder.
- Install only the verified ZIP produced by the converter.
- SOURCE_RELEASE_MANIFEST.txt lists the SHA-256 hash of every payload file.
- The separately published .zip.sha256.txt file authenticates the outer ZIP.

The source edition and standalone EXE edition use the same converter code.
The source edition exists because Nexus Mods commonly sends executable uploads
for manual moderation even when they are clean.
