# Security Notes

Adult Animation Converter is published in two forms. The standalone Windows build is a folder-style app; keep `_internal` beside `Adult Animation Converter.exe`. The transparent Nexus source build contains no EXE or bundled runtime and runs through an existing Python 3.11 installation.

## Nexus Mods Upload Quarantine

Nexus Mods may place executable uploads into manual moderation even when antivirus results are clean. Its upload checks can also reject nested archives. The PyInstaller standalone build necessarily contains `Adult Animation Converter.exe`, runtime DLLs, and an internal Python library archive, so no code refactor can guarantee that build an automatic Nexus approval.

`BUILD_NEXUS_SOURCE.ps1` creates `Adult Animation Converter <version> Nexus Source.zip` for the low-friction Nexus upload. It contains inspectable `.py`/`.pyw` source, data, assets, and documentation only. The build rejects PE signatures, executable/runtime extensions, and nested ZIP/7z/RAR signatures before packaging, then writes a payload manifest and an outer-ZIP SHA-256 file.

This removes the common structural reasons for executable quarantine, but Nexus controls its own security checks and may still choose to review any upload. Do not repeatedly delete and re-upload a held file; follow the current [Nexus quarantine guidance](https://help.nexusmods.com/article/117-why-has-my-mod-been-quarantined) and ask Nexus support to review it.

## How Release Builds Are Hardened

`BUILD_RELEASE.ps1` creates a fresh virtual environment from the pinned `requirements-release.txt` file for every build. This prevents unrelated packages from the maintainer's global Python installation being bundled accidentally.

The release uses PyInstaller one-folder mode, disables UPX compression, embeds Windows product/version metadata, audits the result for known unrelated runtime modules, and writes `BUILD_PROVENANCE.txt` plus SHA-256 hashes into the ZIP. It does not use PyInstaller's one-file temporary extraction mode.

The build script supports Authenticode signing through `AAC_SIGNING_CERT_THUMBPRINT` or `-SigningThumbprint`. When no trusted code-signing certificate is configured, `BUILD_PROVENANCE.txt` and `SHA256SUMS.txt` clearly record that the EXE is unsigned.

## SmartScreen Is Not A Malware Detection

`Windows protected your PC` or `not commonly downloaded` is a Microsoft Defender SmartScreen reputation warning. It is separate from Microsoft Defender Antivirus. Microsoft explains that unsigned files start with no publisher reputation for each new version; even a newly signed file can need time to accumulate reputation.

Do not describe a SmartScreen warning as a clean malware scan, and do not assume a malware detection is only SmartScreen. Ask for the exact window text and detection name.

Microsoft's current guidance is available in the [SmartScreen reputation documentation](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation) and [software developer FAQ](https://learn.microsoft.com/en-us/defender-xdr/developer-faq).

## Check The Download

The release ZIP includes `SHA256SUMS.txt` for the EXE and `BUILD_PROVENANCE.txt` for the build environment. The upload also includes a separate versioned ZIP hash file such as `Adult Animation Converter 6.0.zip.sha256.txt`.

To check the EXE hash in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 ".\Adult Animation Converter.exe"
```

To check the ZIP hash:

```powershell
Get-FileHash -Algorithm SHA256 ".\Adult Animation Converter 6.0.zip"
```

The hash you calculate should match the published hash exactly.

## If Defender Antivirus Reports A Detection

Do not disable antivirus globally and do not add a broad folder exclusion. Keep the file quarantined until the exact detection is investigated.

Record the exact detection name, whether Defender named the ZIP or EXE, the SHA-256 hash, the download source, and the converter version. Compare the hash with the published value. The maintainer should submit a clean release that Microsoft detected incorrectly through the [Microsoft Security Intelligence file submission portal](https://www.microsoft.com/en-us/wdsi/filesubmission) as a software developer and wait for the final determination.

A public multi-engine scan can provide extra evidence, but it does not replace local Defender details, source review, signed build provenance, or Microsoft's own determination.

## Signing A Release

Install a trusted code-signing certificate in the Windows certificate store, then either set `AAC_SIGNING_CERT_THUMBPRINT` or pass its thumbprint directly:

```powershell
.\BUILD_RELEASE.ps1 -SigningThumbprint "CERTIFICATE_THUMBPRINT"
```

The build fails if signing was requested but the final Authenticode signature is not valid. Signing improves publisher identity and lets reputation carry across consistently signed releases, but Microsoft does not guarantee that a brand-new signed hash will never show a SmartScreen warning.

## Running From Source

The Nexus source release includes setup instructions in `NEXUS_SOURCE_README.txt`. Install its two pinned GUI dependencies, then run the source script directly:

```powershell
py -3.11 -m pip install --user -r .\requirements-runtime.txt
py -3.11 .\Osex-to-OStim-Standalone.py
```
