param(
    [string]$BuildSuffix = "",
    [string]$SigningThumbprint = $env:AAC_SIGNING_CERT_THUMBPRINT,
    [string]$TimestampServer = "http://timestamp.digicert.com",
    [switch]$KeepBuildEnvironment
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Remove-WorkspacePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $workspace = (Resolve-Path -LiteralPath $root).Path
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside workspace: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$appName = "Adult Animation Converter"
$script = "Osex-to-OStim-Standalone.py"
$icon = "assets\app_icon.ico"
$requirements = "requirements-release.txt"
$buildEnvironment = Join-Path $root ".release-venv"
$versionMatch = Select-String -LiteralPath $script -Pattern '^CONVERTER_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionMatch) {
    throw "Could not find CONVERTER_VERSION in $script"
}
$version = $versionMatch.Matches[0].Groups[1].Value
$buildSuffix = $BuildSuffix.Trim()
if ($buildSuffix -match '[\\/:*?"<>|]') {
    throw "Build suffix contains characters that are not safe in a Windows filename: $buildSuffix"
}
$releaseLabel = $version
if ($buildSuffix) {
    $releaseLabel = "$version $buildSuffix"
}
$releaseZip = "$appName $releaseLabel.zip"
$releaseHash = "$releaseZip.sha256.txt"
$legacyReleaseZip = "$appName.zip"
$legacyReleaseHash = "$legacyReleaseZip.sha256.txt"

Remove-WorkspacePath "build"
Remove-WorkspacePath "dist"
Remove-WorkspacePath "$appName.spec"
Remove-WorkspacePath "$appName.exe"
Remove-WorkspacePath $releaseZip
Remove-WorkspacePath $releaseHash
Remove-WorkspacePath $legacyReleaseZip
Remove-WorkspacePath $legacyReleaseHash
Remove-WorkspacePath $buildEnvironment

if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Missing pinned release requirements: $requirements"
}

$pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($pythonLauncher) {
    & $pythonLauncher.Source -3.11 -m venv $buildEnvironment
}
else {
    $pythonCommand = Get-Command python.exe -ErrorAction Stop
    & $pythonCommand.Source -m venv $buildEnvironment
}
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the isolated release environment."
}

$releasePython = Join-Path $buildEnvironment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $releasePython -PathType Leaf)) {
    throw "Isolated release Python was not created: $releasePython"
}

& $releasePython -m pip install --disable-pip-version-check --no-input --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the pinned release dependencies."
}

$pythonVersion = (& $releasePython -c "import platform; print(platform.python_version())").Trim()
$pyinstallerVersion = (& $releasePython -c "import PyInstaller; print(PyInstaller.__version__)").Trim()
$dependencySnapshot = @(& $releasePython -m pip freeze --all)

$versionParts = @($version.Split('.') | ForEach-Object { [int]$_ })
while ($versionParts.Count -lt 4) {
    $versionParts += 0
}
$versionTuple = ($versionParts[0..3] -join ", ")
$versionInfoDirectory = Join-Path $root "build"
New-Item -ItemType Directory -Path $versionInfoDirectory -Force | Out-Null
$versionInfoPath = Join-Path $versionInfoDirectory "windows_version_info.txt"
$versionInfo = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($versionTuple),
    prodvers=($versionTuple),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', '$appName'),
         StringStruct('FileDescription', '$appName'),
         StringStruct('FileVersion', '$version.0'),
         StringStruct('InternalName', '$appName'),
         StringStruct('OriginalFilename', '$appName.exe'),
         StringStruct('ProductName', '$appName'),
         StringStruct('ProductVersion', '$version.0'),
         StringStruct('Comments', 'https://www.nexusmods.com/skyrimspecialedition/mods/181197')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"@
$versionInfo | Set-Content -LiteralPath $versionInfoPath -Encoding ASCII

& $releasePython -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --noupx `
    --name $appName `
    --icon $icon `
    --version-file $versionInfoPath `
    --add-data "assets;assets" `
    $script
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$appDir = Join-Path "dist" $appName
$exePath = Join-Path $appDir "$appName.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Build failed: missing $exePath"
}

$forbiddenRuntimeRoots = @(
    "numpy",
    "psutil",
    "yaml",
    "OpenSSL",
    "cryptography",
    "requests",
    "charset_normalizer"
)
$internalRoot = Join-Path $appDir "_internal"
$unexpectedRuntimeRoots = @(
    $forbiddenRuntimeRoots | Where-Object {
        Test-Path -LiteralPath (Join-Path $internalRoot $_)
    }
)
if ($unexpectedRuntimeRoots.Count -gt 0) {
    throw "Release dependency audit failed; unrelated runtime modules were bundled: $($unexpectedRuntimeRoots -join ', ')"
}

$fileVersion = (Get-Item -LiteralPath $exePath).VersionInfo
if (-not $fileVersion.ProductVersion.StartsWith($version, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Build metadata version mismatch: expected $version, found $($fileVersion.ProductVersion)"
}

$signingThumbprint = $SigningThumbprint.Trim().Replace(" ", "")
if ($signingThumbprint) {
    $certificate = Get-ChildItem -Path Cert:\CurrentUser\My, Cert:\LocalMachine\My -CodeSigningCert |
        Where-Object { $_.Thumbprint -eq $signingThumbprint } |
        Select-Object -First 1
    if (-not $certificate) {
        throw "Code-signing certificate was not found for thumbprint $signingThumbprint"
    }
    $signed = Set-AuthenticodeSignature `
        -LiteralPath $exePath `
        -Certificate $certificate `
        -HashAlgorithm SHA256 `
        -TimestampServer $TimestampServer
    if ($signed.Status -ne "Valid") {
        throw "Authenticode signing failed: $($signed.Status) $($signed.StatusMessage)"
    }
}

$signature = Get-AuthenticodeSignature -LiteralPath $exePath
$signatureSubject = if ($signature.SignerCertificate) { $signature.SignerCertificate.Subject } else { "none" }

Copy-Item -LiteralPath "README_NEXUSMODS.md" -Destination (Join-Path $appDir "README_NEXUSMODS.md") -Force
Copy-Item -LiteralPath "BEGINNER_GUIDE.md" -Destination (Join-Path $appDir "BEGINNER_GUIDE.md") -Force
Copy-Item -LiteralPath "TROUBLESHOOTING.md" -Destination (Join-Path $appDir "TROUBLESHOOTING.md") -Force
Copy-Item -LiteralPath "COMPATIBILITY.md" -Destination (Join-Path $appDir "COMPATIBILITY.md") -Force
Copy-Item -LiteralPath "BUG_REPORT_TEMPLATE.md" -Destination (Join-Path $appDir "BUG_REPORT_TEMPLATE.md") -Force
Copy-Item -LiteralPath "README_SECURITY.md" -Destination (Join-Path $appDir "README_SECURITY.md") -Force
Copy-Item -LiteralPath "THIRD_PARTY_LICENSES.md" -Destination (Join-Path $appDir "THIRD_PARTY_LICENSES.md") -Force
Copy-Item -LiteralPath "NEXUSMODS_CHANGELOG.txt" -Destination (Join-Path $appDir "NEXUSMODS_CHANGELOG.txt") -Force
Copy-Item -LiteralPath "compatibility_db.json" -Destination (Join-Path $appDir "compatibility_db.json") -Force
Copy-Item -LiteralPath $script -Destination (Join-Path $appDir $script) -Force
Copy-Item -LiteralPath "convert.py" -Destination (Join-Path $appDir "convert.py") -Force

$provenanceLines = @(
    "$appName $releaseLabel build provenance",
    "",
    "Build layout: PyInstaller one-folder",
    "Isolated build environment: yes",
    "UPX compression: disabled",
    "Python: $pythonVersion",
    "PyInstaller: $pyinstallerVersion",
    "Product version resource: $($fileVersion.ProductVersion)",
    "Authenticode status: $($signature.Status)",
    "Authenticode signer: $signatureSubject",
    "",
    "Pinned build dependencies:",
    $dependencySnapshot
)
$provenanceLines | Set-Content -LiteralPath (Join-Path $appDir "BUILD_PROVENANCE.txt") -Encoding UTF8

$exeHash = Get-FileHash -Algorithm SHA256 -LiteralPath $exePath
$buildKind = "release"
if ($buildSuffix) {
    $buildKind = "$buildSuffix build"
}
$hashLines = @(
    "SHA256 hashes for this ${buildKind}:",
    "",
    "$($exeHash.Hash)  $appName.exe",
    "",
    "This is a folder-style PyInstaller build. Keep the _internal folder beside the EXE.",
    "Authenticode status: $($signature.Status)"
)
$hashLines | Set-Content -LiteralPath (Join-Path $appDir "SHA256SUMS.txt") -Encoding UTF8

if ($buildSuffix) {
    $buildInfo = @(
        "$appName $releaseLabel",
        "",
        "This is a tester/beta build.",
        "Use it for verification before a public release upload."
    )
    $buildInfo | Set-Content -LiteralPath (Join-Path $appDir "BETA_BUILD.txt") -Encoding UTF8
}

$sevenZip = Get-Command 7z.exe -ErrorAction SilentlyContinue
if ($sevenZip) {
    Push-Location "dist"
    try {
        & $sevenZip.Source a -tzip -mx=9 (Join-Path $root $releaseZip) $appName | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Compress-Archive -LiteralPath $appDir -DestinationPath $releaseZip -Force
}

if (-not (Test-Path -LiteralPath $releaseZip)) {
    throw "Build failed: missing $releaseZip"
}

$zipHash = Get-FileHash -Algorithm SHA256 -LiteralPath $releaseZip
"$($zipHash.Hash)  $releaseZip" | Set-Content -LiteralPath $releaseHash -Encoding UTF8

Remove-WorkspacePath "build"
Remove-WorkspacePath "dist"
Remove-WorkspacePath "$appName.spec"
if (-not $KeepBuildEnvironment) {
    Remove-WorkspacePath $buildEnvironment
}

Write-Host "Built $releaseZip"
Write-Host "EXE SHA256: $($exeHash.Hash)"
Write-Host "ZIP SHA256: $($zipHash.Hash)"
Write-Host "Authenticode: $($signature.Status)"
