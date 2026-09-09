param(
    [string]$BuildSuffix = ""
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

function Copy-AllowedTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string[]]$AllowedExtensions
    )

    $sourceRoot = (Resolve-Path -LiteralPath $Source).Path.TrimEnd("\")
    foreach ($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -File) {
        if ($AllowedExtensions -notcontains $file.Extension.ToLowerInvariant()) {
            continue
        }
        $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart("\")
        $target = Join-Path $Destination $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
    }
}

function Get-BlockedFileReason {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

    $blockedExtensions = @(
        ".exe", ".dll", ".pyd", ".com", ".scr", ".msi", ".msp",
        ".bat", ".cmd", ".ps1", ".vbs", ".jar", ".whl",
        ".zip", ".7z", ".rar", ".cab", ".gz", ".bz2", ".xz"
    )
    if ($blockedExtensions -contains $File.Extension.ToLowerInvariant()) {
        return "blocked extension $($File.Extension)"
    }

    $header = New-Object byte[] 8
    $stream = [System.IO.File]::OpenRead($File.FullName)
    try {
        $read = $stream.Read($header, 0, $header.Length)
    }
    finally {
        $stream.Dispose()
    }
    if ($read -ge 2 -and $header[0] -eq 0x4D -and $header[1] -eq 0x5A) {
        return "PE executable signature"
    }
    if ($read -ge 4 -and $header[0] -eq 0x50 -and $header[1] -eq 0x4B -and $header[2] -in 0x03, 0x05, 0x07 -and $header[3] -in 0x04, 0x06, 0x08) {
        return "nested ZIP signature"
    }
    if ($read -ge 6 -and $header[0] -eq 0x37 -and $header[1] -eq 0x7A -and $header[2] -eq 0xBC -and $header[3] -eq 0xAF -and $header[4] -eq 0x27 -and $header[5] -eq 0x1C) {
        return "nested 7z signature"
    }
    if ($read -ge 4 -and $header[0] -eq 0x52 -and $header[1] -eq 0x61 -and $header[2] -eq 0x72 -and $header[3] -eq 0x21) {
        return "nested RAR signature"
    }
    return $null
}

$appName = "Adult Animation Converter"
$script = "Osex-to-OStim-Standalone.py"
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

$archiveName = "$appName $releaseLabel Nexus Source.zip"
$hashName = "$archiveName.sha256.txt"
$stagingRoot = Join-Path $root ".nexus-source-staging"
$appDir = Join-Path $stagingRoot $appName

Remove-WorkspacePath $stagingRoot
Remove-WorkspacePath $archiveName
Remove-WorkspacePath $hashName
New-Item -ItemType Directory -Path $appDir -Force | Out-Null

$topLevelFiles = @(
    "Adult Animation Converter.pyw",
    $script,
    "convert.py",
    "compatibility_db.json",
    "requirements-runtime.txt",
    "NEXUS_SOURCE_README.txt",
    "README_NEXUSMODS.md",
    "BEGINNER_GUIDE.md",
    "TROUBLESHOOTING.md",
    "COMPATIBILITY.md",
    "BUG_REPORT_TEMPLATE.md",
    "README_SECURITY.md",
    "THIRD_PARTY_LICENSES.md",
    "CHANGELOG.md",
    "NEXUSMODS_CHANGELOG.txt",
    "LICENSE"
)
foreach ($path in $topLevelFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required source-release file: $path"
    }
    Copy-Item -LiteralPath $path -Destination (Join-Path $appDir $path) -Force
}

Copy-AllowedTree -Source "animation_converter" -Destination (Join-Path $appDir "animation_converter") -AllowedExtensions @(".py")
Copy-AllowedTree -Source "assets" -Destination (Join-Path $appDir "assets") -AllowedExtensions @(".ico", ".png", ".json", ".txt", ".md")

$blockedFiles = @()
foreach ($file in Get-ChildItem -LiteralPath $appDir -Recurse -File) {
    $reason = Get-BlockedFileReason -File $file
    if ($reason) {
        $relative = $file.FullName.Substring($appDir.Length).TrimStart("\")
        $blockedFiles += "$relative ($reason)"
    }
}
if ($blockedFiles.Count -gt 0) {
    throw "Nexus source release contains blocked content: $($blockedFiles -join ', ')"
}

$manifestRows = foreach ($file in Get-ChildItem -LiteralPath $appDir -Recurse -File | Sort-Object FullName) {
    $relative = $file.FullName.Substring($appDir.Length).TrimStart("\")
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName
    "{0}  {1,12}  {2}" -f $hash.Hash, $file.Length, $relative
}
$manifestLines = @(
    "$appName $releaseLabel transparent source release",
    "",
    "Bundled executable/runtime files: none",
    "Nested archives: none",
    "Archive format: standard ZIP",
    "",
    "SHA256          BYTES  RELATIVE PATH",
    $manifestRows
)
$manifestLines | Set-Content -LiteralPath (Join-Path $appDir "SOURCE_RELEASE_MANIFEST.txt") -Encoding UTF8

Compress-Archive -LiteralPath $appDir -DestinationPath $archiveName -CompressionLevel Optimal -Force
if (-not (Test-Path -LiteralPath $archiveName -PathType Leaf)) {
    throw "Build failed: missing $archiveName"
}

$zipHash = Get-FileHash -Algorithm SHA256 -LiteralPath $archiveName
"$($zipHash.Hash)  $archiveName" | Set-Content -LiteralPath $hashName -Encoding UTF8

Remove-WorkspacePath $stagingRoot

Write-Host "Built $archiveName"
Write-Host "ZIP SHA256: $($zipHash.Hash)"
Write-Host "Nexus source audit: no executable/runtime files or nested archives"
