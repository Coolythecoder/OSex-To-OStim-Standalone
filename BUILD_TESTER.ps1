param(
    [Parameter(Position = 0)]
    [string]$BuildLabel = "beta",
    [string]$SigningThumbprint = $env:AAC_SIGNING_CERT_THUMBPRINT,
    [switch]$KeepBuildEnvironment
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseScript = Join-Path $root "BUILD_RELEASE.ps1"
$label = $BuildLabel.Trim()
if (-not $label) {
    $label = "beta"
}

& $releaseScript `
    -BuildSuffix $label `
    -SigningThumbprint $SigningThumbprint `
    -KeepBuildEnvironment:$KeepBuildEnvironment
