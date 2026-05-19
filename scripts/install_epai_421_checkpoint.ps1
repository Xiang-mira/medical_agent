param(
    [string]$Archive = "$env:USERPROFILE\Downloads\qchen76_2025_0421.tar.gz",
    [string]$Destination = "checkpoints"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$destinationPath = Join-Path $root $Destination
$targetPath = Join-Path $destinationPath "qchen76_2025_0421"

if (Test-Path -LiteralPath $targetPath) {
    Write-Host "ePAI 2025-04-21 checkpoint already exists: $targetPath"
    exit 0
}

New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
tar -xzf $archivePath -C $destinationPath

$datasetJson = Join-Path $targetPath "nnUNetTrainer__nnUNetPlans__3d_fullres\dataset.json"
$checkpoint = Join-Path $targetPath "nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_final.pth"

if (!(Test-Path -LiteralPath $datasetJson)) {
    throw "Missing dataset.json after extraction: $datasetJson"
}
if (!(Test-Path -LiteralPath $checkpoint)) {
    throw "Missing checkpoint_final.pth after extraction: $checkpoint"
}

Write-Host "Installed ePAI 2025-04-21 checkpoint under: $targetPath"
Write-Host "Private weights are ignored by .gitignore and should not be committed or shared publicly."
