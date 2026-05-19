param(
  [Parameter(Mandatory=$true)]
  [string]$ShortcutRoot,

  [string]$LinkRoot = "checkpoints",
  [string]$Manifest = "configs/checkpoint_links_manifest.json"
)

$ErrorActionPreference = "Stop"

function Resolve-ShortcutTarget {
  param([string]$ShortcutPath)
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($ShortcutPath)
  return $shortcut.TargetPath
}

function Get-SafeLinkName {
  param([string]$ShortcutName, [string]$TargetPath)
  $base = [System.IO.Path]::GetFileNameWithoutExtension($ShortcutName)
  $base = $base -replace '\s*-\s*Shortcut$', ''
  $base = $base -replace '\s*-\s*快捷方式$', ''

  $known = @("CADS_series", "MOOSE_series", "nnUNet_private", "UNEST", "VSmTrans", "ePAI", "ePAI_20250421")
  foreach ($name in $known) {
    if ($base -ieq $name -or $TargetPath -imatch [regex]::Escape($name)) {
      if ($name -eq "ePAI_20250421") { return "ePAI_20250421" }
      return $name
    }
  }
  return ($base -replace '[^\w.\-]+', '_')
}

$root = Resolve-Path -LiteralPath $ShortcutRoot
$project = Resolve-Path -LiteralPath "."
$linkRootPath = Join-Path $project $LinkRoot
New-Item -ItemType Directory -Force -Path $linkRootPath | Out-Null

if ((Get-Item -LiteralPath $root).PSIsContainer) {
  $shortcuts = Get-ChildItem -LiteralPath $root -Filter *.lnk -File -Recurse
} else {
  $shortcuts = @(Get-Item -LiteralPath $root)
}

$records = @()
foreach ($lnk in $shortcuts) {
  $target = Resolve-ShortcutTarget $lnk.FullName
  $record = [ordered]@{
    shortcut = $lnk.FullName
    target = $target
    status = $null
    link = $null
    note = "Source target is not modified. This script only creates a junction inside the project checkpoints folder."
  }

  if (-not $target -or -not (Test-Path -LiteralPath $target)) {
    $record.status = "skipped_missing_target"
    $records += [pscustomobject]$record
    continue
  }
  if (-not (Get-Item -LiteralPath $target).PSIsContainer) {
    $record.status = "skipped_target_is_file"
    $records += [pscustomobject]$record
    continue
  }

  $name = Get-SafeLinkName $lnk.Name $target
  $link = Join-Path $linkRootPath $name
  $record.link = $link

  if (Test-Path -LiteralPath $link) {
    $record.status = "skipped_link_already_exists"
    $records += [pscustomobject]$record
    continue
  }

  New-Item -ItemType Junction -Path $link -Target (Resolve-Path -LiteralPath $target) | Out-Null
  $record.status = "linked"
  $records += [pscustomobject]$record
}

$manifestPath = Join-Path $project $Manifest
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $manifestPath) | Out-Null
$records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$records | ConvertTo-Json -Depth 5
