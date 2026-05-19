# PowerShell helper for downloading PanTS/PanTSMini on Windows.
#
# QUICK START (recommended) - stream-download only N cases, no 300 GB needed:
#   conda activate medai-agent
#   python scripts\download_pants_mini.py --n-cases 50
#
# FULL DATASET download (~300 GB) - use this script:
#   .\scripts\download_pants_dataset.ps1
#   .\scripts\download_pants_dataset.ps1 -ImagesOnly
#   .\scripts\download_pants_dataset.ps1 -LabelsOnly
#
# Prefer Git Bash/WSL if tar behavior is inconsistent on Windows.
param(
  [string]$OutDir = "third_party\PanTS-main\data",
  [switch]$ImagesOnly,
  [switch]$LabelsOnly,
  [int]$NumCases = 0        # 0 = full dataset; >0 = call download_pants_mini.py instead
)
$ErrorActionPreference = "Stop"

# Delegate to the smarter Python streaming script when NumCases > 0
if ($NumCases -gt 0) {
  Write-Host "NumCases=$NumCases detected — delegating to download_pants_mini.py (stream mode)"
  $args_list = @("scripts\download_pants_mini.py", "--n-cases", $NumCases, "--output-dir", $OutDir)
  if ($ImagesOnly) { $args_list += "--images-only" }
  if ($LabelsOnly)  { $args_list += "--labels-only" }
  python @args_list
  exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Push-Location $OutDir
Write-Host "Downloading full PanTS/PanTSMini dataset into $((Get-Location).Path)"
Write-Host "This is very large (~300GB+). Make sure you have enough disk space."
Write-Host "TIP: For a quick subset, run instead:"
Write-Host "  python scripts\download_pants_mini.py --n-cases 50"
Write-Host ""

if (-not $LabelsOnly) {
  Invoke-WebRequest -Uri "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" -OutFile "metadata.xlsx"
  New-Item -ItemType Directory -Force -Path "ImageTr" | Out-Null
  New-Item -ItemType Directory -Force -Path "ImageTe" | Out-Null
  for ($i = 1; $i -le 9; $i++) {
    $start = "{0:D8}" -f (($i - 1) * 1000 + 1)
    $end = "{0:D8}" -f ($i * 1000)
    $file = "PanTSMini_ImageTr_${start}_${end}.tar.gz"
    $url = "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
    Write-Host "[$i/9] Downloading $file"
    Invoke-WebRequest -Uri $url -OutFile $file
    tar -xzf $file -C ImageTr
    Remove-Item $file
  }
  $file = "PanTSMini_ImageTe_00009001_00009901.tar.gz"
  $url = "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
  Invoke-WebRequest -Uri $url -OutFile $file
  tar -xzf $file -C ImageTe
  Remove-Item $file
}

if (-not $ImagesOnly) {
  $labelUrl = "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"
  $labelFile = "PanTSMini_Label.tar.gz"
  Invoke-WebRequest -Uri $labelUrl -OutFile $labelFile
  New-Item -ItemType Directory -Force -Path "LabelAll" | Out-Null
  tar -xzf $labelFile -C LabelAll
  Remove-Item $labelFile
  New-Item -ItemType Directory -Force -Path "LabelTr" | Out-Null
  New-Item -ItemType Directory -Force -Path "LabelTe" | Out-Null
  Get-ChildItem LabelAll -Directory | ForEach-Object {
    if ($_.Name -le "PanTS_00009000") { Move-Item $_.FullName LabelTr }
    else { Move-Item $_.FullName LabelTe }
  }
  Remove-Item LabelAll -Force -Recurse -ErrorAction SilentlyContinue
}
Pop-Location
Write-Host "Done. Run: python run_medai_cli.py --json pants-check --pants-root third_party\PanTS-main"
