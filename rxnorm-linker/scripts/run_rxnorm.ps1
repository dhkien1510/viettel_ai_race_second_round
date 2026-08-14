# Interactive runner for RxNorm candidate strategies.
# A separate OutputDir is suggested; choosing InputDir itself enables in-place updates.

param(
    [string]$InputDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$LinkerRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$LabelScript = Join-Path $PSScriptRoot "label_rxnorm_candidates.py"
$StatsScript = Join-Path $PSScriptRoot "stats_rxnorm_tiers.py"

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

$DefaultInput = Join-Path $RepoRoot "submission\0801\Cuong01"
if ([string]::IsNullOrWhiteSpace($InputDir)) {
    $InputAnswer = Read-Host "Input directory [$DefaultInput]"
    $InputDir = if ([string]::IsNullOrWhiteSpace($InputAnswer)) {
        $DefaultInput
    } else {
        $InputAnswer
    }
}
$InputDir = Resolve-RepoPath $InputDir
if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) {
    throw "Input directory does not exist: $InputDir"
}
if ((Get-ChildItem -LiteralPath $InputDir -Filter "*.json" -File).Count -eq 0) {
    throw "Input directory contains no JSON files: $InputDir"
}

$Profiles = @{
    "1" = @{
        Name = "Full: tier 1 + 2 + 3, top 3"
        Suffix = "full_top3"
        Args = @("--top-k", "3")
    }
    "2" = @{
        Name = "Full: tier 1 + 2 + 3, top 1"
        Suffix = "full_top1"
        Args = @("--top-k", "1")
    }
    "3" = @{
        Name = "No tier 3: lexical tier 1 + 2, top 1 (recommended)"
        Suffix = "no_tier3_top1"
        Args = @("--no-tier3", "--top-k", "1")
    }
    "4" = @{
        Name = "No tier 3: lexical tier 1 + 2, top 3"
        Suffix = "no_tier3_top3"
        Args = @("--no-tier3", "--top-k", "3")
    }
    "5" = @{
        Name = "Exact only: exact RxNorm match, top 1"
        Suffix = "exact_only"
        Args = @("--exact-only")
    }
    "6" = @{
        Name = "No tier 3, top 1, unmatched dose falls back to bare ingredient"
        Suffix = "no_tier3_top1_bare"
        Args = @("--no-tier3", "--top-k", "1", "--dose-fallback", "bare")
    }
    "7" = @{
        Name = "Score-guided conservative: heuristic one candidate"
        Suffix = "score_guided_manual"
        Args = @("--score-guided-manual")
    }
}

Write-Host ""
Write-Host "================ RxNorm strategy menu ================" -ForegroundColor Cyan
Write-Host "Input: $InputDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Full tier 1+2+3, top 3"
Write-Host "  2. Full tier 1+2+3, top 1"
Write-Host "  3. No tier 3, top 1 (recommended)" -ForegroundColor Green
Write-Host "  4. No tier 3, top 3"
Write-Host "  5. Exact only"
Write-Host "  6. No tier 3, top 1, bare-dose fallback"
Write-Host "  7. Score-guided conservative heuristic, one candidate" -ForegroundColor Green
Write-Host "  0. Cancel"
Write-Host "=======================================================" -ForegroundColor Cyan

$Choice = Read-Host "Choose a strategy [0-7]"
if ($Choice -eq "0") {
    Write-Host "Cancelled."
    exit 0
}
if (-not $Profiles.ContainsKey($Choice)) {
    throw "Invalid choice: $Choice"
}

$Profile = $Profiles[$Choice]
$JsonCount = (Get-ChildItem -LiteralPath $InputDir -Filter "*.json" -File).Count
$InputItem = Get-Item -LiteralPath $InputDir
$DefaultOutput = Join-Path $InputItem.Parent.FullName ($InputItem.Name + "_" + $Profile.Suffix)
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputAnswer = Read-Host "Output directory [$DefaultOutput]"
    $OutputDir = if ([string]::IsNullOrWhiteSpace($OutputAnswer)) {
        $DefaultOutput
    } else {
        $OutputAnswer
    }
}
$OutputDir = Resolve-RepoPath $OutputDir
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$InPlace = $OutputDir.TrimEnd('\') -ieq $InputDir.TrimEnd('\')

Write-Host ""
Write-Host "Strategy : $($Profile.Name)" -ForegroundColor Yellow
Write-Host "Input    : $InputDir"
Write-Host "Output   : $OutputDir"
Write-Host "Files    : $JsonCount JSON file(s)"
if ($InPlace) {
    Write-Host "MODE     : IN-PLACE; input candidates will be overwritten." -ForegroundColor Red
    Write-Host ""
    $Confirm = Read-Host "Type OVERWRITE to continue"
    if ($Confirm -cne "OVERWRITE") {
        Write-Host "Cancelled; no files were changed."
        exit 0
    }
} elseif (Test-Path -LiteralPath $OutputDir) {
    $ExistingCount = (Get-ChildItem -LiteralPath $OutputDir -Filter "*.json" -File).Count
    $Confirm = Read-Host "Output contains $ExistingCount JSON file(s). Overwrite matching files? [y/N]"
    if ($Confirm -notmatch "^(?i)y(?:es)?$") {
        Write-Host "Cancelled; no files were changed."
        exit 0
    }
}

Push-Location $LinkerRoot
try {
    if ($InPlace) {
        & python $LabelScript --dir $InputDir @($Profile.Args)
    } else {
        & python $LabelScript --dir $InputDir --out $OutputDir @($Profile.Args)
    }
    if ($LASTEXITCODE -ne 0) {
        throw "RxNorm labeling failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Candidate/tier statistics:" -ForegroundColor Cyan
    & python $StatsScript $OutputDir --examples 5
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Labeling succeeded, but statistics failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
if ($InPlace) {
    Write-Host "Done. Input JSON files were updated in place." -ForegroundColor Green
} else {
    Write-Host "Done. Input was not modified." -ForegroundColor Green
}
Write-Host "Output: $OutputDir"
