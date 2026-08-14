# Chay scripts/label_rxnorm_candidates.py cho tung file .json trong submission/0704/02.
# Script tu tinh duong dan tu vi tri file .ps1, nen co the chay tu bat ky cwd nao.
#
# Cach chay (tu thu muc goc rxnorm-linker):
#   powershell -File scripts\run_rxnorm_on_icd_labeled.ps1

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Dir = Join-Path $RepoRoot "submission\0705\05"

Get-ChildItem -Path $Dir -Filter *.json | ForEach-Object {
    $file = Join-Path $Dir $_.Name
    Write-Host "=== $file ==="
    python scripts/label_rxnorm_candidates.py $file
}
