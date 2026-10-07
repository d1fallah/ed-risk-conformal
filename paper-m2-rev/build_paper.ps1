# Build paper/main.tex into a PDF. Mirrors module 1/paper/build_paper.ps1 exactly (same reasoning:
# Windows-side only (MiKTeX); kept out of the reproducible-results pipeline so scientific
# reproduction never depends on a LaTeX install).
#
# Usage (from anywhere):
#   pwsh -File "module2/paper/build_paper.ps1"
param(
    [string]$Target = "main.tex",
    [switch]$SkipBib
)

$ErrorActionPreference = "Continue"
$paperDir = $PSScriptRoot
Set-Location $paperDir

$env:TEXINPUTS = "$paperDir\vendor;" + $env:TEXINPUTS
$env:BSTINPUTS = "$paperDir\vendor;" + $env:BSTINPUTS

$stem = [System.IO.Path]::GetFileNameWithoutExtension($Target)
New-Item -ItemType Directory -Force -Path build -ErrorAction Stop | Out-Null

Write-Output "=== pdflatex (pass 1) ==="
& pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $Target
if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed (exit $LASTEXITCODE) -- see build/$stem.log" }

if (-not $SkipBib -and (Test-Path "refs/m2.bib")) {
    Write-Output "=== bibtex ==="
    & bibtex "build/$stem"
    if ($LASTEXITCODE -ne 0) { Write-Warning "bibtex exited $LASTEXITCODE -- check build/$stem.blg" }

    Write-Output "=== pdflatex (pass 2, resolve citations) ==="
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $Target
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 2 failed (exit $LASTEXITCODE)" }
}

Write-Output "=== pdflatex (final pass, resolve cross-references) ==="
& pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $Target
if ($LASTEXITCODE -ne 0) { throw "pdflatex final pass failed (exit $LASTEXITCODE)" }

Copy-Item "build\$stem.pdf" ".\$stem.pdf" -Force -ErrorAction Stop
$pages = (Select-String -Path "build\$stem.log" -Pattern "Output written on.*\((\d+) page" | Select-Object -Last 1)
Write-Output "=== done: $stem.pdf ($pages) ==="
