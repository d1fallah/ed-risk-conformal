#!/bin/bash
# Build the extended version (main + supplement, every analysis) into a scratch folder, then copy main_extended.pdf and supplement_extended.pdf next to the script.
# Same xr order as ../build_local.sh. Shares ../floats, ../rv_numbers.tex and ../refs with the submitted version.
cd "$(dirname "$0")"
SCR="$TEMP/revbuild_ext"; mkdir -p "$SCR"
SCRW="$(cygpath -w "$SCR" 2>/dev/null || echo "$SCR")"
VEND="$(cd ../vendor && pwd -W 2>/dev/null || (cd ../vendor && pwd))"
export TEXINPUTS="$VEND;$SCRW;"; export BSTINPUTS="$VEND;"
run() { pdflatex -interaction=nonstopmode -output-directory="$SCR" "$1" >/dev/null 2>&1; }
run main.tex
run supplement.tex
run supplement.tex
bibtex "$SCR/main" >/dev/null 2>&1
run main.tex
pdflatex -interaction=nonstopmode -output-directory="$SCR" main.tex 2>&1 | grep -E "^!|Output written|undefined|Undefined" | head -20
pdflatex -interaction=nonstopmode -output-directory="$SCR" supplement.tex 2>&1 | grep -E "^!|Output written|undefined|Undefined" | head -20
cp "$SCR/main.pdf" ./main_extended.pdf
cp "$SCR/supplement.pdf" ./supplement_extended.pdf
