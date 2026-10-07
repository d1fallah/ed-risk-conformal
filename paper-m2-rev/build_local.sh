#!/bin/bash
# Local build of the revised manuscript and its supplement into a scratch folder (does not touch other drafts); then copies both PDFs here.
# main.tex and supplement.tex reference each other through xr (main.aux and supplement.aux), so the order is:
# main (labels) -> supplement x2 (reads main.aux) -> bibtex -> main x2 (reads supplement.aux).
cd "$(dirname "$0")"
SCR="$TEMP/revbuild"; mkdir -p "$SCR"
SCRW="$(cygpath -w "$SCR" 2>/dev/null || echo "$SCR")"
export TEXINPUTS="$(pwd)/vendor;$SCRW;"; export BSTINPUTS="$(pwd)/vendor;"
run() { pdflatex -interaction=nonstopmode -output-directory="$SCR" "$1" >/dev/null 2>&1; }
run main.tex
run supplement.tex
run supplement.tex
bibtex "$SCR/main" >/dev/null 2>&1
run main.tex
pdflatex -interaction=nonstopmode -output-directory="$SCR" main.tex 2>&1 | grep -E "^!|Output written|undefined|Undefined" | head -20
pdflatex -interaction=nonstopmode -output-directory="$SCR" supplement.tex 2>&1 | grep -E "^!|Output written|undefined|Undefined" | head -20
cp "$SCR/main.pdf" ./main.pdf
cp "$SCR/supplement.pdf" ./supplement.pdf
