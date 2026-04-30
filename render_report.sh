#!/usr/bin/env bash
# One-shot renderer for report.md -> report.pdf.
# Requires pandoc + a TeXLive distribution with xelatex on PATH, plus
# the DejaVu Sans and Microsoft YaHei (or another CJK) fonts installed.
#
# Usage:
#   bash render_report.sh           # writes report.pdf in the project root
#   bash render_report.sh -o out.pdf  # write to a different path

set -euo pipefail
cd "$(dirname "$0")"

OUT="report.pdf"
if [[ "${1:-}" == "-o" && -n "${2:-}" ]]; then
  OUT="$2"
fi

# Find pandoc — try PATH first, then common Windows install locations.
PANDOC=""
if command -v pandoc >/dev/null 2>&1; then
  PANDOC="pandoc"
else
  for cand in \
    "/c/Program Files/Pandoc/pandoc.exe" \
    "/c/Program Files (x86)/Pandoc/pandoc.exe" \
    "$HOME/AppData/Local/Pandoc/pandoc.exe"; do
    if [[ -x "$cand" ]]; then PANDOC="$cand"; break; fi
  done
fi
if [[ -z "$PANDOC" ]]; then
  echo "ERROR: pandoc not found on PATH or in standard install dirs." >&2
  echo "Install from https://pandoc.org or add the install dir to PATH." >&2
  exit 1
fi

# Same for xelatex (used by pandoc via --pdf-engine).
XELATEX=""
if command -v xelatex >/dev/null 2>&1; then
  XELATEX="xelatex"
else
  for cand in \
    "/c/texlive/2026/bin/windows/xelatex.exe" \
    "/c/texlive/2025/bin/windows/xelatex.exe" \
    "/c/texlive/2024/bin/windows/xelatex.exe"; do
    if [[ -x "$cand" ]]; then XELATEX="$cand"; break; fi
  done
fi
if [[ -z "$XELATEX" ]]; then
  echo "ERROR: xelatex not found on PATH or in standard texlive dirs." >&2
  exit 1
fi

echo "Using pandoc:  $PANDOC"
echo "Using xelatex: $XELATEX"

"$PANDOC" report.md -o "$OUT" \
  --pdf-engine="$XELATEX" \
  -V geometry:margin=0.85in \
  -V mainfont="DejaVu Sans" \
  -V CJKmainfont="Microsoft YaHei" \
  -V documentclass=article \
  -V fontsize=12pt \
  --include-in-header=report_header.tex

echo "Wrote $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo "$OUT" | grep "^Pages"
fi
