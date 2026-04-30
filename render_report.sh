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

pandoc report.md -o "$OUT" \
  --pdf-engine=xelatex \
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
