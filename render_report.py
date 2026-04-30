#!/usr/bin/env python3
"""One-shot renderer for report.md -> report.pdf.

Run from anywhere, in any shell, on any OS:

    python render_report.py
    python render_report.py -o some_other_name.pdf

Requires pandoc and a TeXLive install (xelatex). Will look on PATH first,
then fall back to common Windows install locations.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- locate pandoc -------------------------------------------------------
def find(name_unix: str, fallbacks: list[str]) -> str:
    """Try shutil.which, then a list of explicit paths."""
    p = shutil.which(name_unix) or shutil.which(name_unix + '.exe')
    if p:
        return p
    for cand in fallbacks:
        if Path(cand).is_file():
            return cand
    return ''


PANDOC_FALLBACKS = [
    r"C:\Program Files\Pandoc\pandoc.exe",
    r"C:\Program Files (x86)\Pandoc\pandoc.exe",
    str(Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"),
    str(Path.home() / "AppData" / "Roaming" / "Pandoc" / "pandoc.exe"),
    r"C:\ProgramData\chocolatey\bin\pandoc.exe",
]
XELATEX_FALLBACKS = [
    r"C:\texlive\2026\bin\windows\xelatex.exe",
    r"C:\texlive\2025\bin\windows\xelatex.exe",
    r"C:\texlive\2024\bin\windows\xelatex.exe",
    r"C:\texlive\2023\bin\windows\xelatex.exe",
    r"C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe",
]

pandoc = find('pandoc', PANDOC_FALLBACKS)
xelatex = find('xelatex', XELATEX_FALLBACKS)

if not pandoc:
    print("ERROR: pandoc not found. Install from https://pandoc.org or add to PATH.",
          file=sys.stderr)
    sys.exit(1)
if not xelatex:
    print("ERROR: xelatex not found. Install TeXLive (https://tug.org/texlive)",
          file=sys.stderr)
    sys.exit(1)

print(f"Using pandoc:  {pandoc}")
print(f"Using xelatex: {xelatex}")

# --- args ----------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', default='report.pdf', help='output PDF path')
ap.add_argument('-i', '--input', default='report.md', help='input markdown path')
args = ap.parse_args()

cmd = [
    pandoc, args.input, '-o', args.output,
    f'--pdf-engine={xelatex}',
    '-V', 'geometry:margin=0.85in',
    '-V', 'mainfont=DejaVu Sans',
    '-V', 'CJKmainfont=Microsoft YaHei',
    '-V', 'documentclass=article',
    '-V', 'fontsize=12pt',
    '--include-in-header=report_header.tex',
]

# pandoc resolves relative paths from CWD; ensure we're in the repo root
os.chdir(HERE)

print('Running:', ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
result = subprocess.run(cmd)
if result.returncode != 0:
    print(f"pandoc exited with {result.returncode}", file=sys.stderr)
    sys.exit(result.returncode)

size = Path(args.output).stat().st_size
print(f"Wrote {args.output} ({size:,} bytes)")

# --- page count (optional) ------------------------------------------------
pdfinfo = shutil.which('pdfinfo') or shutil.which('pdfinfo.exe')
if not pdfinfo:
    for cand in [r"C:\texlive\2026\bin\windows\pdfinfo.exe",
                 r"C:\texlive\2025\bin\windows\pdfinfo.exe"]:
        if Path(cand).is_file():
            pdfinfo = cand
            break
if pdfinfo:
    try:
        out = subprocess.run([pdfinfo, args.output],
                             capture_output=True, encoding='utf-8',
                             errors='replace').stdout or ''
        for line in out.splitlines():
            if line.startswith('Pages'):
                print(line)
                break
    except Exception:
        pass
