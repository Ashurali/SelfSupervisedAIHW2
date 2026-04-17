"""Execute a notebook cell-by-cell in one process, printing separators.

Usage:  python _run_nb_cells.py path/to/nb.ipynb [start_cell] [end_cell]
  Cell indices are 0-based across CODE cells only (markdown is skipped).
  end_cell is exclusive. Omit to run all.
"""
import sys, json, time, traceback
from pathlib import Path

nb_path = Path(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end = int(sys.argv[3]) if len(sys.argv) > 3 else None

nb = json.loads(nb_path.read_text(encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
if end is None: end = len(code_cells)

ns = {'__name__': '__main__'}
for i, cell in enumerate(code_cells):
    if i < start: continue
    if i >= end: break
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    first_line = src.splitlines()[0] if src.strip() else ''
    print(f'\n================ CELL {i}  {first_line[:70]} ================', flush=True)
    t0 = time.time()
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception:
        traceback.print_exc()
        print(f'\n!!! CELL {i} FAILED after {time.time()-t0:.1f}s', flush=True)
        sys.exit(1)
    print(f'--- cell {i} ok  ({time.time()-t0:.1f}s) ---', flush=True)
print('\nAll requested cells completed.')
