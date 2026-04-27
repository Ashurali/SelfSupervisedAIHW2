#!/usr/bin/env bash
# Tier-6: re-run ablations 4-7 at 200 SSL epochs on the lab server (RTX 5090).
#
# Assumes:
#   - conda env "hw2" already created and active (or will activate it)
#   - cwd is the repo root
#   - CIFAR-10 will be downloaded on first run (~170 MB)
#
# Outputs go to results/ablation_*.json (overwrites prior 100-ep runs)
# and figures/ablation_*_curves.png. Phase 9 picks them up automatically.

set -euo pipefail

# Activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "hw2" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hw2
fi

mkdir -p logs

PY=python
RUN="$PY -u _run_nb_cells.py"

echo "==================================================================="
echo "RTX-5090 Tier-6 ablation sweep — 200 SSL epochs each"
echo "==================================================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
echo

for nb in 04_ablation_temperature 05_ablation_batchsize 06_ablation_augmentation 07_ablation_projector; do
  log="logs/${nb}_lab.log"
  echo ">>> $nb (log: $log)"
  $RUN "${nb}.ipynb" 2>&1 | tee "$log"
  echo "<<< $nb done"
  echo
done

echo "All ablations done. Re-run phase 9 locally:"
echo "  python -u _run_nb_cells.py 09_analysis_and_plots.ipynb"
