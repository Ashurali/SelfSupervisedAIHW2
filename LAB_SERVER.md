# Lab-Server Quickstart — RTX 5090 / Ubuntu / Conda

This file documents how to run the heavy stuff on the lab box. The
local AMD machine handles development; the lab box handles the
overnight Tier-5 / Tier-6 runs.

## Hardware target

- Ubuntu (any recent LTS)
- AMD Ryzen 9 9950X · 128 GB RAM
- **NVIDIA RTX 5090 (Blackwell, sm_120)** — needs CUDA 12.8 wheels

## One-time setup

```bash
# 1. clone
git clone https://github.com/Ashurali/SelfSupervisedAIHW2.git
cd SelfSupervisedAIHW2

# 2. conda env
conda create -n hw2 python=3.12 -y
conda activate hw2

# 3. install — CUDA 12.8 wheels (Blackwell needs >= PyTorch 2.6)
pip install -r requirements_cuda.txt \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

# 4. sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.6+cu128  True  NVIDIA GeForce RTX 5090
```

## Running the workloads

### Tier-5 — extended SimCLR baseline (600 epochs)

The single highest-value run. Saves to a separate path so the
existing 200-ep Phase-1 result stays intact.

```bash
python -u _run_nb_cells.py 01b_simclr_extended.ipynb 2>&1 | tee logs/phase1b_lab.log
```

Estimated wall time: ~2 hours on RTX 5090.

Outputs:
- `models/simclr_extended_600ep.pth` (gitignored)
- `results/simclr_extended_600ep_log.json`
- `results/simclr_extended_600ep_linear_probe.json`
- `figures/simclr_extended_600ep_curves.png`

### Tier-6 — re-run ablations at 200 SSL epochs

The four ablation notebooks (Phases 4–7) were originally run at 100
epochs to fit the AMD compute budget. Re-running them at 200 ep makes
every row directly comparable to the 200-epoch Phase-1 baseline.

One shell script kicks off all four sequentially:

```bash
bash run_lab_ablations.sh
```

Estimated wall time: ~5–6 hours total on RTX 5090 (4 + 2 + 4 + 1 SSL
training runs at 200 ep each, ~30s/ep).

Outputs (overwrite the prior 100-ep results in working tree — git
history preserves the originals):
- `results/ablation_temperature.json`
- `results/ablation_batchsize.json`
- `results/ablation_augmentation.json`
- `results/ablation_projector.json`
- `figures/ablation_*_curves.png`

## Resuming after interruption

Every training loop saves an atomic checkpoint every epoch. If the
lab session times out, just re-run the same command — it'll resume
from the last completed epoch.

## After it all finishes

```bash
git add results/ figures/
git commit -m "tier-5/6 lab-server runs: 600-ep simclr + 200-ep ablations"
git push origin main
```

Then on the AMD machine, `git pull` and re-run Phase 9 to refresh the
aggregate figures + summary CSV:

```bash
python -u _run_nb_cells.py 09_analysis_and_plots.ipynb
```
