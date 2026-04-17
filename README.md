# AI HW2 — SimCLR Self-Supervised Learning on CIFAR-10

SimCLR (v1) contrastive learning with a modified ResNet-18 backbone on CIFAR-10, evaluated via kNN monitor and linear probing, compared against supervised and random baselines.

## Hardware & Stack

- **GPU:** AMD RX 6750 GRE 12GB (via DirectML on Windows)
- **CPU/RAM:** Ryzen 7500F · 32 GB RAM
- **Python:** 3.12
- **PyTorch:** 2.4.1 + `torch-directml`

## Setup

```bash
python -m venv hw2_env
hw2_env\Scripts\activate          # Windows
pip install torch-directml         # brings torch==2.4.1 + torchvision==0.19.1
pip install jupyter matplotlib numpy scikit-learn tqdm
```

Launch notebooks:

```bash
jupyter notebook
```

## Project Layout

```
.
├── plan.md                         # full project plan & spec
├── utils.py                        # shared components (model, loss, kNN, etc.)
├── 01_simclr_baseline.ipynb        # Phase 1 — SSL training + linear probe ⭐
├── 02_supervised_baseline.ipynb    # Phase 2 — supervised reference ⭐
├── 03_random_baseline.ipynb        # Phase 3 — random-init lower bound
├── 04_ablation_temperature.ipynb   # Phase 4
├── 05_ablation_batchsize.ipynb     # Phase 5
├── 06_ablation_augmentation.ipynb  # Phase 6
├── 07_ablation_projector.ipynb     # Phase 7
├── 08_transfer_learning.ipynb      # Phase 8
├── 09_analysis_and_plots.ipynb     # Phase 9 — final figures
├── models/                         # checkpoints (.pth, gitignored)
├── results/                        # per-experiment JSON logs
├── figures/                        # publication-quality plots
└── data/                           # CIFAR-10 etc. (gitignored)
```

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | SimCLR baseline (200 ep) | ⏳ |
| 2 | Supervised baseline (200 ep) | — |
| 3 | Random baseline (linear probe only) | — |
| 4 | Temperature ablation (τ ∈ {0.1, 0.5, 1.0, 5.0}) | — |
| 5 | Batch size ablation ({32, 64, 128, 256, 512}) | — |
| 6 | Augmentation ablation | — |
| 7 | Projector head ablation | — |
| 8 | Transfer learning (CIFAR-100, STL-10) | — |
| 9 | Final analysis & report figures | — |

Ablations run at **100 epochs** for time; baselines run at 200.

## DirectML Notes

DirectML uses `privateuseone:0`, not `cuda`. Consequences:

- `torch.cuda.is_available()` returns `False` (expected)
- Use `device = torch_directml.device()` everywhere; `.to(device)` never `.cuda()`
- `pin_memory=False` in DataLoaders
- `torch.compile()` is **not** supported
- Unsupported ops fall back to CPU silently — watch for slowdowns
- Avoid `.item()` every step; accumulate as Python float

See `plan.md` § "DirectML Device Setup" for the full list.
