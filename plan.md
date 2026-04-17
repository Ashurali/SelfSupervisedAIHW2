# AI HW2 — SimCLR Self-Supervised Learning on CIFAR-10

## Overview

Build a SimCLR (v1) contrastive learning framework using ResNet-18 backbone on CIFAR-10 (32×32).
Train from scratch (NO pretrained weights). Evaluate via kNN monitor + linear probing.
Compare SSL vs supervised learning vs random baseline.

**Hardware:** AMD RX 6750 GRE 12GB VRAM · 32GB RAM · Ryzen 7500F
**Framework:** PyTorch 2.4.1 + DirectML (Windows, Python 3.12)
**Format:** Jupyter notebooks (.ipynb), one per phase

---

## File Structure

```
hw2/
├── plan.md                         # this file
├── 01_simclr_baseline.ipynb        # Phase 1: SimCLR training + eval
├── 02_supervised_baseline.ipynb    # Phase 2: supervised training + eval
├── 03_random_baseline.ipynb        # Phase 3: random init linear probe
├── 04_ablation_temperature.ipynb   # Phase 4: temperature sweep
├── 05_ablation_batchsize.ipynb     # Phase 5: batch size sweep
├── 06_ablation_augmentation.ipynb  # Phase 6: augmentation ablation
├── 07_ablation_projector.ipynb     # Phase 7: no projector / projector-as-repr
├── 08_transfer_learning.ipynb      # Phase 8: freeze backbone, probe on other datasets
├── 09_analysis_and_plots.ipynb     # Phase 9: aggregate results, generate report figures
├── models/                         # saved model checkpoints
│   ├── simclr_baseline.pth
│   ├── supervised_baseline.pth
│   ├── random_baseline.pth
│   └── ...
├── results/                        # JSON/CSV logs per experiment
│   ├── simclr_baseline_log.json
│   └── ...
└── figures/                        # publication-quality plots for report
```

---

## Shared Components (define in every notebook or factor into a `utils.py`)

### 1. Modified ResNet-18 for CIFAR-10

```
Changes from standard torchvision ResNet-18:
- conv1: 3×3 kernel, stride=1, padding=1  (was 7×7, stride=2, padding=3)
- Remove max-pool after conv1 → replace with nn.Identity()
- Everything else stays the same
- Output dim = 512 (after global avg pool + flatten)
```

Claude Code instructions:
- Use `torchvision.models.resnet18(weights=None)` and modify in-place
- Wrap in a function `get_cifar_resnet18()` → returns backbone (no fc layer)
- The `fc` layer should be replaced with `nn.Identity()`

### 2. Projector Head

```
MLP: 512 → 512 (ReLU) → 128
```

- Two linear layers with ReLU between them, no activation after last layer
- Used ONLY during SSL training, discarded for evaluation

### 3. NT-Xent Loss

```python
# For a batch of N source images → 2N augmented images
# Compute pairwise cosine similarity matrix (2N × 2N)
# For each image, its "positive" is its augmented mate
# Loss = -log(exp(sim(i,j)/τ) / Σ_{k≠i} exp(sim(i,k)/τ))
# Average over all 2N images
```

Claude Code instructions:
- Implement from scratch (don't use a library)
- Mask out self-similarity (diagonal) from denominator
- Use temperature τ as a parameter
- Input: features tensor of shape (2N, 128), temperature float
- The positive pairs are at indices (i, i+N) and (i+N, i) for i in [0, N)

### 4. Data Augmentations for SimCLR

```python
# SimCLR augmentation pipeline for CIFAR-10 (32×32):
transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```

- NO Gaussian blur for CIFAR-10 (images too small, per SimCLR paper appendix B.9)
- Need a wrapper dataset class that returns TWO augmented views of each image

### 5. kNN Monitor

```
Every K epochs during SSL training:
1. Pass all training images through encoder (no projector) → get representations
2. Pass all test images through encoder → get representations
3. For each test image, find k=20 nearest neighbors in training set
4. Predict label by majority vote (weighted by distance)
5. Report accuracy
```

- Use cosine similarity or L2 distance
- This is the key metric to watch during training (loss alone is misleading)
- Do every 5 epochs

### 6. Linear Probing Protocol (FIXED for all experiments)

```
- Freeze backbone weights entirely
- Attach a single nn.Linear(512, num_classes) layer
- Optimizer: Adam, lr=1e-3, weight_decay=1e-6
- Epochs: 100
- Batch size: same as SSL training (default 256)
- Use FULL training set
- Standard CIFAR-10 test transforms (just normalize, no augmentation)
- Report final test accuracy
```

- This protocol is IDENTICAL across ALL experiments for fair comparison
- For transfer datasets: resize to 32×32, normalize with CIFAR-10 stats

---

## Phase 1 — SimCLR Baseline Training ⭐ REQUIRED

**Notebook:** `01_simclr_baseline.ipynb`

### Cells:

1. **Imports & Config**
   ```
   - torch, torch_directml, torchvision, etc.
   - Device setup: device = torch_directml.device()
   - Config dict:
       batch_size: 256         # start here, reduce to 128 if OOM
       epochs: 200
       lr: 3e-4
       weight_decay: 1e-6
       temperature: 0.5
       projector_dim: 128
       hidden_dim: 512
       knn_k: 20
       knn_interval: 5
   ```

2. **Data Loading**
   ```
   - CIFAR-10 train & test sets
   - Custom dataset wrapper: __getitem__ returns (aug1, aug2, label)
     (label only needed for kNN monitor, NOT for SSL training)
   - SimCLR augmentation pipeline (see shared components)
   - Test set: only normalize (for kNN and linear probe eval)
   - DataLoader with num_workers=4, pin_memory=False, drop_last=True
   ```

3. **Model Definition**
   ```
   - Modified ResNet-18 backbone (see shared components)
   - Projector head: MLP 512→512→128
   - Full SimCLR model wrapping both
   ```

4. **NT-Xent Loss Function**

5. **kNN Monitor Function**
   ```
   @torch.no_grad()
   def knn_monitor(backbone, train_loader, test_loader, k=20, device='cuda'):
       # Extract features, do kNN classification, return accuracy
   ```

6. **Training Loop**
   ```
   - Adam optimizer (lr=3e-4, weight_decay=1e-6)
   - For each epoch:
       - Train one epoch (forward both augmented views, compute NT-Xent loss)
       - Log training loss
       - Every 5 epochs: run kNN monitor, log accuracy
   - Save checkpoint every 50 epochs + final
   - Save training log (loss per epoch, kNN accuracy) to JSON
   ```

7. **Plot Learning Curves**
   ```
   - Two subplots: (1) loss vs epoch, (2) kNN accuracy vs epoch
   - Save to figures/
   ```

8. **Linear Probing**
   ```
   - Load saved backbone
   - Freeze all backbone weights
   - Attach Linear(512, 10)
   - Train with fixed protocol (Adam, lr=1e-3, 100 epochs)
   - Report test accuracy
   - Save linear probe training curve
   ```

**Expected outputs:**
- `models/simclr_baseline.pth`
- `results/simclr_baseline_log.json` (loss, kNN acc per epoch)
- `figures/simclr_baseline_curves.png`
- `results/simclr_baseline_linear_probe.json` (final test acc)

---

## Phase 2 — Supervised Baseline ⭐ REQUIRED

**Notebook:** `02_supervised_baseline.ipynb`

### Cells:

1. **Imports & Config**
   ```
   - Same backbone (Modified ResNet-18), but with classification head:
       backbone → nn.Linear(512, 10)
   - No projector head
   - Standard supervised augmentation (RandomCrop, HorizontalFlip, Normalize)
   - Adam, lr=3e-4, weight_decay=1e-6
   - 200 epochs, same batch_size=256
   - CrossEntropy loss
   ```

2. **Data Loading**
   ```
   - Standard CIFAR-10 augmentations for supervised:
       RandomCrop(32, padding=4)
       RandomHorizontalFlip()
       ToTensor()
       Normalize(CIFAR10_MEAN, CIFAR10_STD)
   ```

3. **Model: ResNet-18 + Linear Head**

4. **Training Loop**
   ```
   - Standard supervised training with CrossEntropy
   - Log train loss, train acc, test acc per epoch
   ```

5. **Evaluation**
   ```
   - Report final test accuracy
   - Compare with SimCLR linear probing result
   ```

6. **Plots**
   ```
   - Loss curve, train/test accuracy curves
   - Save to figures/
   ```

**Expected outputs:**
- `models/supervised_baseline.pth`
- `results/supervised_baseline_log.json`
- `figures/supervised_baseline_curves.png`

---

## Phase 3 — Random Baseline (optional but easy, do it)

**Notebook:** `03_random_baseline.ipynb`

### Cells:

1. **Random Init Backbone**
   ```
   - Initialize ResNet-18 with random weights
   - FREEZE all weights (no training at all)
   - This is the true lower-bound
   ```

2. **Linear Probing on Random Features**
   ```
   - Same protocol as Phase 1 linear probing
   - Report test accuracy
   ```

3. **kNN on Random Features** (for comparison)

**Expected outputs:**
- `results/random_baseline_linear_probe.json`

This should be very fast (~10 min for the 100-epoch linear probe).

---

## Phase 4 — Temperature Ablation (optional)

**Notebook:** `04_ablation_temperature.ipynb`

### Experiments:

| Run | Temperature | Notes |
|-----|------------|-------|
| 1   | 0.1        | Very low — sharp distribution |
| 2   | 0.5        | Default (already done in Phase 1) |
| 3   | 1.0        | Medium |
| 4   | 5.0        | Very high — flat distribution |

### Cells:

1. **Config** — loop/parameterize over temperatures
2. **Training** — full 200-epoch SimCLR for each temperature
3. **Plot overlay** — loss curves and kNN curves for all temperatures on same axes
4. **Linear probing** for each, compare final accuracies

### Key questions to answer:
- How does temperature affect loss magnitude? (Low τ → higher loss values)
- Are loss and kNN accuracy trends consistent? (They may diverge!)
- What's the optimal temperature?

**Expected outputs:**
- `models/simclr_temp_{τ}.pth` for each
- `results/ablation_temperature_log.json`
- `figures/ablation_temperature_curves.png`

---

## Phase 5 — Batch Size Ablation (optional)

**Notebook:** `05_ablation_batchsize.ipynb`

### Experiments:

| Run | Batch Size | Notes |
|-----|-----------|-------|
| 1   | 512       | Largest feasible (12GB VRAM, try first) |
| 2   | 256       | Default |
| 3   | 128       | |
| 4   | 64        | |
| 5   | 32        | Smallest — fewest negatives |

### Cells:

1. **Config** — iterate batch sizes (largest first, to verify VRAM)
2. **Training** — 200 epochs each
3. **Plot overlay** — loss and kNN accuracy for all batch sizes
4. **Linear probing** for each

### Key questions:
- Larger batch → more negatives → better contrastive learning?
- Does the gap shrink with longer training?
- How does batch size affect linear probing accuracy?

**Notes:**
- With DirectML + 12GB VRAM, batch_size=256 should work, 512 is worth trying
- DirectML has higher memory overhead than CUDA/ROCm, so be conservative
- If a batch size crashes, there may be no clean error — just retry smaller
- Keep learning rate constant (don't scale with batch size for simplicity)

**Expected outputs:**
- `models/simclr_bs_{bs}.pth` for each
- `results/ablation_batchsize_log.json`
- `figures/ablation_batchsize_curves.png`

---

## Phase 6 — Augmentation Ablation (optional)

**Notebook:** `06_ablation_augmentation.ipynb`

### Experiments:

| Run | Augmentation Set | Description |
|-----|-----------------|-------------|
| 1   | Full (default)  | Crop + Flip + ColorJitter + Grayscale |
| 2   | No ColorJitter  | Crop + Flip + Grayscale only |
| 3   | No Grayscale    | Crop + Flip + ColorJitter only |
| 4   | Crop only       | Only RandomResizedCrop |
| 5   | Stronger color  | ColorJitter(0.8, 0.8, 0.8, 0.2) |

### Key questions:
- Which augmentation is most critical? (Expected: color distortion)
- Does crop alone provide any useful learning signal?
- Does stronger augmentation help or hurt?

**Expected outputs:**
- `results/ablation_augmentation_log.json`
- `figures/ablation_augmentation_curves.png`

---

## Phase 7 — Projector Head Ablation (optional)

**Notebook:** `07_ablation_projector.ipynb`

### Experiments:

| Run | Config | Training | Evaluation |
|-----|--------|----------|------------|
| 1   | Default | With projector (512→512→128) | Encoder output (512-d) |
| 2   | No projector | Loss on encoder output directly (512-d) | Encoder output (512-d) |
| 3   | Default | With projector | Projector output (128-d) |

### Key questions:
- Does the projector head help? (Expected: yes, significantly)
- Is the encoder or projector representation better for downstream? (Expected: encoder)

### Cells:
1. Run 2: Modify SimCLR model to compute loss on backbone output directly
2. Run 3: At eval time, use projector output (128-d) for kNN and linear probe
3. Compare all three

**Expected outputs:**
- `results/ablation_projector_log.json`
- `figures/ablation_projector_comparison.png`

---

## Phase 8 — Transfer Learning (optional, high value for grade)

**Notebook:** `08_transfer_learning.ipynb`

### Setup:
- Take the three trained backbones: SSL (SimCLR), SL (supervised), Random
- Freeze all backbone weights
- Linear probe on NEW datasets

### Datasets to try (pick 2–3):

| Dataset | Classes | Notes |
|---------|---------|-------|
| CIFAR-100 | 100 | Same domain, more classes — good first choice |
| STL-10 | 10 | Similar to CIFAR-10 but 96×96 — resize to 32×32 |
| Flowers-102 | 102 | Fine-grained, out of domain |

### Cells:

1. **Load pretrained backbones** (SimCLR, Supervised, Random)
2. **For each dataset:**
   ```
   - Download & preprocess (resize to 32×32, normalize)
   - Extract features from each backbone (frozen)
   - Linear probe with FIXED protocol (Adam, lr=1e-3, 100 epochs)
   - Report test accuracy
   ```
3. **Comparison table:**
   ```
   | Dataset   | Random | Supervised | SimCLR |
   |-----------|--------|------------|--------|
   | CIFAR-10  | ??%    | ??%        | ??%    |
   | CIFAR-100 | ??%    | ??%        | ??%    |
   | STL-10    | ??%    | ??%        | ??%    |
   ```
4. **Analysis** — Is SSL or SL better for transfer? (Theory: SSL may generalize better)

**Expected outputs:**
- `results/transfer_learning_results.json`
- `figures/transfer_learning_comparison.png`

---

## Phase 9 — Final Analysis & Report Figures

**Notebook:** `09_analysis_and_plots.ipynb`

### Cells:

1. **Load all results** from JSON/CSV files
2. **Summary table** — all experiments, final metrics
3. **Publication-quality plots:**
   - Fig 1: SimCLR baseline learning curves (loss + kNN)
   - Fig 2: SSL vs SL vs Random comparison bar chart
   - Fig 3: Temperature ablation overlay
   - Fig 4: Batch size ablation overlay
   - Fig 5: Augmentation ablation comparison
   - Fig 6: Projector ablation comparison
   - Fig 7: Transfer learning grouped bar chart
4. **Save all figures** to `figures/` at 300 DPI

---

## DirectML Device Setup (IMPORTANT — read this first)

DirectML uses `privateuseone:0` as the device string, NOT `cuda`.
Every notebook must use the following pattern:

```python
import torch
import torch_directml

# Device setup — use this everywhere instead of torch.device('cuda')
device = torch_directml.device()  # returns device('privateuseone:0')

# Moving tensors/models to GPU:
model = model.to(device)
x = x.to(device)

# NOTE: torch.cuda.is_available() returns False — that's expected!
# NOTE: torch.cuda.memory_allocated() won't work — no VRAM monitoring
# NOTE: Some ops may fall back to CPU silently (DirectML limitation)
```

### DirectML Gotchas for Claude Code:

1. **No `device='cuda'` anywhere.** Always use the `device` variable from above.
2. **No `.cuda()` calls.** Always use `.to(device)`.
3. **No `torch.cuda.*` functions** (memory tracking, synchronize, etc.)
4. **`pin_memory=False`** in DataLoaders (pin_memory is a CUDA feature).
5. **`torch.backends.cudnn.*` calls will fail** — skip or guard them.
6. **Some loss functions may need labels on CPU then moved to device.**
7. **If an op is unsupported**, DirectML falls back to CPU silently — watch for
   unexpected slowdowns as a clue.
8. **`torch.compile()` does NOT work with DirectML.** Do not use it.
9. **`.item()` on a DirectML tensor may be slow** — avoid calling it every step.
   Accumulate loss as a float and log every N steps.

### Verified install:

```bash
pip install torch-directml   # brings torch==2.4.1 + torchvision==0.19.1
pip install jupyter matplotlib numpy scikit-learn tqdm
```

### VRAM & Batch Size Estimates:

DirectML has higher memory overhead than ROCm/CUDA. Start conservative:

```
batch_size=128  → safe starting point
batch_size=256  → probably fine (try this first as default)
batch_size=512  → may work, watch for OOM
```

If a batch size OOMs, there's no clean error — the process may just crash.
Reduce batch size and retry.

### Training Time Estimates (200 epochs, RX 6750 GRE via DirectML):

DirectML is roughly 2-3x slower than native ROCm/CUDA for training.

| Experiment | Est. Time |
|-----------|-----------|
| SimCLR baseline (bs=256) | ~1.5-2.5 hours |
| Supervised baseline | ~45-90 min |
| Random baseline (linear probe only) | ~20 min |
| Temperature ablation (4 runs) | ~6-10 hours |
| Batch size ablation (5 runs) | ~8-12 hours |
| Augmentation ablation (5 runs) | ~8-12 hours |
| Projector ablation (2 extra runs) | ~3-5 hours |
| Transfer learning (3 datasets × 3 models) | ~2-3 hours |
| **Total** | **~30-55 hours** |

With these time estimates, you may want to reduce epochs to 100 for ablation
experiments if time is tight, or run them overnight in sequence.

---

## Execution Order & Dependencies

```
Phase 1 (SimCLR baseline)     ──┐
Phase 2 (Supervised baseline)  ──┼── Phase 3 (Random baseline)
                                 │
                                 ├── Phase 8 (Transfer learning — needs all 3 backbones)
                                 │
Phase 1 done ───┬── Phase 4 (Temperature ablation)
                ├── Phase 5 (Batch size ablation)
                ├── Phase 6 (Augmentation ablation)
                └── Phase 7 (Projector ablation)
                                 │
                        All done ├── Phase 9 (Analysis & plots)
```

### Priority order (if short on time):
1. ⭐ Phase 1 — SimCLR baseline (REQUIRED)
2. ⭐ Phase 2 — Supervised baseline (REQUIRED)
3. Phase 3 — Random baseline (5 min of work, easy points)
4. Phase 4 — Temperature ablation (most interesting results)
5. Phase 5 — Batch size ablation (also interesting)
6. Phase 8 — Transfer learning (high value, shows understanding)
7. Phase 6 — Augmentation ablation
8. Phase 7 — Projector ablation

---

## Report Structure (10 pages max, single-spaced, ≥12pt font)

1. **Introduction & Research Question** (~0.5 page)
   - What is SSL? Why SimCLR? What are we investigating?

2. **Methods** (~1.5 pages)
   - SimCLR framework description (reference the paper)
   - Modified ResNet-18 architecture for CIFAR-10
   - NT-Xent loss formulation
   - Evaluation protocol (kNN monitor + linear probing)
   - NOT a code walkthrough — describe concepts

3. **Experiments & Results** (~5-6 pages)
   - 3.1 Baseline SimCLR training (learning curves, linear probe result)
   - 3.2 Supervised learning comparison
   - 3.3 Random baseline comparison
   - 3.4 Temperature ablation
   - 3.5 Batch size ablation
   - 3.6 Augmentation ablation
   - 3.7 Projector head ablation
   - 3.8 Transfer learning
   - Use tables (NOT screenshots) and plots

4. **Discussion** (~1.5 pages) — WRITE THIS YOURSELF
   - Are results expected? What surprised you?
   - Factors affecting results
   - What would you do with more time?
   - What did you learn?

5. **References** (~0.5 page)

6. **Appendix: Code** (doesn't count toward 10 pages)

---

## Key Hyperparameters Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | ResNet-18 (modified) | 3×3 conv1, no maxpool |
| Optimizer | Adam | For both SSL and probing |
| SSL LR | 3e-4 | |
| SSL weight decay | 1e-6 | |
| SSL epochs | 200 | |
| Temperature | 0.5 | Default |
| Projector | 512→512→128 | MLP with ReLU |
| kNN k | 20 | |
| kNN interval | 5 epochs | |
| Linear probe LR | 1e-3 | FIXED across all experiments |
| Linear probe WD | 1e-6 | FIXED |
| Linear probe epochs | 100 | FIXED |
| Batch size | 256 (default) | Try up to 512 |

---

## CIFAR-10 Normalization Constants

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
```
