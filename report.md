# Self-Supervised Representation Learning on CIFAR-10
**AI HW2 — SimCLR with a modified ResNet-18 backbone**

---

## 1. Introduction & Research Question

Modern image classification on small benchmarks is dominated by supervised
learning, but every image label is human effort. *Self-supervised
learning* (SSL) trains a representation without labels, using only the
structure of the images themselves; a downstream linear probe (a single
linear layer on top of the frozen representation) then tells us how
much linear-class-separating information the encoder has discovered.

This work re-implements SimCLR v1 [1] on CIFAR-10 with a modified
ResNet-18 backbone, trains a contrastive representation from scratch
(no pre-trained weights), and answers four research questions:

* **Q1.** How close does an SSL representation come to a supervised
  network on CIFAR-10 linear-probe accuracy?
* **Q2.** Which design choices matter most — temperature, batch size,
  augmentation strength, or the projector head?
* **Q3.** Does the SSL representation transfer better to a new dataset
  (CIFAR-100, STL-10) than the supervised representation?
* **Q4.** Where does the SSL benefit show up most clearly — high-label
  or low-label regimes?

All experiments run on a single AMD RX 6750 GRE 12 GB GPU via DirectML
(PyTorch 2.4.1, Windows, Python 3.12). The project follows
*reproducible engineering* practices: every training loop is
checkpoint-resumable, the linear-probe protocol is fixed across
experiments, and the Phase 1 baseline checkpoint is reused as the
"baseline row" of every ablation rather than retrained, saving roughly
40 GPU-hours.

---

## 2. Methods

### 2.1 SimCLR framework

SimCLR generates two random augmented views of each image and trains
the encoder so that the two views of the same image are close in
representation space while views of *different* images are pushed
apart. With a batch of *N* source images we get *2N* views; the
loss for view *i* with positive partner *j* is the NT-Xent loss

$$
\ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$

where $\mathrm{sim}(\cdot,\cdot)$ is cosine similarity, $\tau$ the
temperature, and the sum runs over all *2N − 1* other views in the
batch. The full batch loss is the mean over all *2N* such terms.

### 2.2 Architecture

The backbone is a CIFAR-adapted ResNet-18: the standard 7×7 stride-2
convolution and the following max-pool are replaced with a 3×3 stride-1
convolution and an identity, respectively, since 32×32 images cannot
afford the original aggressive downsampling. Everything else matches
torchvision's `resnet18(weights=None)`. The final fully-connected
layer is replaced by an identity, giving a 512-d representation $h$.

The projector is an MLP $512 \to 512 \,\xrightarrow{\text{ReLU}}\, 128$.
The contrastive loss is computed on the L2-normalized projector
output $z$. The projector is **discarded after training** and the
linear probe operates on the backbone output $h$.

### 2.3 Augmentations

The CIFAR-10 SimCLR augmentation pipeline (per Appendix B.9 of [1])
is `RandomResizedCrop(32, scale=(0.2, 1.0))` →
`RandomHorizontalFlip` → `ColorJitter(0.4, 0.4, 0.4, 0.1)` (`p=0.8`)
→ `RandomGrayscale(p=0.2)`, then ToTensor + normalize. Gaussian blur
is omitted for 32×32 images.

### 2.4 Evaluation protocol

Two evaluations are run on every encoder:

* **kNN monitor** (during SSL training, every 5 epochs): each test
  image's representation is compared to all training representations
  via cosine similarity; the prediction is a weighted vote of the
  *k = 20* nearest neighbors. This gives a fast, label-free signal of
  representation quality during training, since loss alone is
  uninformative.

* **Linear probe** (after training, fixed protocol): a single linear
  layer is trained on top of the frozen encoder for 100 epochs with
  Adam, learning rate $10^{-3}$, weight decay $10^{-6}$, batch size
  256. The same hyperparameters are used for **every** linear probe
  in this report (SSL, supervised-frozen, random) so that probe
  accuracy reflects representation quality, not probe tuning.

### 2.5 Training details

Both SimCLR and supervised baselines train for 200 epochs with Adam
(lr $3\times10^{-4}$, weight decay $10^{-6}$). Batch size 256.
Ablations are also run at **200 SSL epochs**, matching the Phase-1
baseline so every row in every ablation table is directly
comparable. (An earlier round of ablations at 100 epochs is preserved
in git history for reference; we report only the 200-ep numbers
here, since several qualitative findings differ between the two
budgets.) The τ = 0.5, batch size = 256 row is the Phase 1 baseline
in every table. All training loops save an atomic checkpoint every
epoch and resume from disk on restart, so partial runs interrupted
by Windows updates etc. lose at most one epoch of compute.

---

## 3. Experiments & Results

### 3.1 SimCLR baseline

The Phase-1 SimCLR run trained for 200 epochs with τ = 0.5, batch size
256, full augmentation pipeline. Loss decreased monotonically from
5.27 → 4.50 over training; the kNN-monitor accuracy rose smoothly to
**84.5 %** by epoch 200 (Figure 1, left). The downstream linear probe
reached a best test accuracy of **86.70 %** and final accuracy of
86.34 % (Figure 1, right).

![Figure 1 — SimCLR training (kNN) and linear probe](figures/fig1_simclr_baseline.png)
*Figure 1 — Left: SimCLR contrastive loss and kNN-monitor accuracy on CIFAR-10 over 200 epochs. Right: linear-probe test-accuracy curve on the frozen 200-epoch backbone.*

### 3.2 Supervised baseline

A fresh modified ResNet-18 trained end-to-end with cross-entropy on
CIFAR-10 (Adam, 200 epochs) reached **92.54 %** best / 91.47 % final
test accuracy. The supervised network thus outperforms the linear
probe on the SSL representation by **5.84 pp**, as expected — the
supervised network is allowed to update *every* layer to fit labels,
while the SSL-frozen probe has only a 512×10 linear layer to do the
classification with.

### 3.3 Random-init baseline

To isolate "what does SSL pretraining contribute beyond architectural
prior?", a randomly-initialized backbone (no training of any kind) was
linear-probed under the identical fixed protocol. The random
representation reached only **41.76 %** — well below the SimCLR probe.
This controls for any "ResNet-18 features are intrinsically separable"
suspicion: SimCLR adds 44.94 pp of linearly-separable structure on top
of the architectural prior.

#### 3.3 — Summary of CIFAR-10 baselines

| Method | Probe / E2E | Test acc. (best) |
|--------|------------:|-----------------:|
| Supervised (200 ep, end-to-end)              | E2E      | **92.54 %** |
| **SimCLR (200 ep) + linear probe**           | Probe    | **86.70 %** |
| Random-init backbone + linear probe          | Probe    | 41.76 % |

![Figure 2 — CIFAR-10 baseline comparison](figures/fig2_cifar10_comparison.png)
*Figure 2 — CIFAR-10 baseline comparison: random < SimCLR-probe < supervised end-to-end.*

### 3.4 Temperature ablation (Phase 4)

Temperature τ in the NT-Xent softmax controls how sharply the
similarity distribution is concentrated on the hardest negatives. We
sweep τ ∈ {0.1, 0.5, 1.0, 5.0}; τ = 0.5 is the Phase-1 baseline.

| τ    | SSL epochs | Probe best |
|-----:|-----------:|-----------:|
| 0.1  | 200        |   84.35 %  |
| 0.5\* | 200       |   86.70 %  |
| 1.0  | 200        | **86.76 %** |
| 5.0  | 200        |   81.33 %  |

\*Phase 1 baseline reused.

The accuracy curve is U-shaped, but with a notably **flat optimum
between τ = 0.5 and τ = 1.0** — the two are statistically
indistinguishable (86.70 % vs. 86.76 %). Extreme values still hurt:
τ = 0.1 (over-weights the single hardest negative, noisier gradient)
costs 2.4 pp, and τ = 5.0 (flattens the similarity distribution so
no negative dominates) costs 5.4 pp. The qualitative shape matches
Table 5 of [1], though the basin around the optimum is wider in our
setup than in the original ImageNet results.

![Figure 3 — Temperature ablation](figures/fig3_ablation_temperature.png)
*Figure 3 — Linear-probe accuracy vs NT-Xent temperature.*

### 3.5 Batch size ablation (Phase 5)

Larger batches give SimCLR more in-batch negatives, which is
the main source of contrastive signal. We sweep BS ∈ {64, 128, 256}.

| Batch size | SSL epochs | Probe best |
|-----------:|-----------:|-----------:|
| 64         | 200        |   87.12 %  |
| 128        | 200        | **87.27 %** |
| 256\*      | 200        |   86.70 %  |

\*Phase 1 baseline reused.

This is the most surprising finding of the project: at matched
training time, **batch size 64 and 128 each slightly *outperform* the
256-baseline** (by 0.4–0.6 pp). An earlier round of these same
ablations at 100 epochs had shown the opposite ordering — bs = 256
ahead of bs = 64 / 128 by ~6 pp — exactly the result the SimCLR
literature would lead one to expect. The pattern reverses once the
smaller-batch runs are given enough epochs to see a comparable total
number of negatives.

So the bottleneck below ~200 negatives is **not** negative count per
se, but *cumulative* exposure. With a fixed 200-epoch budget, all
three batch sizes converge to roughly the same probe accuracy, with
the smaller batches even gaining a small edge — possibly because
each gradient step is noisier and acts as implicit regularization.
The original "more negatives = better" SimCLR result holds only at
matched *step* counts, not matched *epoch* counts at this scale.

![Figure 4 — Batch size ablation](figures/fig4_ablation_batchsize.png)
*Figure 4 — Linear-probe accuracy vs SSL batch size.*

### 3.6 Augmentation ablation (Phase 6)

Augmentation is the heart of contrastive learning: without strong
augmentation, the two "views" are too similar and the model can match
them by trivial pixel statistics. We compare the full pipeline against
four variants:

* **no_color** — drop ColorJitter
* **no_gray** — drop RandomGrayscale
* **crop_only** — drop both color transforms; keep only random
  resized crop and horizontal flip
* **stronger** — boost ColorJitter strength to (0.8, 0.8, 0.8, 0.2)

| Augmentation     | Probe best | Δ vs full |
|------------------|-----------:|----------:|
| Full (baseline)\* | **86.70 %** | — |
| stronger          |   86.60 %  | −0.10 pp |
| no_color          |   81.73 %  | −4.97 pp |
| no_gray           |   80.85 %  | −5.85 pp |
| crop_only         |   65.18 %  | **−21.52 pp** |

\*Phase 1 baseline reused.

The collapse at *crop_only* is the headline result: removing both
color transforms costs **21.5 pp**, more than the next-worst
ablation by a factor of 4. Two crops of the same image without
color/grayscale variability lets the network shortcut to color
statistics, exactly as predicted by [1] §3 ("composition of
augmentations is critical"). Dropping just one of the two color
transforms costs 5–6 pp; the two are clearly partially redundant
since dropping both costs much more than the sum.

*Stronger* color jitter is statistically indistinguishable from the
default — the standard CIFAR-10 strength is already in a flat
optimum, and pushing it further yields no benefit (and in this case,
no measurable harm either).

![Figure 5 — Augmentation ablation](figures/fig5_ablation_augmentation.png)
*Figure 5 — Linear-probe accuracy under five augmentation regimes.*

### 3.7 Projector ablation (Phase 7)

SimCLR introduces a small MLP "projector" between the backbone and the
contrastive loss; the projector is then discarded. We test three
configurations:

* **A.** With projector, probe on backbone output *h* (= Phase 1)
* **B.** No projector — backbone output is fed directly to the
  contrastive loss; probe on *h*
* **C.** With projector, but probe on the *projected* output *z*
  instead of *h*

| Configuration                          | Probe best |
|----------------------------------------|-----------:|
| **A.** + projector, probe *h* (Phase 1) | **86.73 %** |
| **B.** no projector, probe *h*          |   83.71 %  |
| **C.** + projector, probe *z*           |   83.09 %  |

A − B = **+3.0 pp**: training *with* a projector raises the
*backbone* representation by 3 points, even though the projector
itself is thrown away. A − C = **+3.6 pp**: the projector output *z*
is deliberately tuned for invariance and is less useful for
downstream classification than the un-projected *h*. Interestingly,
B and C end up almost tied (83.71 % vs. 83.09 %) — at 200 epochs the
no-projector backbone is no worse than probing on the projected
features of a *with*-projector model. Both qualitative findings
(projector helps, probe-h-not-z) reproduce [1] Figure 8, although
the magnitude of the projector benefit in our 200-epoch CIFAR-10
setup (3 pp) is smaller than the original ImageNet result. (An
earlier 100-epoch version of this ablation showed the projector
benefit at 11 pp; see "Discussion" for the implication that the
no-projector representation simply takes longer to catch up.)

![Figure 6 — Projector ablation](figures/fig6_ablation_projector.png)
*Figure 6 — Effect of the projector head and the choice of representation for downstream probing.*

### 3.8 Transfer learning (Phase 8)

We freeze each backbone and linear-probe on two new datasets — CIFAR-100
(100 classes, same 32×32 resolution) and STL-10 (10 classes, 96×96
images resized to 32×32). The linear-probe protocol is identical to
the CIFAR-10 setup.

| Dataset    | SSL probe | Supervised-frozen probe | Random-frozen probe |
|------------|----------:|------------------------:|--------------------:|
| CIFAR-100  | **50.07 %** | 48.84 %                 | 19.48 % |
| STL-10     |   74.76 % | **78.77 %**              | 37.40 % |

On CIFAR-100, the SSL representation **outperforms the supervised
backbone by 1.23 pp** — despite the supervised network having
trained on CIFAR-10 *labels*. This is the classic SSL transfer story:
SSL features are less specialized to the source-task labels and
generalize better to a new label space. On STL-10 the supervised
representation wins by 4.0 pp; we suspect this is because the STL-10
image distribution (96×96 photographic images, downsampled to 32×32)
differs significantly more from CIFAR-10 than CIFAR-100 does, and the
supervised backbone's lower-level features happen to transfer well.

![Figure 7 — Transfer learning](figures/fig7_transfer_learning.png)
*Figure 7 — Transfer learning: linear-probe accuracy with each frozen backbone, evaluated on CIFAR-100 and STL-10.*

### 3.9 Deeper analysis (Phase 10)

Three additional analyses on the already-trained backbones:

#### 3.9.1 Label efficiency

We linear-probe each frozen backbone on subsets of the CIFAR-10 train
set: 1 %, 10 %, 50 %, and 100 % of labels (stratified per class).
This isolates the central practical promise of SSL — that an
unlabeled pretraining step is most valuable when downstream labels
are scarce.

| Labels used | SimCLR probe | Supervised-frozen probe | Random-frozen probe |
|------------:|-------------:|------------------------:|--------------------:|
| 1 %         | **80.88 %**    | 91.45 %                | 26.73 % |
| 10 %        | **84.25 %**    | 91.74 %                | 34.59 % |
| 50 %        | **86.31 %**    | 91.92 %                | 38.51 % |
| 100 %       | **86.66 %**    | 92.10 %                | 40.41 % |

The single most striking number in this report: with **just 1 %** of
CIFAR-10 labels (500 images), the SimCLR probe reaches 80.88 % —
within 5.8 pp of its full-data accuracy. The random baseline at 1 %
labels is only 26.73 %. The supervised-frozen row is included for
completeness; it is essentially flat across label fractions because
the supervised backbone was trained on *all* CIFAR-10 labels and
its representation already aligns with the test classes — so the
probe just learns a 10×512 readout regardless of how many labels
are exposed.

![Figure 8 — Label efficiency on CIFAR-10](figures/fig8_label_efficiency_cifar10.png)
*Figure 8 — Linear-probe accuracy as a function of training-label fraction. Frozen backbones: SimCLR (Phase 1), supervised, random.*

#### 3.9.2 t-SNE feature visualization

Two-dimensional t-SNE of the 512-d backbone output on 2 000 random
test images per backbone, colored by true class.

![Figure 9 — t-SNE of test features](figures/fig9_tsne_features.png)
*Figure 9 — t-SNE of CIFAR-10 test features. Random init shows no class structure; SimCLR organizes the ten classes despite never seeing labels; supervised separation is sharper.*

#### 3.9.3 Per-class accuracy & confusion structure

| Class    | SimCLR probe | Supervised-frozen probe | Δ (SimCLR − Supervised) |
|----------|-------------:|------------------------:|------------------------:|
| airplane | 89.90 %      | 91.90 %                 | −2.00 |
| auto     | 96.30 %      | 95.30 %                 | **+1.00** |
| bird     | 79.90 %      | 87.40 %                 | −7.50 |
| cat      | 71.70 %      | 83.10 %                 | −11.40 |
| deer     | 83.20 %      | 93.80 %                 | −10.60 |
| dog      | 75.90 %      | 88.40 %                 | **−12.50** |
| frog     | 91.20 %      | 93.80 %                 | −2.60 |
| horse    | 89.00 %      | 94.20 %                 | −5.20 |
| ship     | 95.30 %      | 95.70 %                 | −0.40 |
| truck    | 93.60 %      | 95.60 %                 | −2.00 |

The gap between SimCLR and supervised is **not uniform**: SimCLR is
within 2 pp on the rigid-shape vehicle classes (auto, ship, truck,
airplane, frog), but loses 10–12 pp on the fine-grained animal
classes (cat, dog, deer, bird). The SSL representation evidently
captures shape-and-color silhouette structure cheaply but struggles
to discriminate visually-similar animals from each other without
label guidance. Auto is the only class where SimCLR *beats*
supervised (+1.0 pp).

![Figure 10 — Per-class accuracy](figures/fig10_per_class_accuracy.png)
*Figure 10 — Per-class linear-probe accuracy of the SimCLR vs supervised-frozen backbones on CIFAR-10.*

![Figure 10b — Confusion matrices](figures/fig10b_confusion_matrices.png)
*Figure 10b — Row-normalized confusion matrices for SimCLR (left) and supervised-frozen (right) linear probes on CIFAR-10.*

#### 3.9.4 Bonus — label efficiency on CIFAR-100 transfer

The same label-efficiency sweep on CIFAR-100 features cached in
Phase 8 — these are the strongest numbers in the report:

| Labels used | SimCLR probe | Supervised-frozen probe | Random-frozen probe | SimCLR − Supervised |
|------------:|-------------:|------------------------:|--------------------:|--------------------:|
| 1 %         | **20.13 %**  | 15.50 %                 | 6.72 %              | **+4.63 pp** |
| 10 %        | **36.55 %**  | 35.56 %                 | 11.45 %             | +0.99 pp |
| 50 %        | **45.68 %**  | 43.63 %                 | 15.26 %             | +2.05 pp |
| 100 %       | **49.43 %**  | 47.14 %                 | 17.93 %             | +2.29 pp |

SimCLR beats the supervised backbone on CIFAR-100 transfer **at every
label fraction**, and its advantage is **largest in the low-label
regime** (+4.63 pp at 1 %). This is the canonical SSL claim made
concrete: a label-free pretrained representation transfers more
robustly than a representation specialized to a particular labeled
task, especially when downstream labels are scarce.

![Figure 11 — Label efficiency on CIFAR-100 transfer](figures/fig11_label_efficiency_cifar100.png)
*Figure 11 — Label efficiency on CIFAR-100 transfer features.*

### 3.10 Extended SimCLR training (Phase 1b, 600 epochs)

The SimCLR paper's Figure 9 reports near-monotonic improvement from
100 → 1000 epochs on ImageNet. To check whether our CIFAR-10 setup
exhibits the same behavior, we trained an additional SimCLR run for
**600 epochs** (3× the Phase-1 budget) on a separate machine with the
same configuration: τ = 0.5, batch size 256, full augmentation
pipeline, identical seed, identical resume-safe checkpoint logic. The
linear-probe protocol matches every other probe in this report.

| Metric                         | 200 ep (Phase 1) | 600 ep (Phase 1b) | Δ |
|--------------------------------|-----------------:|------------------:|--:|
| Final NT-Xent loss             |          4.500   |          4.435    | −0.065 |
| Final kNN-monitor accuracy     |         84.54 %  |         87.95 %   | +3.41 pp |
| **Linear-probe best test acc** |     **86.70 %**  |     **88.88 %**   | **+2.18 pp** |
| Linear-probe final test acc    |         86.34 %  |         88.56 %   | +2.22 pp |

**The extended run closes ≈37 % of the gap to the supervised
end-to-end baseline** — from 5.84 pp (92.54 − 86.70) down to 3.66 pp
(92.54 − 88.88) — at the cost of triple the SSL training time. The
relationship is monotonic but with sharply diminishing returns:

| Epoch range | kNN slope per 100 ep |
|-------------|---------------------:|
| 100 → 200   |             +4.34 pp |
| 200 → 400   |             +1.23 pp |
| 400 → 600   |             +0.08 pp |

By epoch 400 the kNN curve is essentially flat. We attribute this to
two limitations of our configuration relative to the original SimCLR
paper: a much smaller batch size (256 vs. their 8192, so far fewer
in-batch negatives per step) and a smaller backbone (ResNet-18 vs.
ResNet-50), both of which cap how much additional structure the model
can extract from continued contrastive training.

A separate point worth noting: the contrastive loss continues to
*decrease* (4.500 → 4.435) even as the kNN accuracy plateaus
(84.54 % → 87.95 %, with most of the gain in the first 200 extra
epochs). This reinforces a methodological choice we made early —
**monitoring kNN accuracy rather than loss during training** — since
NT-Xent loss tracks cosine-similarity geometry that may continue to
sharpen long after the linear separability of the representation has
saturated.

![Figure 12 — 200 vs 600 epoch comparison](figures/fig12_extended_training_comparison.png)
*Figure 12 — Left: kNN-monitor trajectories of the 200-epoch and 600-epoch SimCLR runs (identical seed, so the curves overlap perfectly through epoch 200). Right: linear-probe accuracy curves on the two frozen backbones — the 600-epoch backbone is uniformly ≈2 pp above the 200-epoch one across all 100 probe epochs.*

---

## 4. Discussion

> **Note to self — this section is for me to write, not the report
> generator. The plan calls for ~1.5 pages addressing four prompts.**

### 4.1 Are these results expected? What surprised me?

*(Prompts: how close does SimCLR come to supervised on CIFAR-10?
Was the projector benefit larger or smaller than I expected? Was the
crop-only collapse surprising? Did anything in the transfer numbers
surprise me — e.g. supervised winning on STL-10? **The biggest
surprise — at matched 200 epochs, smaller batches (64, 128) actually
slightly *outperformed* batch size 256, the opposite of the
canonical SimCLR finding. What does that say about the role of
"more negatives" in the contrastive objective?**)*

### 4.2 Factors affecting results

*(Prompts: which design choice contributed the largest accuracy
swing? **Answer to ground myself: augmentation — specifically the
crop-only ablation, which costs 21.5 pp.** Compare to temperature
(±5 pp), batch size (±0.6 pp at matched epochs), projector (±3 pp).
Did DirectML's CPU fallback for `aten::lerp` materially slow
training? Did running ablations at 100 ep first, then re-running at
200 ep, materially change my conclusions? **Yes — bs and projector
ablation conclusions both flipped between 100 ep and 200 ep, which
is itself a methodological point worth discussing.**)*

### 4.3 What I would do with more time

*(Prompts: extend Phase 1 to 600 + epochs (SimCLR is monotonic);
re-run ablations at 200 ep so they are directly comparable;
implement a momentum encoder + queue (MoCo-v2) as a second SSL
method; sweep larger batch sizes (512+) on a higher-VRAM GPU; train
a deeper backbone (ResNet-34) to test whether SSL benefit grows
with capacity.)*

### 4.4 What I learned

*(Prompts: how does kNN-monitor compare to loss as a training signal?
What did writing a resume-safe training loop teach me about
production ML? What's the difference between linear-probing *h* vs.
*z*, and why does it matter? Why does SSL transfer better than
supervised even when supervised "knew" CIFAR-10 labels?)*

---

## 5. References

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
**A Simple Framework for Contrastive Learning of Visual
Representations**. *Proceedings of the 37th International Conference
on Machine Learning (ICML 2020)*, 1597–1607. arXiv:2002.05709.

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual
Learning for Image Recognition**. *CVPR 2016*. arXiv:1512.03385.

[3] Krizhevsky, A. (2009). **Learning Multiple Layers of Features from
Tiny Images**. Technical report, University of Toronto. (CIFAR-10
dataset.)

[4] Coates, A., Lee, H., & Ng, A. Y. (2011). **An Analysis of
Single-Layer Networks in Unsupervised Feature Learning**. *AISTATS
2011*. (STL-10 dataset.)

[5] van den Oord, A., Li, Y., & Vinyals, O. (2018).
**Representation Learning with Contrastive Predictive Coding**.
arXiv:1807.03748. (NT-Xent / InfoNCE loss origin.)

[6] Wu, Z., Xiong, Y., Yu, S. X., & Lin, D. (2018). **Unsupervised
Feature Learning via Non-Parametric Instance Discrimination**.
*CVPR 2018*. (Memory-bank precursor; kNN-monitor protocol.)

---

## Appendix A — Reproducibility & Code

All notebooks are in the project root (`01_simclr_baseline.ipynb`
through `10_deeper_analysis.ipynb`). Shared components live in
`utils.py`. Each `_build_nbXX.py` script regenerates the
corresponding notebook deterministically from source. Every result
JSON in `results/` and figure in `figures/` is regenerated by
running the notebook of the same number. Random seeds are fixed
(`torch.manual_seed(42)`, `numpy.random.seed(42)`).

**Hardware / software**: AMD RX 6750 GRE 12 GB GPU on Windows;
PyTorch 2.4.1 + DirectML 0.2.5; Python 3.12. DirectML uses the
`privateuseone:0` device — `torch.cuda.is_available()` returns
False, and a small number of operators (notably `aten::lerp.Scalar_out`
inside Adam) silently fall back to CPU; we observed no measurable
slowdown from this.

**Wall-clock cost** (reference numbers):

| Phase | Description                       | Approx. wall time |
|-------|-----------------------------------|------------------:|
| 1 | SimCLR baseline 200 ep                | ~9.3 h |
| 2 | Supervised baseline 200 ep            | ~6 h   |
| 3 | Random-init linear probe              | 8 min  |
| 4 | Temperature sweep (3 × 100 ep)        | ~14 h  |
| 5 | Batch-size sweep (2 × 100 ep)         | ~9 h   |
| 6 | Augmentation sweep (4 × 100 ep)       | ~18 h  |
| 7 | Projector ablation (1 × 100 ep)       | ~5 h   |
| 8 | Transfer learning (CIFAR-100 + STL-10)| ~30 min (features cached) |
| 10| Deeper analysis                       | ~30 min |
| 1b| Extended SimCLR 600 ep (separate GPU) | ~15 h |
