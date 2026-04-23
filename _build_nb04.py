"""Generate 04_ablation_temperature.ipynb.

Phase 4: sweep the NT-Xent temperature. Train a short SimCLR run per tau,
then linear-probe each with the fixed protocol. tau=0.5 is the Phase 1
default, so we pull that result in from disk for comparison instead of
re-running it.

Fewer epochs than Phase 1 (100 SSL / 50 probe) so the whole sweep fits in
one overnight window. Every (tau, epoch) step is resumable.
"""
from pathlib import Path
import nbformat as nbf

cells = []
def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))


md("""# Phase 4 — Temperature Ablation

NT-Xent has one knob that changes the loss landscape a lot: the softmax
temperature τ. Low τ makes the similarity distribution sharp (the hardest
negative dominates the gradient); high τ flattens it (all negatives
contribute roughly equally).

Sweep τ ∈ {0.1, 0.5, 1.0, 5.0}. τ=0.5 was the Phase 1 default, so we reuse
that checkpoint + probe result instead of retraining it. The other three
get a shortened SimCLR run (100 epochs vs. 200) followed by the fixed
linear probe. Every temperature's checkpoint is resumable.
""")

# --- Cell 1: Imports & Config
code("""# --- Cell 1: Imports & Config ---
import json, os, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import (
    CIFAR10_MEAN, CIFAR10_STD,
    get_device, get_cifar_resnet18, get_eval_transform, get_simclr_transform,
    ProjectionHead, SimCLRModel, nt_xent_loss, knn_monitor, SimCLRPairDataset,
)

SMOKE_TEST = False  # flip for a fast end-to-end check

# tau=0.5 is Phase 1 — pulled from disk, not retrained
TEMPS = [0.1, 1.0, 5.0]           # tau values to train here
BASELINE_TEMP = 0.5               # Phase 1 default

cfg = dict(
    batch_size    = 256,
    ssl_epochs    = 3  if SMOKE_TEST else 100,
    probe_epochs  = 3  if SMOKE_TEST else 50,
    lr            = 3e-4,
    probe_lr      = 1e-3,
    weight_decay  = 1e-6,
    seed          = 42,
    num_workers   = 0,
    data_root     = './data',
    knn_k         = 20,
    knn_interval  = 10,            # every N SSL epochs
    ckpt_dir      = 'models/ablation_temperature' if not SMOKE_TEST else 'models/ablation_temperature_smoke',
    results_path  = 'results/ablation_temperature.json' if not SMOKE_TEST else 'results/ablation_temperature_smoke.json',
    fig_path      = 'figures/ablation_temperature_curves.png' if not SMOKE_TEST else 'figures/ablation_temperature_smoke.png',
)
torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
os.makedirs(cfg['ckpt_dir'], exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

device = get_device()
print('Device:', device)
print('Temps to train:', TEMPS, ' (baseline tau=', BASELINE_TEMP, 'reused from Phase 1)')
print('Config:', cfg)
""")

# --- Cell 2: Data
code("""# --- Cell 2: Data ---
# Same SSL pipeline as Phase 1, so the only difference between runs is tau.
ssl_transform  = get_simclr_transform(32)
eval_transform = get_eval_transform(32)

train_pair_ds = SimCLRPairDataset(
    CIFAR10(cfg['data_root'], train=True,  download=True, transform=None),
    transform=ssl_transform,
)
train_eval_ds = CIFAR10(cfg['data_root'], train=True,  download=True, transform=eval_transform)
test_ds       = CIFAR10(cfg['data_root'], train=False, download=True, transform=eval_transform)

ssl_loader = DataLoader(
    train_pair_ds, batch_size=cfg['batch_size'], shuffle=True,
    num_workers=cfg['num_workers'], pin_memory=False, drop_last=True,
)
memory_loader = DataLoader(train_eval_ds, batch_size=512, shuffle=False,
                           num_workers=cfg['num_workers'], pin_memory=False)
test_loader   = DataLoader(test_ds,       batch_size=512, shuffle=False,
                           num_workers=cfg['num_workers'], pin_memory=False)
# Probe uses the full train set with eval (no aug) transforms — matches Phase 1/3
probe_train_loader = DataLoader(train_eval_ds, batch_size=cfg['batch_size'],
                                shuffle=True, num_workers=cfg['num_workers'],
                                pin_memory=False, drop_last=False)
print(f'train={len(train_eval_ds)} | test={len(test_ds)} | ssl_batches={len(ssl_loader)}')
""")

# --- Cell 3: Per-tau training routine (resumable)
code("""# --- Cell 3: Train one SimCLR model at a given tau (resumable) ---
def _ckpt_path(tau: float) -> str:
    # e.g. models/ablation_temperature/simclr_tau_0p10.pth
    tag = f'{tau:.2f}'.replace('.', 'p')
    return os.path.join(cfg['ckpt_dir'], f'simclr_tau_{tag}.pth')

def train_simclr_at_tau(tau: float) -> dict:
    ckpt_path = _ckpt_path(tau)
    # Fresh model but seeded, so each tau starts from the same init
    torch.manual_seed(cfg['seed'])
    model = SimCLRModel(get_cifar_resnet18(), ProjectionHead(512, 512, 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    log = {'epoch': [], 'loss': [], 'knn_epoch': [], 'knn_acc': [], 'epoch_sec': []}
    start_epoch = 1

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state']); model.to(device)
        optimizer.load_state_dict(ck['optimizer_state'])
        log = ck['log']
        start_epoch = ck['epoch'] + 1
        print(f\"  [tau={tau}] RESUMING from epoch {start_epoch}\", flush=True)
    else:
        print(f\"  [tau={tau}] starting fresh\", flush=True)

    for epoch in range(start_epoch, cfg['ssl_epochs']+1):
        model.train()
        t0 = time.time()
        loss_sum = n_batches = 0
        pbar = tqdm(ssl_loader, desc=f'tau={tau} ep {epoch}/{cfg[\"ssl_epochs\"]}', leave=False)
        for v1, v2, _ in pbar:
            v1 = v1.to(device); v2 = v2.to(device)
            x = torch.cat([v1, v2], dim=0)
            _, z = model(x)
            z = F.normalize(z, dim=1)
            loss = nt_xent_loss(z, temperature=tau)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu()); n_batches += 1
        avg_loss = loss_sum / max(n_batches, 1)
        dt = time.time() - t0
        log['epoch'].append(epoch); log['loss'].append(avg_loss); log['epoch_sec'].append(dt)

        # periodic kNN
        if epoch == 1 or epoch % cfg['knn_interval'] == 0 or epoch == cfg['ssl_epochs']:
            acc = knn_monitor(model.backbone, memory_loader, test_loader,
                              device=device, k=cfg['knn_k'])
            log['knn_epoch'].append(epoch); log['knn_acc'].append(acc)
            print(f'    [tau={tau}] ep {epoch:3d}/{cfg[\"ssl_epochs\"]}  loss={avg_loss:.4f}  knn={acc*100:.2f}%  ({dt:.1f}s)', flush=True)

        # atomic checkpoint
        tmp = ckpt_path + '.tmp'
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'log': log, 'tau': tau}, tmp)
        os.replace(tmp, ckpt_path)

    return {'ckpt_path': ckpt_path, 'log': log, 'model': model}

print('training routine ready.')
""")

# --- Cell 4: Run SSL for each tau
code("""# --- Cell 4: Train all non-baseline temperatures ---
t_all = time.time()
runs = {}
for tau in TEMPS:
    print(f'>>> SSL run  tau={tau}', flush=True)
    t0 = time.time()
    r = train_simclr_at_tau(tau)
    runs[tau] = {'ckpt_path': r['ckpt_path'], 'log': r['log']}
    print(f'    done in {(time.time()-t0)/60:.1f} min', flush=True)
print(f'\\nall SSL runs done in {(time.time()-t_all)/60:.1f} min')
""")

# --- Cell 5: Linear probe routine
code("""# --- Cell 5: Linear probe each backbone (fixed protocol) ---
@torch.no_grad()
def _eval_with_classifier(backbone, classifier):
    backbone.eval(); classifier.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = classifier(backbone(x)).argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    return correct/total

def probe(backbone, epochs: int) -> dict:
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False
    torch.manual_seed(cfg['seed'])
    clf = nn.Linear(512, 10).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg['probe_lr'],
                           weight_decay=cfg['weight_decay'])
    log = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(1, epochs+1):
        clf.train()
        ls = corr = tot = 0
        for x, y in probe_train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): h = backbone(x)
            logits = clf(h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ls += float(loss.detach().cpu())*y.size(0)
            corr += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
        tr_loss = ls/tot; tr_acc = corr/tot; te_acc = _eval_with_classifier(backbone, clf)
        log['epoch'].append(epoch); log['train_loss'].append(tr_loss)
        log['train_acc'].append(tr_acc); log['test_acc'].append(te_acc)
    return {'final_test_acc': log['test_acc'][-1],
            'best_test_acc':  max(log['test_acc']),
            'curve': log}

print('probe routine ready.')
""")

# --- Cell 6: Run probes for all temps (including baseline reuse)
code("""# --- Cell 6: Run probes for each tau ---
# Pull Phase 1 baseline for tau=0.5 from disk so we don't retrain.
baseline_probe_path = Path('results/simclr_baseline_linear_probe.json')
baseline_result = None
if baseline_probe_path.exists():
    d = json.loads(baseline_probe_path.read_text())
    baseline_result = {
        'final_test_acc': d.get('final_test_acc'),
        'best_test_acc':  d.get('best_test_acc'),
        'curve':          d.get('curve'),
    }
    print(f'Phase 1 baseline (tau={BASELINE_TEMP}) loaded: final={baseline_result[\"final_test_acc\"]*100:.2f}%  best={baseline_result[\"best_test_acc\"]*100:.2f}%')
else:
    print('WARN: Phase 1 probe result not found — comparison table will be missing the tau=0.5 row.')

probe_results = {}
for tau in TEMPS:
    print(f'>>> probe  tau={tau}', flush=True)
    # reload backbone from the disk ckpt (keeps memory lean and lets this cell re-run independently)
    ck = torch.load(_ckpt_path(tau), map_location='cpu', weights_only=False)
    bb = get_cifar_resnet18().to(device)
    # SimCLRModel stores the backbone under .backbone.*, strip that prefix
    bb_state = {k[len('backbone.'):]: v for k, v in ck['model_state'].items() if k.startswith('backbone.')}
    bb.load_state_dict(bb_state)
    t0 = time.time()
    r = probe(bb, cfg['probe_epochs'])
    probe_results[tau] = r
    print(f'    final={r[\"final_test_acc\"]*100:.2f}%  best={r[\"best_test_acc\"]*100:.2f}%  ({time.time()-t0:.1f}s)', flush=True)

# Assemble final result object
final = {
    'temps_trained': TEMPS,
    'baseline_temp': BASELINE_TEMP,
    'ssl_epochs':    cfg['ssl_epochs'],
    'probe_epochs':  cfg['probe_epochs'],
    'runs': {str(tau): {'ssl_log': runs[tau]['log'], 'probe': probe_results[tau]} for tau in TEMPS},
    'baseline': baseline_result,
}
with open(cfg['results_path'], 'w') as f: json.dump(final, f, indent=2)
print('Saved', cfg['results_path'])
""")

# --- Cell 7: Results table
code("""# --- Cell 7: Results table ---
def _fmt(x): return f'{x*100:.2f}%' if x is not None else '   n/a '

rows = []
if baseline_result is not None:
    rows.append((BASELINE_TEMP, baseline_result['final_test_acc'], baseline_result['best_test_acc'], '(Phase 1, 200 ep)'))
for tau in TEMPS:
    r = probe_results[tau]
    rows.append((tau, r['final_test_acc'], r['best_test_acc'], f\"({cfg['ssl_epochs']} ep)\"))
rows.sort(key=lambda t: t[0])

print('Temperature ablation — CIFAR-10 linear probe')
print('=' * 58)
print(f\"{'tau':>6}  {'final':>8}  {'best':>8}   notes\")
print('-' * 58)
for tau, fi, be, note in rows:
    print(f\"{tau:>6}  {_fmt(fi):>8}  {_fmt(be):>8}   {note}\")
""")

# --- Cell 8: Overlay plots
code("""# --- Cell 8: Overlay plots — SSL loss, kNN acc, probe test-acc ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 8a SSL loss
for tau in TEMPS:
    log = runs[tau]['log']
    axes[0].plot(log['epoch'], log['loss'], label=f'tau={tau}')
axes[0].set_xlabel('SSL epoch'); axes[0].set_ylabel('NT-Xent loss')
axes[0].set_title('SSL training loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

# 8b kNN during SSL
for tau in TEMPS:
    log = runs[tau]['log']
    axes[1].plot(log['knn_epoch'], [a*100 for a in log['knn_acc']],
                 marker='o', label=f'tau={tau}')
axes[1].set_xlabel('SSL epoch'); axes[1].set_ylabel('kNN acc (%)')
axes[1].set_title('kNN monitor during SSL'); axes[1].legend(); axes[1].grid(alpha=0.3)

# 8c probe test acc
for tau in TEMPS:
    curve = probe_results[tau]['curve']
    axes[2].plot(curve['epoch'], [a*100 for a in curve['test_acc']], label=f'tau={tau}')
if baseline_result is not None and baseline_result.get('curve'):
    c = baseline_result['curve']
    axes[2].plot(c['epoch'], [a*100 for a in c['test_acc']],
                 linestyle='--', color='black', label=f'tau={BASELINE_TEMP} (Phase 1)')
axes[2].set_xlabel('probe epoch'); axes[2].set_ylabel('test acc (%)')
axes[2].set_title('Linear probe test accuracy'); axes[2].legend(); axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(cfg['fig_path'], dpi=150)
plt.show()
print('Saved', cfg['fig_path'])
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    'kernelspec': {'display_name': 'Python (hw2_env)', 'language': 'python', 'name': 'hw2_env'},
    'language_info': {'name': 'python', 'version': '3.12'},
}
out = Path('04_ablation_temperature.ipynb')
with out.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote', out)
