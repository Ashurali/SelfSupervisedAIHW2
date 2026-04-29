"""Verification: re-train bs=256 SimCLR for 200 ep on whatever GPU runs this,
then linear-probe. Use this to disambiguate whether the bs=64/128 vs bs=256
gap (from ablation_batchsize.json) is a true batch-size effect or just
CUDA-vs-DirectML floating-point drift between the two training hosts.

Run on the same machine that trained bs=64 and bs=128 (the 4090):

    python -u verify_bs256_4090.py

Resume-safe — every epoch atomically saves a checkpoint. Saves results to
results/verify_bs256_4090.json (separate from the main ablation file so
nothing existing is disturbed).
"""
import json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from utils import (
    get_device, get_cifar_resnet18, get_eval_transform, get_simclr_transform,
    SimCLRModel, nt_xent_loss, knn_monitor, SimCLRPairDataset,
)

# Match Phase 1 / Phase 5 exactly, except now we *train* bs=256 fresh
# on this host instead of reusing the DirectML Phase-1 checkpoint.
cfg = dict(
    bs            = 256,
    ssl_epochs    = 200,
    probe_epochs  = 50,
    lr            = 3e-4,
    probe_lr      = 1e-3,
    weight_decay  = 1e-6,
    temperature   = 0.5,
    seed          = 42,
    num_workers   = 0,
    data_root     = './data',
    knn_k         = 20,
    knn_interval  = 10,
    ckpt_path     = 'models/verify/simclr_bs256_verify.pth',
    results_path  = 'results/verify_bs256_4090.json',
)
torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
os.makedirs(os.path.dirname(cfg['ckpt_path']), exist_ok=True)
os.makedirs('results', exist_ok=True)

device = get_device()
print('Device:', device)
print('Config:', cfg)

# --- Data
ssl_transform  = get_simclr_transform(32)
eval_transform = get_eval_transform(32)

pair_ds       = SimCLRPairDataset(CIFAR10(cfg['data_root'], train=True, download=True, transform=None),
                                  transform=ssl_transform)
train_eval_ds = CIFAR10(cfg['data_root'], train=True,  download=True, transform=eval_transform)
test_ds       = CIFAR10(cfg['data_root'], train=False, download=True, transform=eval_transform)

ssl_loader    = DataLoader(pair_ds, batch_size=cfg['bs'], shuffle=True,
                           num_workers=cfg['num_workers'], pin_memory=False, drop_last=True)
memory_loader = DataLoader(train_eval_ds, batch_size=512, shuffle=False,
                           num_workers=cfg['num_workers'], pin_memory=False)
test_loader   = DataLoader(test_ds, batch_size=512, shuffle=False,
                           num_workers=cfg['num_workers'], pin_memory=False)
probe_train_loader = DataLoader(train_eval_ds, batch_size=256,
                                shuffle=True, num_workers=cfg['num_workers'],
                                pin_memory=False, drop_last=False)
print(f'data ready. ssl_batches={len(ssl_loader)}')

# --- SSL training (resumable)
torch.manual_seed(cfg['seed'])
model = SimCLRModel(get_cifar_resnet18(), hidden_dim=512, proj_dim=128).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
log   = {'epoch': [], 'loss': [], 'knn_epoch': [], 'knn_acc': [], 'epoch_sec': []}
start_epoch = 1

if os.path.exists(cfg['ckpt_path']):
    ck = torch.load(cfg['ckpt_path'], map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state']); model.to(device)
    opt.load_state_dict(ck['optimizer_state'])
    log = ck['log']; start_epoch = ck['epoch'] + 1
    print(f'RESUMING from epoch {start_epoch}', flush=True)
else:
    print('starting fresh', flush=True)

for epoch in range(start_epoch, cfg['ssl_epochs'] + 1):
    model.train()
    t0 = time.time()
    loss_sum, n_batches = 0.0, 0
    pbar = tqdm(ssl_loader, desc=f'ep {epoch}/{cfg["ssl_epochs"]}', leave=False)
    for v1, v2, _ in pbar:
        v1 = v1.to(device); v2 = v2.to(device)
        x = torch.cat([v1, v2], dim=0)
        _, z = model(x)
        z = F.normalize(z, dim=1)
        loss = nt_xent_loss(z, temperature=cfg['temperature'])
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss.detach()); n_batches += 1
    epoch_sec = time.time() - t0
    log['epoch'].append(epoch)
    log['loss'].append(loss_sum / n_batches)
    log['epoch_sec'].append(epoch_sec)

    if epoch % cfg['knn_interval'] == 0 or epoch == 1 or epoch == cfg['ssl_epochs']:
        acc = knn_monitor(model.backbone, memory_loader, test_loader,
                          k=cfg['knn_k'], device=device)
        log['knn_epoch'].append(epoch)
        log['knn_acc'].append(acc)
        print(f'  ep {epoch:3d}  loss {log["loss"][-1]:.4f}  kNN {acc*100:.2f}%  ({epoch_sec:.1f}s)', flush=True)

    # atomic checkpoint
    tmp = cfg['ckpt_path'] + '.tmp'
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(), 'log': log,
                'config': cfg}, tmp)
    os.replace(tmp, cfg['ckpt_path'])

print(f'SSL done in {sum(log["epoch_sec"])/60:.1f} min')

# --- Linear probe
print('--- linear probe ---')
backbone = model.backbone
backbone.eval()
for p in backbone.parameters(): p.requires_grad = False

head = nn.Linear(512, 10).to(device)
probe_opt = torch.optim.Adam(head.parameters(), lr=cfg['probe_lr'],
                             weight_decay=cfg['weight_decay'])

curve = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}
best, final = 0.0, 0.0
for ep in range(1, cfg['probe_epochs'] + 1):
    head.train()
    train_loss, n_correct, n_total = 0.0, 0, 0
    for x, y in probe_train_loader:
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            h = backbone(x)
        logits = head(h)
        loss = F.cross_entropy(logits, y)
        probe_opt.zero_grad(); loss.backward(); probe_opt.step()
        train_loss += float(loss.detach()) * y.size(0)
        n_correct  += int((logits.argmax(1) == y).sum())
        n_total    += y.size(0)
    head.eval()
    n_test_correct, n_test = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            n_test_correct += int((head(backbone(x)).argmax(1) == y).sum())
            n_test         += y.size(0)
    test_acc = n_test_correct / n_test
    curve['epoch'].append(ep)
    curve['train_loss'].append(train_loss / n_total)
    curve['train_acc'].append(n_correct / n_total)
    curve['test_acc'].append(test_acc)
    final = test_acc
    best  = max(best, test_acc)
    if ep % 10 == 0 or ep == 1 or ep == cfg['probe_epochs']:
        print(f'  probe ep {ep:3d}  test {test_acc*100:.2f}%  best {best*100:.2f}%', flush=True)

result = {
    'host': str(device),
    'ssl_epochs': cfg['ssl_epochs'],
    'probe_epochs': cfg['probe_epochs'],
    'final_test_acc': final,
    'best_test_acc': best,
    'curve': curve,
    'final_loss': log['loss'][-1],
    'final_knn': log['knn_acc'][-1],
}
with open(cfg['results_path'], 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved {cfg["results_path"]}')
print(f'\n=== bs=256 verification on this host ===')
print(f'  final loss : {log["loss"][-1]:.4f}')
print(f'  final kNN  : {log["knn_acc"][-1]*100:.2f}%')
print(f'  probe best : {best*100:.2f}%')
print(f'  probe final: {final*100:.2f}%')
print()
print('Compare to existing:')
print('  ablation_batchsize.json bs=64 : 87.12%')
print('  ablation_batchsize.json bs=128: 87.27%')
print('  Phase 1 (DirectML) bs=256     : 86.70%  <-- old, different hardware')
