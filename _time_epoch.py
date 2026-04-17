"""Time a single training epoch of SimCLR on DirectML."""
import time, torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils import (get_device, SimCLRModel, nt_xent_loss,
                   SimCLRPairDataset, get_simclr_transform)

device = get_device()
train_base = CIFAR10('./data', train=True, download=False, transform=None)
ds = SimCLRPairDataset(train_base, get_simclr_transform(32))
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
model = SimCLRModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

# Warm-up a few batches then time 20
it = iter(loader)
for _ in range(3):
    v1, v2, _ = next(it); v1, v2 = v1.to(device), v2.to(device)
    _, z1 = model(v1); _, z2 = model(v2)
    loss = nt_xent_loss(torch.cat([z1,z2]), 0.5)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print('warmed up')

N = 20
t0 = time.time()
for _ in range(N):
    v1, v2, _ = next(it); v1, v2 = v1.to(device), v2.to(device)
    _, z1 = model(v1); _, z2 = model(v2)
    loss = nt_xent_loss(torch.cat([z1,z2]), 0.5)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
dt = time.time() - t0
sec_per_batch = dt / N
batches_per_epoch = len(loader)
print(f'sec/batch = {sec_per_batch:.3f}  batches/epoch = {batches_per_epoch}')
print(f'estimated epoch time = {sec_per_batch*batches_per_epoch:.1f} s = {sec_per_batch*batches_per_epoch/60:.2f} min')
print(f'estimated 200-epoch run = {sec_per_batch*batches_per_epoch*200/3600:.2f} h')
