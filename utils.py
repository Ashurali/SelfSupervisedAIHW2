"""Shared components for the SimCLR HW2 project.

Imported by every phase notebook. Contains:
    - CIFAR10 normalization constants
    - get_cifar_resnet18()      : modified ResNet-18 backbone (fc = Identity)
    - ProjectionHead            : 512 -> 512 -> 128 MLP
    - SimCLRModel               : backbone + projector
    - nt_xent_loss              : from-scratch NT-Xent contrastive loss
    - SimCLRPairDataset         : wraps a dataset to return (view1, view2, label)
    - get_simclr_transform      : SimCLR augmentation pipeline (CIFAR size)
    - get_eval_transform        : test-time / linear-probe transform
    - knn_monitor               : kNN classification accuracy using encoder features
    - get_device                : torch_directml device helper (+ CPU fallback)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


# --- Constants -----------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# --- Device --------------------------------------------------------------
def get_device():
    """Return the best available device.

    Priority: DirectML (this project's primary target — AMD on Windows)
    -> CUDA (for portability when running on an NVIDIA box)
    -> CPU (last resort).
    """
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- Backbone ------------------------------------------------------------
def get_cifar_resnet18() -> nn.Module:
    """Modified ResNet-18 for 32x32 inputs.

    - conv1 : 3x3 stride=1 padding=1  (replaces the default 7x7 stride=2)
    - maxpool: Identity               (removes the first 2x downsample)
    - fc    : Identity                (caller attaches projector / linear head)
    Output feature dim = 512.
    """
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


# --- Projector -----------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """Backbone + projector. forward(x) returns (h, z)."""

    def __init__(self, backbone: nn.Module | None = None,
                 hidden_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone if backbone is not None else get_cifar_resnet18()
        self.projector = ProjectionHead(512, hidden_dim, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


# --- NT-Xent loss --------------------------------------------------------
def nt_xent_loss(features: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (normalized temperature-scaled cross-entropy) loss.

    features : (2N, D) — concatenation of two augmented views; pair i pairs with i+N.
    Positive mates at indices (i, i+N) and (i+N, i).
    """
    n2 = features.shape[0]
    if n2 % 2:
        raise ValueError("features must have even first dim (2N)")
    n = n2 // 2

    z = F.normalize(features, dim=1)
    sim = z @ z.t() / temperature                  # (2N, 2N)

    # Mask self-similarity (diagonal) from the logits.
    # NB: torch.eye falls back to CPU on DirectML and can return a 0-length
    # tensor, so build the mask on CPU and move it.
    mask = torch.eye(n2, dtype=torch.bool)
    # Use a large negative instead of -inf: DirectML produces NaN with -inf here.
    sim = sim.masked_fill(mask.to(sim.device), -1e9)

    # Positive index for each row: (i + N) mod 2N
    targets = (torch.arange(n2) + n) % n2
    targets = targets.to(sim.device)

    return F.cross_entropy(sim, targets)


# --- Augmentations -------------------------------------------------------
def get_simclr_transform(image_size: int = 32,
                         color_jitter_strength: float = 0.4,
                         use_color_jitter: bool = True,
                         use_grayscale: bool = True) -> transforms.Compose:
    s = color_jitter_strength
    ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_color_jitter:
        ops.append(transforms.RandomApply(
            [transforms.ColorJitter(s, s, s, s * 0.25)], p=0.8))
    if use_grayscale:
        ops.append(transforms.RandomGrayscale(p=0.2))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


def get_eval_transform(image_size: int = 32) -> transforms.Compose:
    ops = []
    if image_size != 32:
        ops.append(transforms.Resize((image_size, image_size)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


# --- Paired-view dataset -------------------------------------------------
class SimCLRPairDataset(Dataset):
    """Wraps a torchvision-style dataset so __getitem__ returns (view1, view2, label).

    Expects the base dataset to yield PIL images (i.e. construct it without a
    transform). Labels are only used for kNN monitoring, not for SSL training.
    """

    def __init__(self, base_dataset, transform: transforms.Compose):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), self.transform(img), label


# --- kNN monitor ---------------------------------------------------------
@torch.no_grad()
def knn_monitor(backbone: nn.Module,
                memory_loader,
                test_loader,
                device,
                k: int = 20,
                num_classes: int = 10,
                temperature: float = 0.1) -> float:
    """kNN classification accuracy (weighted by exp(sim/temperature)).

    memory_loader : DataLoader over training images with EVAL transform (no aug).
    Returns top-1 accuracy in [0, 1].
    """
    backbone.eval()

    feats, labels = [], []
    for x, y in memory_loader:
        x = x.to(device)
        h = backbone(x)
        h = F.normalize(h, dim=1)
        feats.append(h.cpu())
        labels.append(y)
    memory_feats = torch.cat(feats)            # (Nm, D)
    memory_labels = torch.cat(labels)          # (Nm,)

    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        h = backbone(x)
        h = F.normalize(h, dim=1).cpu()

        sim = h @ memory_feats.t()             # (B, Nm)
        top_sim, top_idx = sim.topk(k=k, dim=1)
        top_labels = memory_labels[top_idx]    # (B, k)

        weights = (top_sim / temperature).exp()
        # one-hot over classes, weighted sum
        one_hot = F.one_hot(top_labels, num_classes).float()  # (B, k, C)
        scores = (one_hot * weights.unsqueeze(-1)).sum(dim=1) # (B, C)
        pred = scores.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total
