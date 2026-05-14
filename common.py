"""
common.py — Shared utilities for all inference experiments.

Provides:
  load_model()         — load pretrained weights from weights/ directory
  get_test_loader()    — test DataLoader for MNIST or CIFAR-10
  set_plot_style()     — apply publication-quality matplotlib style
  save_fig()           — save figure to results/ with tight layout
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models import LeNet5, ResNet20

WEIGHTS_DIR = "weights"
DATA_DIR    = "{weights,results,data}/data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════

_MODEL_REGISTRY = {
    "mnist":   (LeNet5,   "lenet5_mnist.pth"),
    "cifar10": (ResNet20, "resnet20_cifar10.pth"),
}


def load_model(dataset: str, device: torch.device) -> nn.Module:
    """
    Load a pretrained model from weights/ directory.

    Parameters
    ----------
    dataset : "mnist" or "cifar10"
    device  : torch device

    Returns
    -------
    model in eval mode on the specified device.
    """
    cls, fname = _MODEL_REGISTRY[dataset]
    path = os.path.join(WEIGHTS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Weights not found at {path}. "
            f"Run  python train.py --dataset {dataset}  first."
        )
    ckpt = torch.load(path, map_location=device)
    model = cls()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    clean_acc = ckpt.get("test_acc", float("nan"))
    print(f"  Loaded {cls.__name__} ({dataset.upper()})  "
          f"clean acc = {clean_acc*100:.2f}%  [{path}]")
    return model


# ══════════════════════════════════════════════════════════════
# Data loaders (test split only — no augmentation)
# ══════════════════════════════════════════════════════════════

def get_test_loader(dataset: str, batch_size: int = 512,
                    num_workers: int = 2) -> DataLoader:
    """Return the test DataLoader for the specified dataset."""
    if dataset == "mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

    elif dataset == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ══════════════════════════════════════════════════════════════
# Matplotlib publication style
# ══════════════════════════════════════════════════════════════

def set_plot_style():
    """Apply a clean, publication-quality style to all subsequent figures."""
    matplotlib.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,

        # Lines & markers
        "lines.linewidth":    1.5,
        "lines.markersize":   6,
        "lines.markeredgewidth": 0.8,

        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.6,

        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


def save_fig(fig: plt.Figure, name: str):
    """Save figure to results/<name>.pdf and results/<name>.png."""
    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════
# Sigma grid for sweep experiments
# ══════════════════════════════════════════════════════════════

def sigma_grid(n: int = 40, lo: float = 1e-4, hi: float = 2e-1) -> np.ndarray:
    """Log-uniform σ_norm grid for Experiment A sweeps."""
    return np.logspace(np.log10(lo), np.log10(hi), n)


# ══════════════════════════════════════════════════════════════
# CC-ENOB helpers (analytical, no simulation)
# ══════════════════════════════════════════════════════════════

def cc_enob(k: float | np.ndarray,
            sigma_norm: float | np.ndarray) -> np.ndarray:
    """
    CC-ENOB(k) = −½ log₂(k² · σ²_norm)

    High-tolerance task ↔ small k (large overflow probability accepted).
    """
    k  = np.asarray(k,          dtype=float)
    s  = np.asarray(sigma_norm, dtype=float)
    return -0.5 * np.log2(k**2 * s**2)


def confidence(k: float | np.ndarray) -> np.ndarray:
    """C(k) = erf(k / √2)  — probability that |ε| ≤ k·σ."""
    from scipy.special import erf
    return erf(np.asarray(k) / np.sqrt(2.0))


def overflow_prob(k: float | np.ndarray) -> np.ndarray:
    """P_ov(k) = 1 − C(k)  — overflow probability."""
    return 1.0 - confidence(k)


def delta_quant_theory(k: float | np.ndarray) -> np.ndarray:
    """
    Theoretical digitisation bit-penalty (eq. delta_quant):

        Δ_quant(k) = ½ log₂(1 + 12 · erfc(k/√2))

    Equals ≈ 1.85 bits at k→0 (high tolerance);
    decays to 0 as k→∞ (low tolerance, ADC cost negligible).
    """
    from scipy.special import erfc
    return 0.5 * np.log2(1.0 + 12.0 * erfc(np.asarray(k) / np.sqrt(2.0)))


def classic_enob(sigma_norm: float | np.ndarray) -> np.ndarray:
    """Classic ENOB = ½ log₂(1 / (12 σ²_norm))  (k = 1/√12 reference)."""
    s = np.asarray(sigma_norm, dtype=float)
    return 0.5 * np.log2(1.0 / (12.0 * s**2))
