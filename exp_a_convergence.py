"""
exp_a_convergence.py — Experiment A: High-Tolerance Convergence

Sub-experiments
---------------
A1  σ_norm sweep: accuracy vs σ_norm curves for MNIST and CIFAR-10.
    Identifies the three regimes:
      • Noise-immune plateau   (accuracy ≈ clean)
      • Graceful degradation   (accuracy declines smoothly)
      • Catastrophic collapse  (accuracy → chance level)

A2  Architecture overlay: the three ONN operating points (σ_norm derived
    from Claim 1 physical formula) are marked on the A1 curves.
    ΔAcc between architectures is measured and compared to the Convergence
    Theorem prediction.

Output
------
    results/exp_a1_plateau.pdf   — accuracy vs σ_norm (both tasks)
    results/exp_a2_overlay.pdf   — A1 + architecture markers

Usage
-----
    python exp_a_convergence.py [--device cpu|cuda] [--n_mc 20] [--n_sigma 40]
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import (load_model, get_test_loader, sigma_grid,
                    set_plot_style, save_fig)
from noise_injection import (precompute_logits, fast_sweep_gaussian,
                              evaluate)
from arch_params import ARCHS, sigma_norm as compute_sigma_norm


# ══════════════════════════════════════════════════════════════
# A1 — σ_norm sweep (logit-cache Monte Carlo)
# ══════════════════════════════════════════════════════════════

def run_sweep(model, loader, device, sigma_values, n_mc: int):
    """
    Fast σ_norm sweep using pre-computed logits (output-referred noise).
    One forward pass caches all logits; noise is added as a tensor operation.
    Speed-up vs hook-based approach: ≈ N_sigma × N_mc × N_modes ≥ 1 000×.
    """
    print(f"    Pre-computing clean logits (1 forward pass)...")
    logits, labels, scale = precompute_logits(model, loader, device)
    print(f"    Logits cached: {tuple(logits.shape)},  scale={scale:.3f}")
    print(f"    Running {len(sigma_values)} σ pts × {n_mc} MC (tensor ops only)...")
    means, stds = fast_sweep_gaussian(logits, labels, scale, sigma_values, n_mc)
    for i, (sig, mu, sd) in enumerate(zip(sigma_values, means, stds)):
        if (i + 1) % 10 == 0 or i == 0 or i == len(sigma_values) - 1:
            print(f"    σ={sig:.2e}  acc={mu*100:.2f}% ± {sd*100:.2f}%  "
                  f"[{i+1}/{len(sigma_values)}]")
    return means, stds


# ══════════════════════════════════════════════════════════════
# Region annotation helpers
# ══════════════════════════════════════════════════════════════

def find_breakpoints(sigma_values, means, clean_acc,
                     plateau_drop: float = 0.01,
                     collapse_level: float = 0.20):
    """
    Identify:
      sigma_star  : largest σ where acc ≥ clean_acc − plateau_drop
      sigma_col   : smallest σ where acc ≤ collapse_level
    Returns (sigma_star, sigma_col) — both may be None if not found.
    """
    sigma_star = None
    for sig, acc in zip(sigma_values, means):
        if acc >= clean_acc - plateau_drop:
            sigma_star = sig

    sigma_col = None
    for sig, acc in zip(sigma_values, means):
        if acc <= collapse_level:
            sigma_col = sig
            break

    return sigma_star, sigma_col


def annotate_regions(ax, sigma_values, means, clean_acc,
                     n_classes: int = 10):
    """Shade the three qualitative regions on an axes."""
    s_star, s_col = find_breakpoints(sigma_values, means, clean_acc)
    chance = 1.0 / n_classes
    xmin, xmax = sigma_values[0], sigma_values[-1]

    if s_star is not None:
        ax.axvspan(xmin, s_star, alpha=0.06, color="#185FA5",
                   label="Noise-immune plateau")
    if s_star is not None and s_col is not None:
        ax.axvspan(s_star, s_col, alpha=0.06, color="#E8A020",
                   label="Graceful degradation")
    if s_col is not None:
        ax.axvspan(s_col, xmax, alpha=0.06, color="#D85A30",
                   label="Catastrophic collapse")

    ax.axhline(chance * 100, color="gray", ls=":", lw=1.0, label="Chance level")

    return s_star


# ══════════════════════════════════════════════════════════════
# Plot A1
# ══════════════════════════════════════════════════════════════

def plot_a1(results: dict, figsize=(9, 4)):
    """
    results: dict keyed by dataset name, each value is
             {'sigma': array, 'means': array, 'stds': array, 'clean': float}
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    dataset_cfg = {
        "mnist":   {"title": "(a) MNIST — LeNet-5",   "n_cls": 10,
                    "color": "#185FA5"},
        "cifar10": {"title": "(b) CIFAR-10 — ResNet-20", "n_cls": 10,
                    "color": "#1D9E75"},
    }

    sigma_stars = {}
    for ax, (ds, res) in zip(axes, results.items()):
        cfg    = dataset_cfg[ds]
        sigmas = res["sigma"]
        means  = res["means"]
        stds   = res["stds"]
        clean  = res["clean"]

        # Shaded regions
        s_star = annotate_regions(ax, sigmas, means, clean)
        sigma_stars[ds] = s_star

        # Mark σ* with vertical dashed line
        if s_star is not None:
            ax.axvline(s_star, color=cfg["color"], ls="--", lw=1.2, alpha=0.8)
            ax.text(s_star * 1.15, clean * 100 * 0.5,
                    rf"$\sigma^*={s_star:.1e}$",
                    color=cfg["color"], fontsize=8, rotation=90, va="center")

        # Accuracy curve with 1σ band
        ax.semilogx(sigmas, means * 100, color=cfg["color"], lw=1.8, zorder=5)
        ax.fill_between(sigmas,
                         (means - stds) * 100,
                         (means + stds) * 100,
                         alpha=0.18, color=cfg["color"])

        # Clean-accuracy reference
        ax.axhline(clean * 100, color="black", ls=":", lw=0.9, alpha=0.6)
        ax.text(sigmas[0] * 1.2, clean * 100 + 0.5,
                f"clean {clean*100:.1f}%", fontsize=8, alpha=0.7)

        ax.set_title(cfg["title"])
        ax.set_xlabel(r"Normalised noise $\sigma_\mathrm{norm}$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlim(sigmas[0], sigmas[-1])
        ax.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    return fig, sigma_stars


# ══════════════════════════════════════════════════════════════
# Plot A2 — architecture overlay
# ══════════════════════════════════════════════════════════════

def plot_a2(results: dict, figsize=(9, 4)):
    """
    Overlay the three ONN architecture operating points on the A1 curves,
    read off their accuracies, and annotate ΔAcc.
    """
    # Compute architecture σ_norm values from physical formula
    arch_sigs = {a.label: compute_sigma_norm(a) for a in ARCHS}
    print("\n  Architecture operating points:")
    for lbl, sig in arch_sigs.items():
        print(f"    {lbl:35s}  σ_norm = {sig:.3e}")

    dataset_cfg = {
        "mnist":   {"title": "(a) MNIST — LeNet-5",   "color": "#185FA5"},
        "cifar10": {"title": "(b) CIFAR-10 — ResNet-20", "color": "#1D9E75"},
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, (ds, res) in zip(axes, results.items()):
        cfg    = dataset_cfg[ds]
        sigmas = res["sigma"]
        means  = res["means"]
        stds   = res["stds"]
        clean  = res["clean"]

        # Background curve
        ax.semilogx(sigmas, means * 100, color=cfg["color"],
                    lw=1.8, zorder=3, label="Accuracy curve")
        ax.fill_between(sigmas,
                         (means - stds) * 100,
                         (means + stds) * 100,
                         alpha=0.15, color=cfg["color"])
        ax.axhline(clean * 100, color="black", ls=":", lw=0.9, alpha=0.5)

        # Architecture operating points
        arch_accs = {}
        for arch in ARCHS:
            sig_a = compute_sigma_norm(arch)
            # Interpolate accuracy at this σ
            acc_a = float(np.interp(sig_a, sigmas, means))
            arch_accs[arch.label] = acc_a
            ax.scatter([sig_a], [acc_a * 100],
                       color=arch.color, marker=arch.marker,
                       s=80, zorder=6, edgecolors="white", linewidths=0.8,
                       label=f"{arch.label}  ({acc_a*100:.1f}%)")
            ax.axvline(sig_a, color=arch.color, ls=":", lw=0.8, alpha=0.5)

        # Annotate ΔAcc
        accs_list = list(arch_accs.values())
        delta_acc = max(accs_list) - min(accs_list)
        ymin_ax = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 10.0
        ax.text(0.97, 0.08,
                rf"$\Delta\mathrm{{Acc}} = {delta_acc*100:.2f}\%$",
                transform=ax.transAxes, ha="right", fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="gray", alpha=0.8))

        ax.set_title(cfg["title"])
        ax.set_xlabel(r"Normalised noise $\sigma_\mathrm{norm}$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlim(sigmas[0], sigmas[-1])
        ax.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default=None)
    parser.add_argument("--n_mc",    type=int, default=20,
                        help="Monte Carlo repetitions per σ point")
    parser.add_argument("--n_sigma", type=int, default=40,
                        help="Number of σ_norm grid points")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    set_plot_style()

    sigmas = sigma_grid(n=args.n_sigma, lo=1e-4, hi=2e-1)

    results = {}
    for ds in ("mnist", "cifar10"):
        print(f"\n{'═'*54}")
        print(f"  Dataset: {ds.upper()}")
        print("═" * 54)

        model  = load_model(ds, device)
        loader = get_test_loader(ds)

        # Clean accuracy (no noise, n_mc=1)
        clean_acc = evaluate(model, loader, device)
        print(f"  Clean accuracy: {clean_acc*100:.2f}%")

        print(f"  Running σ_norm sweep ({args.n_sigma} pts × {args.n_mc} MC)...")
        means, stds = run_sweep(model, loader, device, sigmas, args.n_mc)

        results[ds] = {
            "sigma": sigmas,
            "means": means,
            "stds":  stds,
            "clean": clean_acc,
        }

    # ── Figure A1 ───────────────────────────────────────────
    print("\n  Plotting Exp A1...")
    fig_a1, sigma_stars = plot_a1(results)
    save_fig(fig_a1, "exp_a1_plateau")
    plt.close(fig_a1)

    # ── Figure A2 ───────────────────────────────────────────
    print("\n  Plotting Exp A2...")
    fig_a2 = plot_a2(results)
    save_fig(fig_a2, "exp_a2_overlay")
    plt.close(fig_a2)

    # ── Numerical summary ────────────────────────────────────
    print("\n  ─── Numerical Summary ───────────────────────────────")
    for ds, res in results.items():
        s_star = sigma_stars.get(ds)
        print(f"\n  {ds.upper()}")
        print(f"    Clean acc  : {res['clean']*100:.2f}%")
        print(f"    σ*_norm    : {s_star:.3e}  (tolerance threshold)")
        print(f"    Architecture operating points:")
        for arch in ARCHS:
            sig_a = compute_sigma_norm(arch)
            acc_a = float(np.interp(sig_a, res["sigma"], res["means"]))
            plateau = "✓ IN plateau" if s_star and sig_a <= s_star else "✗ OUTSIDE"
            print(f"      {arch.label:35s}  "
                  f"σ={sig_a:.2e}  acc={acc_a*100:.1f}%  {plateau}")
    print()


if __name__ == "__main__":
    main()
