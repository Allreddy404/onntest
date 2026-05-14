"""
exp_b_ccenob.py — Experiment B: CC-ENOB Analysis

Sub-experiments
---------------
B1  ENOB(k) curves for the three ONN architectures over k ∈ [0.1, 10].
    Key axis direction: high-tolerance tasks → SMALL k (left side).
    Shows that the RELATIVE ENOB gap shrinks as k → 0.

B2  Task k* extraction: back-calculate the effective operating k* for MNIST
    and CIFAR-10 from the σ*_norm values obtained in Exp A, then mark these
    on the B1 figure to confirm they reside in the small-k convergence zone.

All computations in this experiment are ANALYTICAL (no neural network
simulation required).  The σ_norm values for each architecture are derived
from Claim 1; the k* values for each task are derived from Exp A outputs.

Output
------
    results/exp_b1_ccenob.pdf   — ENOB(k) + C(k) for 3 architectures
    results/exp_b2_taskmarker.pdf — B1 annotated with task k* points

Usage
-----
    python exp_b_ccenob.py [--sigma_star_mnist S] [--sigma_star_cifar S]

If --sigma_star_* is not supplied the script uses conservative defaults.
Override with the σ*_norm values printed by exp_a_convergence.py.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import erf

from common import (set_plot_style, save_fig, cc_enob, confidence,
                    overflow_prob, delta_quant_theory, classic_enob)
from arch_params import ARCHS, sigma_norm as compute_sigma_norm


# ══════════════════════════════════════════════════════════════
# B1 — ENOB(k) curves
# ══════════════════════════════════════════════════════════════

def plot_b1(k_values: np.ndarray,
            figsize=(8, 5),
            n_bit_refs=(4, 6, 8)) -> plt.Figure:
    """
    Two-panel figure:
      Left  — ENOB(k) for 3 architectures, with relative gap annotation.
      Right — Confidence C(k) and overflow probability P_ov(k).

    High-tolerance region = small k (LEFT side of x-axis).
    """
    fig, (ax_enob, ax_conf) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: ENOB curves ───────────────────────────────────
    for arch in ARCHS:
        sig = compute_sigma_norm(arch)
        enob_vals = cc_enob(k_values, sig)
        ax_enob.semilogx(k_values, enob_vals,
                         color=arch.color, lw=1.8, label=arch.label)

    # Reference digital bit widths (horizontal dashed lines)
    for nb in n_bit_refs:
        ax_enob.axhline(nb, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax_enob.text(k_values[-1] * 0.85, nb + 0.15, f"{nb}-bit",
                     fontsize=8, color="gray", ha="right")

    # Annotate the RELATIVE ENOB gap at a representative small k
    k_annot = 1.0
    sigs = [compute_sigma_norm(a) for a in ARCHS]
    enobs_at_k = [float(cc_enob(k_annot, s)) for s in sigs]
    delta_abs = abs(enobs_at_k[0] - enobs_at_k[-1])
    delta_rel = delta_abs / abs(enobs_at_k[0])
    ax_enob.annotate("",
                     xy=(k_annot, enobs_at_k[-1]),
                     xytext=(k_annot, enobs_at_k[0]),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax_enob.text(k_annot * 1.4,
                 (enobs_at_k[0] + enobs_at_k[-1]) / 2,
                 rf"$\Delta$ENOB={delta_abs:.1f} bit"
                 f"\n({delta_rel*100:.0f}% rel.)",
                 fontsize=8, va="center")

    ax_enob.set_xlabel(r"Confidence parameter $k$  "
                       r"[$\leftarrow$ high tolerance / low tolerance $\rightarrow$]")
    ax_enob.set_ylabel("CC-ENOB (bits)")
    ax_enob.set_title("(a) CC-ENOB vs $k$ for three ONN architectures")
    ax_enob.legend(fontsize=8, loc="upper right")
    ax_enob.set_xlim(k_values[0], k_values[-1])
    ax_enob.set_ylim(0, None)

    # Shade the high-tolerance (small-k) convergence region
    k_thresh = 0.5
    ax_enob.axvspan(k_values[0], k_thresh, alpha=0.07,
                    color="#185FA5", label="Convergence zone (small k)")
    ax_enob.text(np.sqrt(k_values[0] * k_thresh), ax_enob.get_ylim()[1] * 0.92,
                 "Convergence\nzone", fontsize=7.5, ha="center",
                 color="#185FA5", alpha=0.85)

    # ── Right: Confidence and overflow ──────────────────────
    C    = confidence(k_values) * 100
    P_ov = overflow_prob(k_values) * 100

    ax_conf.semilogx(k_values, C,    color="#185FA5", lw=1.8,
                     label=r"$\mathcal{C}(k)=\mathrm{erf}(k/\sqrt{2})$")
    ax_conf.semilogx(k_values, P_ov, color="#D85A30", lw=1.8, ls="--",
                     label=r"$P_\mathrm{ov}(k)=1-\mathcal{C}(k)$")

    ax_conf.axhline(50, color="gray", ls=":", lw=0.8)
    ax_conf.text(k_values[-1] * 0.85, 51, "50%", fontsize=8,
                 color="gray", ha="right")

    ax_conf.set_xlabel(r"Confidence parameter $k$")
    ax_conf.set_ylabel("Probability (%)")
    ax_conf.set_title(r"(b) Confidence $\mathcal{C}(k)$ and overflow $P_\mathrm{ov}(k)$")
    ax_conf.legend(fontsize=8, loc="center")
    ax_conf.set_xlim(k_values[0], k_values[-1])
    ax_conf.set_ylim(0, 100)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# B2 — Task k* markers on B1
# ══════════════════════════════════════════════════════════════

def sigma_star_to_k_star(sigma_star: float, arch_sigma: float) -> float:
    """
    Given the tolerance threshold σ*_norm (from Exp A), compute the effective
    operating confidence level k* for that task.

    Rationale: at the plateau boundary, the task can tolerate a normalised
    noise of σ*_norm.  The effective k* is defined as the k value at which
    the CC-ENOB computed for the BEST architecture (Arch-A, lowest σ) equals
    the CC-ENOB computed for the WORST architecture (Arch-C) at the same k.

    Simpler operational definition used here:
        k* = σ*_norm / arch_sigma
    This gives the number of σ's that the tolerance threshold represents
    relative to the best architecture's noise level.  A small k* confirms
    that the task operates in the high-tolerance / convergence regime.
    """
    return sigma_star / arch_sigma


def plot_b2(k_values: np.ndarray,
            task_info: dict,
            figsize=(8, 5)) -> plt.Figure:
    """
    Reproduce the B1 ENOB panel and overlay task k* markers.

    task_info: dict mapping task_name → {'sigma_star': float, 'color': str,
                                          'marker': str, 'label': str}
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Architecture ENOB curves
    for arch in ARCHS:
        sig = compute_sigma_norm(arch)
        enob_vals = cc_enob(k_values, sig)
        ax.semilogx(k_values, enob_vals,
                    color=arch.color, lw=1.6, label=arch.label)

    # High-tolerance shading
    ax.axvspan(k_values[0], 0.5, alpha=0.07, color="#185FA5")
    ax.text(np.sqrt(k_values[0] * 0.5), ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 15,
            "Convergence\nzone", fontsize=8, ha="center", color="#185FA5", alpha=0.8)

    # Task k* markers
    arch_sigs = [compute_sigma_norm(a) for a in ARCHS]
    best_sig  = min(arch_sigs)    # Arch-A (lowest noise)
    worst_sig = max(arch_sigs)    # Arch-C (highest noise)

    print("\n  Task k* values (operating confidence level):")
    for task_name, info in task_info.items():
        s_star = info["sigma_star"]
        # For each architecture, interpolate ENOB at k* = s_star / arch_sigma
        # We mark where on the k-axis the task tolerance σ* sits relative to
        # each architecture's σ_norm.  For the overlay, we use the best arch.
        k_star = s_star / best_sig
        print(f"    {task_name:10s}  σ*={s_star:.2e}  "
              f"k*={k_star:.3f} (vs Arch-A)  "
              f"{'[convergence zone]' if k_star < 0.5 else '[degradation zone]'}")

        # Mark k* vertical line and annotate ENOB at this k* for each arch
        ax.axvline(k_star, color=info["color"], ls=":", lw=1.2, alpha=0.7)
        enob_best  = float(cc_enob(k_star, best_sig))
        enob_worst = float(cc_enob(k_star, worst_sig))
        delta = abs(enob_best - enob_worst)
        ax.annotate(f"$k^*_\\mathrm{{{task_name}}}={k_star:.2f}$\n"
                    rf"$\Delta$ENOB={delta:.1f} bit",
                    xy=(k_star, (enob_best + enob_worst) / 2),
                    xytext=(k_star * 2.5,  (enob_best + enob_worst) / 2 - 1.5),
                    fontsize=8, color=info["color"],
                    arrowprops=dict(arrowstyle="->", color=info["color"],
                                   lw=0.8))

    ax.set_xlabel(r"Confidence parameter $k$  "
                  r"[$\leftarrow$ high tolerance / low tolerance $\rightarrow$]")
    ax.set_ylabel("CC-ENOB (bits)")
    ax.set_title("CC-ENOB for three architectures with task operating points")
    ax.legend(fontsize=8)
    ax.set_xlim(k_values[0], k_values[-1])
    ax.set_ylim(0, None)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Relative ENOB gap vs k  (supplementary analysis)
# ══════════════════════════════════════════════════════════════

def plot_relative_gap(k_values: np.ndarray, figsize=(6, 4)) -> plt.Figure:
    """
    Plot the relative ENOB gap (ΔAbs / ENOB_best) vs k.
    Converges to 0 as k → 0 (high tolerance), demonstrating the theorem.
    """
    sigs = [compute_sigma_norm(a) for a in ARCHS]
    enob_a = cc_enob(k_values, min(sigs))
    enob_c = cc_enob(k_values, max(sigs))

    delta_abs = enob_a - enob_c          # constant in k (= log₂(σ_C/σ_A))
    delta_rel = delta_abs / enob_a       # → 0 as k → 0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].semilogx(k_values, delta_abs, color="#185FA5", lw=1.8)
    axes[0].axhline(delta_abs[0], color="gray", ls="--", lw=0.9)
    axes[0].set_xlabel(r"$k$")
    axes[0].set_ylabel("ENOB gap $|$ENOB$_A$ − ENOB$_C|$ (bits)")
    axes[0].set_title("(a) Absolute gap — constant in $k$")
    axes[0].set_xlim(k_values[0], k_values[-1])

    axes[1].semilogx(k_values, delta_rel * 100, color="#D85A30", lw=1.8)
    axes[1].axvspan(k_values[0], 0.5, alpha=0.07, color="#185FA5")
    axes[1].set_xlabel(r"$k$  [$\leftarrow$ high tolerance]")
    axes[1].set_ylabel("Relative ENOB gap (%)")
    axes[1].set_title("(b) Relative gap → 0  as $k \\to 0$")
    axes[1].set_xlim(k_values[0], k_values[-1])
    axes[1].set_ylim(0, None)

    fig.suptitle("ENOB gap between Arch-A and Arch-C", y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_star_mnist",   type=float, default=None,
                        help="σ*_norm for MNIST from Exp A (default: estimate)")
    parser.add_argument("--sigma_star_cifar10", type=float, default=None,
                        help="σ*_norm for CIFAR-10 from Exp A (default: estimate)")
    args = parser.parse_args()

    set_plot_style()

    # k-axis: high tolerance (small k) on the left
    k_values = np.logspace(-1.5, 1.0, 300)    # k ∈ [~0.03, 10]

    # ── Print σ_norm for all architectures ──────────────────
    print("\n  Architecture σ_norm values:")
    for arch in ARCHS:
        sig = compute_sigma_norm(arch)
        enob_k1 = float(cc_enob(1.0, sig))
        print(f"    {arch.label:35s}  σ={sig:.3e}  "
              f"ENOB(k=1)={enob_k1:.2f} bit")

    # ── Figure B1 ───────────────────────────────────────────
    print("\n  Plotting Exp B1 (CC-ENOB curves)...")
    fig_b1 = plot_b1(k_values)
    save_fig(fig_b1, "exp_b1_ccenob")
    plt.close(fig_b1)

    # ── Relative gap analysis ────────────────────────────────
    print("\n  Plotting ENOB gap analysis...")
    fig_gap = plot_relative_gap(k_values)
    save_fig(fig_gap, "exp_b1_relative_gap")
    plt.close(fig_gap)

    # ── Task k* extraction ───────────────────────────────────
    # Use values from Exp A if provided, else conservative defaults
    best_sig = min(compute_sigma_norm(a) for a in ARCHS)

    sigma_star_mnist   = args.sigma_star_mnist   or best_sig * 0.3
    sigma_star_cifar10 = args.sigma_star_cifar10 or best_sig * 0.1

    task_info = {
        "MNIST": {
            "sigma_star": sigma_star_mnist,
            "color":      "#185FA5",
            "marker":     "o",
            "label":      f"MNIST  (σ*={sigma_star_mnist:.2e})",
        },
        "CIFAR-10": {
            "sigma_star": sigma_star_cifar10,
            "color":      "#1D9E75",
            "marker":     "s",
            "label":      f"CIFAR-10  (σ*={sigma_star_cifar10:.2e})",
        },
    }

    print("\n  Plotting Exp B2 (task k* overlay)...")
    fig_b2 = plot_b2(k_values, task_info)
    save_fig(fig_b2, "exp_b2_taskmarker")
    plt.close(fig_b2)

    # ── Numerical table: ENOB at task k* for each architecture ──
    print("\n  ─── ENOB at task k* for each architecture ──────────")
    for task_name, info in task_info.items():
        s_star = info["sigma_star"]
        print(f"\n  {task_name}  (σ*={s_star:.2e})")
        enob_vals = []
        for arch in ARCHS:
            sig   = compute_sigma_norm(arch)
            k_eff = s_star / sig     # effective k at this architecture
            enob_val = float(cc_enob(k_eff, sig))
            enob_vals.append(enob_val)
            print(f"    {arch.label:35s}  k_eff={k_eff:.3f}  "
                  f"ENOB={enob_val:.2f} bit")
        print(f"    Max ENOB gap across architectures: "
              f"{max(enob_vals)-min(enob_vals):.2f} bit")
    print()


if __name__ == "__main__":
    main()
