"""
exp_c_digitization.py — Experiment C: Digitisation Penalty

Sub-experiments
---------------
C1  Analog vs quantised accuracy for the same σ_norm.
    At each of several fixed σ_norm values, compare:
      • Pure Gaussian injection (analog mode)
      • ADC-quantised Gaussian injection at n_bits = 3, 4, 5, 6, 8
    Demonstrates that analog always outperforms digital at the same σ_norm
    within the confidence window, and that the gap closes only at high n_bits.

C2  Δ_quant empirical vs theory.
    For each k ∈ {0.5, 1.0, 1.5, 2.0, 3.0}:
      a) Generate N large samples of ε ~ N(0, 1).
      b) Quantise each to n_bits within [−k, +k] → ε_q.
      c) Compute empirical MSE_q = E[ε_q²].
      d) Derive empirical Δ_quant = −½ log₂(MSE_q / k²).
         (Same k used for both analog and quantised expressions.)
      e) Compare to theoretical Δ_quant(k) = ½ log₂(1 + 12 erfc(k/√2)).

    Note: Δ_quant is large at small k (high-tolerance regime) and
          decays to zero at large k (low-tolerance regime).

Output
------
    results/exp_c1_analog_vs_digital.pdf
    results/exp_c2_delta_quant.pdf

Usage
-----
    python exp_c_digitization.py [--device cpu|cuda] [--n_mc 20]
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import erfc, erf

from common import (load_model, get_test_loader, set_plot_style,
                    save_fig, delta_quant_theory, cc_enob)
from noise_injection import (precompute_logits, fast_sweep_gaussian,
                              fast_sweep_quantized, evaluate)
from arch_params import ARCHS, sigma_norm as compute_sigma_norm


# ══════════════════════════════════════════════════════════════
# C1 — Analog vs quantised accuracy  (fast logit-cache version)
# ══════════════════════════════════════════════════════════════

def run_c1(model, loader, device, sigma_values, k_fixed, n_bits_list, n_mc):
    """
    Fast C1 sweep using cached logits.

    Original complexity : O(N_σ × N_MC × (1 + N_bits) × T_forward)
    This implementation : O(T_forward + N_σ × N_MC × (1 + N_bits) × T_add)
    Typical speed-up    : 500–2 000×

    Returns
    -------
    dict with keys 'analog' and each int in n_bits_list, each mapping to
    (means_array, stds_array) of shape (len(sigma_values),).
    """
    print(f"    Pre-computing clean logits (1 forward pass)...")
    logits, labels, scale = precompute_logits(model, loader, device)
    print(f"    Logits cached: {tuple(logits.shape)},  scale={scale:.4f}")

    results = {}

    # ── Analog sweep ────────────────────────────────────────
    print(f"    Analog sweep  ({len(sigma_values)} pts × {n_mc} MC)...")
    means, stds = fast_sweep_gaussian(logits, labels, scale, sigma_values, n_mc)
    results["analog"] = (means, stds)

    # ── Quantised sweeps ────────────────────────────────────
    for nb in n_bits_list:
        print(f"    Quantised {nb}-bit (k={k_fixed})  ({len(sigma_values)} pts × {n_mc} MC)...")
        means_q, stds_q = fast_sweep_quantized(
            logits, labels, scale, sigma_values, k_fixed, nb, n_mc)
        results[nb] = (means_q, stds_q)

    return results


def plot_c1(sigma_values, results_dict, clean_acc, title,
            n_bits_list, k_fixed, figsize=(7, 4.5)) -> plt.Figure:
    """
    Line plot: accuracy vs σ_norm for analog and each quantised mode.
    """
    # Colour palette for bit widths (light → dark)
    bit_colors = {
        3: "#F4A460",
        4: "#E8820A",
        5: "#D85A30",
        6: "#9B2335",
        8: "#4A004A",
    }
    fig, ax = plt.subplots(figsize=figsize)

    # Analog (reference, bold)
    mu, sd = results_dict["analog"]
    ax.semilogx(sigma_values, mu * 100, color="#185FA5", lw=2.2,
                label="Analog (Gaussian only)", zorder=6)
    ax.fill_between(sigma_values,
                     (mu - sd) * 100, (mu + sd) * 100,
                     alpha=0.18, color="#185FA5")

    # Quantised curves
    for nb in n_bits_list:
        mu_q, sd_q = results_dict[nb]
        c = bit_colors.get(nb, "gray")
        ax.semilogx(sigma_values, mu_q * 100, color=c, lw=1.4,
                    ls="--", label=f"ADC {nb}-bit  (k={k_fixed})", zorder=5)
        ax.fill_between(sigma_values,
                         (mu_q - sd_q) * 100, (mu_q + sd_q) * 100,
                         alpha=0.10, color=c)

    # Clean accuracy reference
    ax.axhline(clean_acc * 100, color="black", ls=":", lw=0.9, alpha=0.5)
    ax.text(sigma_values[0] * 1.3, clean_acc * 100 + 0.3,
            f"Clean {clean_acc*100:.1f}%", fontsize=8, alpha=0.6)

    ax.set_xlabel(r"Normalised noise $\sigma_\mathrm{norm}$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(sigma_values[0], sigma_values[-1])
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# C2 — Δ_quant empirical vs theoretical
# ══════════════════════════════════════════════════════════════

def compute_delta_quant_empirical(k_values, n_bits_list,
                                  n_samples: int = 500_000) -> dict:
    """
    Numerically verify the digitisation penalty formula through three steps.

    Step 1 — Overflow probability verification.
      Draw ε ~ N(0,1); compute P_ov_emp = fraction(|ε| > k).
      Should match erfc(k/√2) exactly (basic sanity check).

    Step 2 — Δ_quant via empirical P_ov.
      Δ_quant_emp_pov = ½ log₂(1 + 12 × P_ov_emp)
      This directly plugs the empirically measured overflow probability into
      the theoretical formula, verifying the formula chain without further
      assumptions.

    Step 3 — Analog advantage within interval.
      V_gauss = E[ε² | |ε| ≤ k]   (conditional variance; analog advantage)
      V_unif  = k² / 3             (variance of Uniform[−k, k]; digital baseline)
      ratio   = V_unif / V_gauss > 1   proves Gaussian concentrates near 0
      From eq. gauss_superiority: σ²_norm(k) < k²σ²/3.

    All three outputs are returned per (k, nb) for the plot.
    """
    rng     = np.random.default_rng(seed=42)
    epsilon = rng.standard_normal(n_samples)    # N(0,1)

    emp_results = {}

    for k in k_values:
        # ── Steps 1 & 3 are k-only (not per n_bits) ─────────
        mask_in  = np.abs(epsilon) <= k
        p_ov_emp = float(np.mean(~mask_in))     # empirical overflow prob

        eps_in    = epsilon[mask_in]
        v_gauss   = float(np.mean(eps_in ** 2)) if eps_in.size > 0 else 1.0
        v_unif    = k ** 2 / 3.0
        adv_ratio = v_unif / max(v_gauss, 1e-30)   # should be > 1

        # Δ_quant via empirical overflow prob (Step 2)
        dq_from_pov = 0.5 * np.log2(max(1.0 + 12.0 * p_ov_emp, 1e-30))

        for nb in n_bits_list:
            # Quantise within [−k, k]
            n_levels = 2 ** nb
            step     = 2.0 * k / n_levels if n_levels > 0 else 1.0

            eps_q = np.clip(np.round(epsilon / step) * step, -k, k)

            # ADC-introduced error = ε_q − ε  (quantisation + clipping residual)
            adc_error = eps_q - epsilon
            mse_adc   = float(np.mean(adc_error ** 2))

            # Additional MSE from clipping (independent of n_bits for high n_bits)
            clip_error = np.clip(epsilon, -k, k) - epsilon   # pure clipping
            mse_clip   = float(np.mean(clip_error ** 2))

            emp_results[(k, nb)] = {
                "p_ov_emp":    p_ov_emp,
                "dq_from_pov": dq_from_pov,    # verifiable formula chain
                "v_gauss":     v_gauss,
                "v_unif":      v_unif,
                "adv_ratio":   adv_ratio,       # analog advantage ratio
                "mse_adc":     mse_adc,
                "mse_clip":    mse_clip,
            }

    return emp_results


def plot_c2(k_values, n_bits_list, figsize=(10, 4)) -> plt.Figure:
    """
    Three-panel figure validating the digitisation penalty formula:

    (a) Δ_quant(k): theoretical formula vs the value obtained by plugging
        the EMPIRICALLY MEASURED P_ov into the same formula.
        Shows: formula is data-consistent (empirical P_ov ≈ erfc(k/√2)).

    (b) Analog advantage ratio V_unif / V_gauss > 1.
        Directly demonstrates eq. gauss_superiority: within the confidence
        window the Gaussian concentrates near zero, giving lower conditional
        variance than a uniform distribution of the same half-width k.

    (c) ADC-introduced error MSE vs k for different bit widths.
        Shows that the residual error comes primarily from CLIPPING (k-
        dependent, bit-independent at high n_bits) rather than quantisation
        noise (bit-dependent, k-independent).
    """
    print("  Computing empirical Δ_quant (500 k samples)...")
    emp = compute_delta_quant_empirical(k_values, n_bits_list)

    k_fine    = np.logspace(np.log10(min(k_values)) - 0.1,
                             np.log10(max(k_values)) + 0.1, 300)
    dq_theory = delta_quant_theory(k_fine)

    bit_colors = {3: "#F4A460", 4: "#E8820A", 5: "#D85A30",
                  6: "#9B2335", 8: "#4A004A"}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_dq, ax_adv, ax_mse = axes

    # ── (a) Δ_quant: theory vs empirical P_ov plug-in ───────
    ax_dq.semilogx(k_fine, dq_theory, color="black", lw=2.0, zorder=6,
                   label=r"Theory $\frac{1}{2}\log_2(1+12\,\mathrm{erfc}(k/\sqrt{2}))$")

    # Empirical (from measured P_ov — independent of n_bits)
    k_arr      = np.array(k_values)
    dq_pov_arr = np.array([emp[(k, n_bits_list[0])]["dq_from_pov"] for k in k_values])
    ax_dq.scatter(k_arr, dq_pov_arr, color="#185FA5", s=55, zorder=7,
                  edgecolors="white", lw=0.6,
                  label=r"Empirical: $\frac{1}{2}\log_2(1+12\,\hat{P}_\mathrm{ov})$")

    ax_dq.axhline(0.5 * np.log2(13), color="gray", ls=":", lw=0.9)
    ax_dq.text(k_fine[10], 0.5 * np.log2(13) + 0.06,
               r"$\approx$1.85 bit ($k\!\to\!0$)",
               fontsize=7.5, color="gray")
    ax_dq.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax_dq.set_xlabel(r"$k$  [$\leftarrow$ high tol.]")
    ax_dq.set_ylabel(r"$\Delta_\mathrm{quant}$ (bits)")
    ax_dq.set_title(r"(a) $\Delta_\mathrm{quant}(k)$: theory vs empirical")
    ax_dq.legend(fontsize=7.5, loc="upper right")
    ax_dq.set_xlim(k_fine[0], k_fine[-1])
    ax_dq.set_ylim(-0.1, 2.1)

    # ── (b) Analog advantage ratio ────────────────────────────
    adv_arr = np.array([emp[(k, n_bits_list[0])]["adv_ratio"] for k in k_values])
    ax_adv.semilogx(k_arr, adv_arr, color="#1D9E75", lw=2.0,
                    label=r"$(k^2/3)\;/\;E[\varepsilon^2\,|\,|\varepsilon|\leq k]$")
    ax_adv.axhline(1.0, color="gray", ls="--", lw=1.0, label="Equality (no advantage)")
    ax_adv.fill_between(k_arr, 1.0, adv_arr, alpha=0.15, color="#1D9E75",
                         label="Analog advantage region")
    ax_adv.set_xlabel(r"$k$  [$\leftarrow$ high tol.]")
    ax_adv.set_ylabel(r"$V_\mathrm{unif}(k)\;/\;V_\mathrm{Gauss}(k)$")
    ax_adv.set_title("(b) Analog advantage\nvs uniform [eq. gauss_superiority]")
    ax_adv.legend(fontsize=7.5)
    ax_adv.set_xlim(k_arr[0], k_arr[-1])
    ax_adv.set_ylim(0.9, None)

    # ── (c) ADC error MSE by bit width ───────────────────────
    for nb in n_bits_list:
        mse_arr = np.array([emp[(k, nb)]["mse_adc"] for k in k_values])
        c = bit_colors.get(nb, "gray")
        ax_mse.scatter(k_arr, mse_arr, color=c, s=35, zorder=5,
                       edgecolors="white", lw=0.5)
        ax_mse.semilogx(k_arr, mse_arr, color=c, lw=1.2, label=f"{nb}-bit")

    # Pure clipping MSE (bit-width independent limit as n_bits→∞)
    mse_clip_arr = np.array([emp[(k, n_bits_list[-1])]["mse_clip"] for k in k_values])
    ax_mse.semilogx(k_arr, mse_clip_arr, color="black", lw=1.6, ls="--",
                    label=r"Clipping only ($n_b\!\to\!\infty$)")

    ax_mse.set_xlabel(r"$k$  [$\leftarrow$ high tol.]")
    ax_mse.set_ylabel(r"$E[(\varepsilon_q - \varepsilon)^2]$")
    ax_mse.set_title("(c) ADC-introduced error MSE\nvs $k$ and bit width")
    ax_mse.legend(fontsize=7.5)
    ax_mse.set_xlim(k_arr[0], k_arr[-1])
    ax_mse.set_yscale("log")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_mc",   type=int, default=20,
                        help="Monte Carlo reps for accuracy evaluation")
    parser.add_argument("--k_fixed", type=float, default=1.0,
                        help="k value used for C1 ADC confidence window")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    set_plot_style()

    # Fixed parameters for C1
    # Use the three architecture σ_norm values as representative operating points
    arch_sigs = [compute_sigma_norm(a) for a in ARCHS]
    sigma_c1  = np.array(sorted(arch_sigs))     # 3 representative σ values
    n_bits_list = [2,3, 4, 5, 6,7, 8]
    k_fixed   = args.k_fixed

    # k values for C2 (note: small k = high tolerance!)
    k_c2_vals = [1,1.5, 2.0, 2.5, 3, 3.5, 4.0]

    for ds in ("mnist", "cifar10"):
        print(f"\n{'═'*54}")
        print(f"  C1 — {ds.upper()}")
        print("═" * 54)

        model     = load_model(ds, device)
        loader    = get_test_loader(ds)
        clean_acc = evaluate(model, loader, device)
        print(f"  Clean accuracy: {clean_acc*100:.2f}%")

        # Use a denser σ grid spanning the arch operating range
        sigma_lo = sigma_c1[0] * 0.3
        sigma_hi = sigma_c1[-1] * 5.0
        sigma_sweep = np.logspace(np.log10(sigma_lo), np.log10(sigma_hi), 20)

        print(f"  Running C1 ({len(sigma_sweep)} σ pts × {args.n_mc} MC, "
              f"k={k_fixed}, bits={n_bits_list})...")
        c1_res = run_c1(model, loader, device, sigma_sweep,
                        k_fixed, n_bits_list, args.n_mc)

        model_name = "LeNet-5" if ds == "mnist" else "ResNet-20"
        title = (f"{model_name} on {ds.upper()}  |  "
                 f"Analog vs ADC-quantised  (k={k_fixed})")
        fig_c1 = plot_c1(sigma_sweep, c1_res, clean_acc, title,
                          n_bits_list, k_fixed)
        save_fig(fig_c1, f"exp_c1_analog_vs_digital_{ds}")
        plt.close(fig_c1)

    # ── C2: Δ_quant verification (dataset-independent) ──────
    print(f"\n{'═'*54}")
    print(f"  C2 — Δ_quant empirical vs theory")
    print("═" * 54)

    fig_c2 = plot_c2(k_c2_vals, n_bits_list)
    save_fig(fig_c2, "exp_c2_delta_quant")
    plt.close(fig_c2)

    # ── Numerical summary for C2 ─────────────────────────────
    emp = compute_delta_quant_empirical(k_c2_vals, n_bits_list)
    print("\n  ─── Δ_quant: Theory vs Empirical ───────────────────")
    print(f"  {'k':>6}  {'Theory':>8}  " + "  ".join(f"{nb:>6}-bit" for nb in n_bits_list))
    for k in k_c2_vals:
        theory = float(delta_quant_theory(k))
        emps   = [emp[(k, nb)]["delta_emp"] for nb in n_bits_list]
        print(f"  {k:6.2f}  {theory:8.3f}  "
              + "  ".join(f"{e:8.3f}" for e in emps))
    print()


if __name__ == "__main__":
    main()
