"""
arch_params.py — ONN physical parameters and σ_norm computation.

All σ_norm values are derived from the Claim 1 formula (eq. normalized_var_simplified):

    σ²_norm = (C_η·η + C_coh + C_shot + C_bg) · B

where the four unit-spectrum noise coefficients [Hz⁻¹] are:

    C_η    = (1−ρ)·RIN + ERIN          (quadratic incoherent)
    C_coh  = ρ·RIN                      (quadratic coherent, ρ→1: hard floor)
    C_shot = 2q / (ℛ·P_max)            (linear signal-dependent)
    C_bg   = (4k_BT/R_L + 2qI_d)       (constant background)
             / (ℛ·P_max)²

Physical constants
------------------
q  = 1.602 × 10⁻¹⁹  C
k_B = 1.381 × 10⁻²³  J/K
T   = 300 K
"""

from dataclasses import dataclass, field
import numpy as np

# ── Physical constants ──────────────────────────────────────
Q_E = 1.602e-19     # electron charge [C]
K_B = 1.381e-23     # Boltzmann constant [J/K]
T   = 300.0         # operating temperature [K]


@dataclass
class ONNArch:
    """
    All physical parameters required to evaluate σ_norm for one ONN architecture.

    Parameters
    ----------
    name   : descriptive name
    label  : short string for figure legend
    rho    : channel correlation coefficient ρ ∈ [0, 1]
    RIN    : laser relative intensity noise [Hz⁻¹]  (not dBc/Hz — use dBc_to_lin())
    ERIN   : excess RIN from modulators/components [Hz⁻¹]
    P_max  : full-scale optical power per channel [W]
    B      : operating bandwidth [Hz]
    R      : photodetector responsivity [A/W]
    R_L    : transimpedance load resistance [Ω]
    I_d    : photodetector dark current [A]
    eta    : data sparsity index η = Σz_i² (worst-case full-scale: varies; use N⁻¹ for dense)
    color  : matplotlib colour string
    marker : matplotlib marker string
    """
    name   : str
    label  : str
    rho    : float
    RIN    : float
    ERIN   : float
    P_max  : float
    B      : float
    R      : float   = 0.8
    R_L    : float   = 50.0
    I_d    : float   = 1e-9
    eta    : float   = 0.1
    color  : str     = "#333333"
    marker : str     = "o"


def dBc_to_lin(dBc_per_Hz: float) -> float:
    """Convert dBc/Hz to linear [Hz⁻¹]. E.g. −150 dBc/Hz → 10⁻¹⁵ Hz⁻¹."""
    return 10.0 ** (dBc_per_Hz / 10.0)


def noise_coefficients(arch: ONNArch) -> dict:
    """
    Compute the four unit-spectrum noise coefficients [Hz⁻¹].
    Returns dict with keys: C_eta, C_coh, C_shot, C_bg.
    """
    I_fs = arch.R * arch.P_max            # full-scale photocurrent [A]
    C_eta  = (1.0 - arch.rho) * arch.RIN + arch.ERIN
    C_coh  = arch.rho * arch.RIN
    C_shot = 2.0 * Q_E / I_fs
    C_bg   = (4.0 * K_B * T / arch.R_L + 2.0 * Q_E * arch.I_d) / I_fs**2
    return dict(C_eta=C_eta, C_coh=C_coh, C_shot=C_shot, C_bg=C_bg)


def sigma_norm(arch: ONNArch) -> float:
    """
    σ_norm = sqrt[(C_η·η + C_coh + C_shot + C_bg) · B]

    This is the normalised output noise standard deviation under the
    full-scale worst-case boundary condition (Σz_i ≈ 1).
    """
    c = noise_coefficients(arch)
    var = (c["C_eta"] * arch.eta + c["C_coh"] + c["C_shot"] + c["C_bg"]) * arch.B
    return float(np.sqrt(max(var, 0.0)))


def sigma_norm_breakdown(arch: ONNArch) -> dict:
    """
    Return σ_norm plus the fractional contribution of each noise term.
    Useful for noise budget figures.
    """
    c   = noise_coefficients(arch)
    var_eta  = c["C_eta"]  * arch.eta * arch.B
    var_coh  = c["C_coh"]              * arch.B
    var_shot = c["C_shot"]             * arch.B
    var_bg   = c["C_bg"]               * arch.B
    var_tot  = var_eta + var_coh + var_shot + var_bg
    sig      = float(np.sqrt(max(var_tot, 0.0)))
    return dict(
        sigma_norm = sig,
        var_total  = var_tot,
        frac_eta   = var_eta  / var_tot,
        frac_coh   = var_coh  / var_tot,
        frac_shot  = var_shot / var_tot,
        frac_bg    = var_bg   / var_tot,
        coeffs     = c,
    )


# ══════════════════════════════════════════════════════════════
# Three representative ONN architectures
# ══════════════════════════════════════════════════════════════
#
# Parameter rationale (all grounded in Claim 1):
#
#   Arch-A  MRR weight bank — independent sources (ρ = 0)
#           Low bandwidth (1 GHz), high power (1 mW).
#           C_coh = 0 → no coherence floor; best-case precision.
#           σ_norm ≈ 1.0 × 10⁻³
#
#   Arch-B  Photonic crossbar — partially shared source (ρ = 0.5)
#           Medium bandwidth (10 GHz), medium power (0.5 mW).
#           Partial coherence floor from shared laser.
#           σ_norm ≈ 5.9 × 10⁻³
#
#   Arch-C  MZI mesh / SVD processor — fully shared source (ρ → 1)
#           High bandwidth (20 GHz), low power (0.2 mW).
#           Coherence floor C_coh · B dominates; worst-case precision.
#           σ_norm ≈ 1.8 × 10⁻²
#
# The ~18× spread in σ_norm (≈ 4.2 bits in raw ENOB) between Arch-A and
# Arch-C is intentional: the High-Tolerance Convergence Theorem predicts
# that standard ML tasks will still show convergent accuracy despite this
# large raw-precision gap.

ARCHS = [
    ONNArch(
        name  = "incoherent",
        label = "Arch-A  eta=0.1 ρ=0",
        rho   = 0.0,
        RIN   = dBc_to_lin(-150),       # 1.00 × 10⁻¹⁵ Hz⁻¹
        ERIN  = dBc_to_lin(-155),       # 3.16 × 10⁻¹⁶ Hz⁻¹
        P_max = 10.0e-3,                 # 1 mW
        B     = 1.0e9,                  # 1 GHz
        eta   = 0.1,
        color = "#185FA5",
        marker= "o",
    ),
    ONNArch(
        name  = "paritial coherent",
        label = "Arch-B  eta=0.5  ρ=0.5",
        rho   = 0.5,
        RIN   = dBc_to_lin(-150),
        ERIN  = dBc_to_lin(-155),
        P_max = 10.0e-3,                 # 0.5 mW
        B     = 1.0e9,                 # 10 GHz
        eta   = 0.5,
        color = "#1D9E75",
        marker= "s",
    ),
    ONNArch(
        name  = "coherent",
        label = "Arch-C  eta=0.5 ρ=1",
        rho   = 1.0,
        RIN   = dBc_to_lin(-150),
        ERIN  = dBc_to_lin(-155),
        P_max = 10.0e-3,                 # 0.2 mW
        B     = 1.0e9,                 # 20 GHz
        eta   = 0.5,
        color = "#D85A30",
        marker= "^",
    ),
]


# ── Convenience: print a noise budget summary ───────────────

def print_budget(arch: ONNArch):
    bd = sigma_norm_breakdown(arch)
    print(f"\n{'─'*54}")
    print(f"  {arch.name}")
    print(f"  ρ={arch.rho}  B={arch.B/1e9:.0f} GHz  "
          f"P_max={arch.P_max*1e3:.1f} mW")
    print(f"  σ_norm = {bd['sigma_norm']:.3e}")
    print(f"  Noise budget:")
    print(f"    C_η·η  : {bd['frac_eta']*100:6.2f}%")
    print(f"    C_coh  : {bd['frac_coh']*100:6.2f}%")
    print(f"    C_shot : {bd['frac_shot']*100:6.2f}%")
    print(f"    C_bg   : {bd['frac_bg']*100:6.2f}%")


if __name__ == "__main__":
    print("\nONN Architecture σ_norm Summary")
    print("(derived from Claim 1 physical formula)")
    for a in ARCHS:
        print_budget(a)
    print()

    sigs = [sigma_norm(a) for a in ARCHS]
    ratio = sigs[-1] / sigs[0]
    enob_gap = 0.5 * np.log2((sigs[-1] / sigs[0])**2)
    print(f"  σ_norm(Arch-C) / σ_norm(Arch-A) = {ratio:.1f}×")
    print(f"  Raw ENOB gap (A→C)              = {enob_gap:.2f} bits\n")
