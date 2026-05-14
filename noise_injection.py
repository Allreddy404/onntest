"""
noise_injection.py — Noise injection hooks for ONN inference experiments.

Two hook classes:
  GaussianNoiseHook  — pure Gaussian noise (analog photonic mode)
  QuantizedNoiseHook — Gaussian noise quantised by n_b-bit ADC (hybrid mode)

Design notes
------------
• Noise is injected at the OUTPUT of every MAC layer (Conv2d / Linear),
  after the linear operation but before any activation. This matches the
  physical model where photocurrent noise appears at the detector output.

• Noise amplitude is scaled to the DYNAMIC RANGE of each layer's output
  batch-wise: scale = output.abs().max(). This ensures σ_norm is the
  normalised quantity as defined in Claim 1 regardless of layer depth or
  weight magnitude.

• For quantised injection the ADC maps the range [−k·σ_abs, +k·σ_abs]
  to 2^n_bits uniform levels (step size q = 2k·σ_abs / 2^n_bits).
  Values outside this range are CLIPPED — corresponding to the overflow
  event whose probability is P_ov(k) = 1 − erf(k/√2).

Usage
-----
    hooks = attach_gaussian_hooks(model, sigma_norm=1e-3)
    acc   = evaluate(model, loader, device)
    detach_hooks(hooks)

    hooks = attach_quantized_hooks(model, sigma_norm=1e-3, k=1.0, n_bits=4)
    acc   = evaluate(model, loader, device)
    detach_hooks(hooks)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List


# ══════════════════════════════════════════════════════════════
# Base hook class
# ══════════════════════════════════════════════════════════════

class _NoiseHookBase:
    """Base class — stores the hook handle and provides remove()."""

    def __init__(self):
        self._handle = None

    def register(self, module: nn.Module) -> "_NoiseHookBase":
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, module, input, output):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════
# Analog Gaussian hook
# ══════════════════════════════════════════════════════════════

class GaussianNoiseHook(_NoiseHookBase):
    """
    Injects i.i.d. Gaussian noise ε ~ N(0, σ²_abs) into layer outputs,
    where σ_abs = σ_norm × dynamic_range(output).

    Parameters
    ----------
    sigma_norm : float
        Normalised noise standard deviation (dimensionless, as in Claim 1).
    """

    def __init__(self, sigma_norm: float):
        super().__init__()
        self.sigma_norm = sigma_norm

    def _hook_fn(self, module, input, output: torch.Tensor) -> torch.Tensor:
        if self.sigma_norm == 0.0:
            return output
        # Scale to the output's dynamic range (batch-level full-scale estimate)
        scale = output.detach().abs().max().clamp(min=1e-12).item()
        sigma_abs = self.sigma_norm * scale
        noise = torch.randn_like(output) * sigma_abs
        return output + noise


# ══════════════════════════════════════════════════════════════
# Quantised (ADC) noise hook
# ══════════════════════════════════════════════════════════════

class QuantizedNoiseHook(_NoiseHookBase):
    """
    Injects quantised Gaussian noise, simulating an n_bits-bit ADC placed
    after each photonic MAC stage (hybrid analog-digital mode).

    Procedure per forward call:
      1. Generate ε ~ N(0, σ²_abs)                       [analog noise sample]
      2. Clip ε to [−k·σ_abs, +k·σ_abs]                  [ADC input range]
      3. Quantise to 2^n_bits levels with step q           [ADC digitisation]
      4. Add quantised ε_q to the layer output

    Overflow events (|ε| > k·σ_abs) are clipped deterministically,
    incurring the bit penalty Δ_quant(k) from eq. delta_quant.

    Parameters
    ----------
    sigma_norm : float   normalised noise std (Claim 1)
    k          : float   confidence boundary; high-tolerance → small k
    n_bits     : int     ADC resolution
    """

    def __init__(self, sigma_norm: float, k: float, n_bits: int):
        super().__init__()
        self.sigma_norm = sigma_norm
        self.k          = k
        self.n_bits     = n_bits

    def _hook_fn(self, module, input, output: torch.Tensor) -> torch.Tensor:
        if self.sigma_norm == 0.0:
            return output

        scale     = output.detach().abs().max().clamp(min=1e-12).item()
        sigma_abs = self.sigma_norm * scale
        clip      = self.k * sigma_abs                    # ADC full-scale boundary
        n_levels  = 2 ** self.n_bits
        step      = (2.0 * clip) / n_levels               # LSB step size q

        # Generate Gaussian noise
        epsilon = torch.randn_like(output) * sigma_abs

        if step > 0:
            # Quantise: round to nearest step, then clip to ADC range
            epsilon_q = torch.clamp(
                torch.round(epsilon / step) * step,
                -clip, clip
            )
        else:
            epsilon_q = torch.zeros_like(epsilon)

        return output + epsilon_q


# ══════════════════════════════════════════════════════════════
# Convenience: attach / detach hooks to all MAC layers
# ══════════════════════════════════════════════════════════════

def _mac_modules(model: nn.Module) -> List[nn.Module]:
    """Return all Conv2d and Linear modules (MAC operations only)."""
    return [m for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def attach_gaussian_hooks(model: nn.Module,
                          sigma_norm: float) -> List[GaussianNoiseHook]:
    """
    Attach GaussianNoiseHook to every MAC layer in model.
    Returns list of hooks — call detach_hooks(hooks) after evaluation.
    """
    hooks = []
    for m in _mac_modules(model):
        h = GaussianNoiseHook(sigma_norm)
        h.register(m)
        hooks.append(h)
    return hooks


def attach_quantized_hooks(model: nn.Module,
                           sigma_norm: float,
                           k: float,
                           n_bits: int) -> List[QuantizedNoiseHook]:
    """
    Attach QuantizedNoiseHook to every MAC layer in model.
    Returns list of hooks — call detach_hooks(hooks) after evaluation.
    """
    hooks = []
    for m in _mac_modules(model):
        h = QuantizedNoiseHook(sigma_norm, k, n_bits)
        h.register(m)
        hooks.append(h)
    return hooks


def detach_hooks(hooks: list):
    """Remove all registered forward hooks."""
    for h in hooks:
        h.remove()


# ══════════════════════════════════════════════════════════════
# Evaluation loop (shared across all experiments)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_logits(model: nn.Module, loader, device: torch.device) -> tuple:
    """
    Single clean forward pass over the full test set.

    Returns
    -------
    logits : (N, C) float32 CPU tensor  — clean model outputs
    labels : (N,)   int64  CPU tensor   — ground-truth class indices
    scale  : float  — logit dynamic range (abs-max), used to normalise noise

    Why this exists
    ───────────────
    For σ-sweep experiments the model weights are fixed.  Running the full
    forward pass once and caching the logits, then injecting noise directly
    at the logit level ("output-referred noise" approximation), is ≥ 1 000×
    faster than re-running the model for every (σ, MC trial, bit-width)
    combination.  The approximation is physically valid for comparative
    sweeps where the inter-layer noise propagation profile is held constant.
    """
    model.eval()
    logit_list, label_list = [], []
    for x, y in loader:
        logit_list.append(model(x.to(device)).cpu())
        label_list.append(y)
    logits = torch.cat(logit_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    scale  = float(logits.abs().max().clamp(min=1e-12))
    return logits, labels, scale


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def fast_sweep_gaussian(logits, labels, scale, sigma_values, n_mc=20):
    """
    Vectorised σ sweep — analog Gaussian noise on cached logits.
    Complexity: O(N_sigma × N_mc × tensor_add)  vs O(N_sigma × N_mc × forward_pass).

    Returns (means, stds) arrays, shape (len(sigma_values),).
    """
    means, stds = [], []
    for sig in sigma_values:
        sigma_abs = sig * scale
        accs = [_acc(logits + torch.randn_like(logits) * sigma_abs, labels)
                for _ in range(n_mc)]
        t = torch.tensor(accs)
        means.append(t.mean().item())
        stds.append(t.std().item())
    return np.array(means), np.array(stds)


def fast_sweep_quantized(logits, labels, scale, sigma_values,
                         k, n_bits, n_mc=20):
    """
    Same as fast_sweep_gaussian but with an n_bits-bit ADC applied to the
    Gaussian noise sample before injection.  ADC range: [−k·σ_abs, +k·σ_abs].
    """
    means, stds = [], []
    n_levels = 2 ** n_bits
    for sig in sigma_values:
        sigma_abs = sig * scale
        clip      = k * sigma_abs
        step      = (2.0 * clip / n_levels) if clip > 0 else 1.0
        accs = []
        for _ in range(n_mc):
            eps   = torch.randn_like(logits) * sigma_abs
            eps_q = (torch.clamp(torch.round(eps / step) * step, -clip, clip)
                     if step > 0 else torch.zeros_like(eps))
            accs.append(_acc(logits + eps_q, labels))
        t = torch.tensor(accs)
        means.append(t.mean().item())
        stds.append(t.std().item())
    return np.array(means), np.array(stds)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             device: torch.device) -> float:
    """
    Run inference and return top-1 accuracy.
    Hooks (if attached) are active during this call.
    """
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += x.size(0)
    return correct / total


@torch.no_grad()
def evaluate_mc(model: nn.Module,
                loader,
                device: torch.device,
                sigma_norm: float,
                n_mc: int = 20,
                hook_fn=attach_gaussian_hooks,
                **hook_kwargs) -> tuple:
    """
    Monte Carlo evaluation: repeat n_mc times with freshly sampled noise,
    return (mean_accuracy, std_accuracy).

    Parameters
    ----------
    hook_fn     : attach_gaussian_hooks or attach_quantized_hooks
    hook_kwargs : extra keyword arguments forwarded to hook_fn
                  (e.g. k=1.0, n_bits=4 for quantized hooks)
    """
    accs = []
    for _ in range(n_mc):
        hooks = hook_fn(model, sigma_norm=sigma_norm, **hook_kwargs)
        acc   = evaluate(model, loader, device)
        detach_hooks(hooks)
        accs.append(acc)
    accs = torch.tensor(accs)
    return accs.mean().item(), accs.std().item()
