"""
run_all.py — Top-level runner for the ONN precision experiment suite.

Executes all three experiments in sequence after verifying that trained
model weights exist.  Results (PDF + PNG) are written to results/.

Usage
-----
    # Step 1: train models (only once)
    python train.py --dataset both

    # Step 2: run all inference experiments
    python run_all.py [--device cpu|cuda] [--n_mc 20] [--n_sigma 40] [--fast]

    --fast   reduces MC repeats and sigma grid for quick smoke-testing
"""

import argparse
import os
import sys
import subprocess
import torch


def check_weights():
    """Abort early if trained weights are missing."""
    required = [
        os.path.join("weights", "lenet5_mnist.pth"),
        os.path.join("weights", "resnet20_cifar10.pth"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing model weights:")
        for p in missing:
            print(f"  {p}")
        print("\nRun  python train.py --dataset both  first.")
        sys.exit(1)
    print("  ✓ Model weights found.")


def run_script(script: str, extra_args: list):
    """Run a Python script in a subprocess, inheriting stdout/stderr."""
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'═'*60}")
    print(f"  Running: {' '.join(cmd)}")
    print("═" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default=None)
    parser.add_argument("--n_mc",    type=int, default=20)
    parser.add_argument("--n_sigma", type=int, default=40)
    parser.add_argument("--k_fixed", type=float, default=1.0,
                        help="ADC confidence window k for Exp C1")
    parser.add_argument("--fast",    action="store_true",
                        help="Quick smoke test (n_mc=3, n_sigma=12)")
    args = parser.parse_args()

    if args.fast:
        args.n_mc    = 3
        args.n_sigma = 12
        print("  [fast mode: n_mc=3, n_sigma=12]")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device_str}")

    check_weights()

    common_args = ["--device", device_str]
    """
    # ── Experiment A ─────────────────────────────────────────
    run_script("exp_a_convergence.py", common_args + [
        "--n_mc",    str(args.n_mc),
        "--n_sigma", str(args.n_sigma),
    ])
    """
    """
    # ── Experiment B ─────────────────────────────────────────
    # Uses σ*_norm defaults; override manually with printed values from A
    run_script("exp_b_ccenob.py", common_args)
    """
    # ── Experiment C ─────────────────────────────────────────
    run_script("exp_c_digitization.py", common_args + [
        "--n_mc",    str(args.n_mc),
        "--k_fixed", str(args.k_fixed),
    ])

    print("\n" + "═" * 60)
    print("  All experiments complete.")
    print("  Figures saved to results/")
    print("═" * 60)


if __name__ == "__main__":
    main()
