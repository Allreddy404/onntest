"""
train.py — Training script for LeNet-5 (MNIST) and ResNet-20 (CIFAR-10).

Usage:
    python train.py --dataset mnist          # ~5 min on CPU
    python train.py --dataset cifar10        # ~60 min on GPU
    python train.py --dataset both

Weights are saved to weights/ directory and loaded by all inference scripts.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import LeNet5, ResNet20

WEIGHTS_DIR = "weights"
DATA_DIR    = "{weights,results,data}/data"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════

def get_mnist_loaders(batch_size: int = 256, num_workers: int = 2):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=512,       shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=512,       shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld


# ══════════════════════════════════════════════════════════════
# Generic train / eval loops
# ══════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════
# MNIST — LeNet-5
# ══════════════════════════════════════════════════════════════

def train_mnist(device: torch.device, epochs: int = 20):
    print("\n" + "═" * 60)
    print("  LeNet-5 on MNIST")
    print("═" * 60)

    model     = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ld, test_ld = get_mnist_loaders()
    best_acc  = 0.0
    save_path = os.path.join(WEIGHTS_DIR, "lenet5_mnist.pth")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, optimizer, device, training=True)
        te_loss, te_acc = run_epoch(model, test_ld,  criterion, optimizer, device, training=False)
        scheduler.step()
        dt = time.time() - t0

        print(f"  Ep {ep:3d}/{epochs}  "
              f"train {tr_acc*100:6.2f}%  test {te_acc*100:6.2f}%  "
              f"({dt:.1f}s)")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "epoch":      ep,
                "state_dict": model.state_dict(),
                "test_acc":   te_acc,
                "dataset":    "mnist",
                "arch":       "lenet5",
            }, save_path)

    print(f"\n  ✓ Best test accuracy : {best_acc*100:.2f}%")
    print(f"  ✓ Weights saved to   : {save_path}\n")
    return best_acc


# ══════════════════════════════════════════════════════════════
# CIFAR-10 — ResNet-20
# ══════════════════════════════════════════════════════════════

def train_cifar10(device: torch.device, epochs: int = 200):
    print("\n" + "═" * 60)
    print("  ResNet-20 on CIFAR-10")
    print("═" * 60)

    model     = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    # Standard LR schedule: decay at epoch 100 and 150
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100, 150],
                                               gamma=0.1)

    train_ld, test_ld = get_cifar10_loaders()
    best_acc  = 0.0
    save_path = os.path.join(WEIGHTS_DIR, "resnet20_cifar10.pth")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, optimizer, device, training=True)
        te_loss, te_acc = run_epoch(model, test_ld,  criterion, optimizer, device, training=False)
        scheduler.step()
        dt = time.time() - t0

        if ep % 10 == 0 or ep <= 5:
            print(f"  Ep {ep:3d}/{epochs}  "
                  f"train {tr_acc*100:6.2f}%  test {te_acc*100:6.2f}%  "
                  f"(lr={scheduler.get_last_lr()[0]:.1e}, {dt:.1f}s)")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "epoch":      ep,
                "state_dict": model.state_dict(),
                "test_acc":   te_acc,
                "dataset":    "cifar10",
                "arch":       "resnet20",
            }, save_path)

    print(f"\n  ✓ Best test accuracy : {best_acc*100:.2f}%")
    print(f"  ✓ Weights saved to   : {save_path}\n")
    return best_acc


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ONN baseline models.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "both"],
                        default="both", help="Which dataset to train on.")
    parser.add_argument("--epochs_mnist",   type=int, default=20)
    parser.add_argument("--epochs_cifar10", type=int, default=200)
    parser.add_argument("--device", default=None,
                        help="torch device string (default: cuda if available else cpu)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device : {device}")
    torch.manual_seed(42)

    if args.dataset in ("mnist", "both"):
        train_mnist(device, epochs=args.epochs_mnist)

    if args.dataset in ("cifar10", "both"):
        train_cifar10(device, epochs=args.epochs_cifar10)
