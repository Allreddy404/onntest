"""
models.py — Neural network architectures for ONN precision experiments.

LeNet-5  : MNIST  (1×28×28 → 10 classes)
ResNet-20: CIFAR-10 (3×32×32 → 10 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# LeNet-5
# ══════════════════════════════════════════════════════════════

class LeNet5(nn.Module):
    """Classic LeNet-5 with tanh activations (matches analog ONN style)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6,  kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)               # logits; no softmax here
        return x

    def mac_layers(self):
        """Return all layers that perform MAC operations (Conv + Linear)."""
        return [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]


# ══════════════════════════════════════════════════════════════
# ResNet-20
# ══════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10.
    Architecture: 1 stem conv + 3 stages × 3 BasicBlocks.
    Target clean accuracy ≥ 91%.
    """

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.layer1  = self._make_layer(16, 16, n_blocks=3, stride=1)
        self.layer2  = self._make_layer(16, 32, n_blocks=3, stride=2)
        self.layer3  = self._make_layer(32, 64, n_blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, 10)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

    def mac_layers(self):
        """Return all Conv2d and Linear layers (MAC operations only)."""
        return [m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
