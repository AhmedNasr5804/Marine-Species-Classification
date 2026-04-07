"""
model_a.py — Custom CNN from scratch (Model A).

Architecture
------------
Input (3 × 224 × 224)
  └─ Block 1: Conv(3→32,  3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → 32 × 112 × 112
  └─ Block 2: Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → 64 ×  56 ×  56
  └─ Block 3: Conv(64→128,3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → 128 × 28 × 28
  └─ Block 4: Conv(128→256,3×3,pad=1) → BN → ReLU → AdaptAvgPool(4×4) → 256 × 4 × 4
  └─ Flatten                                                          → 4096
  └─ FC(4096 → 512) → ReLU → Dropout(p)
  └─ FC(512 → num_classes)

Architectural justifications (required by §4.2)
------------------------------------------------
3×3 convolutional filters
    Two stacked 3×3 convolutions cover the same receptive field as a single 5×5 with
    fewer parameters and an extra non-linearity. VGGNet (Simonyan & Zisserman, 2015)
    demonstrated this is both parameter-efficient and empirically effective.

Progressive channel doubling (32→64→128→256)
    Doubling the number of feature maps after each spatial downsampling compensates for
    the halved resolution, maintaining roughly constant representational bandwidth per
    layer. This is a well-established heuristic from VGG and AlexNet families.

BatchNorm placed after convolution, before activation
    Batch Normalisation (Ioffe & Szegedy, 2015) whitens intermediate activations,
    stabilises gradient flow, and reduces sensitivity to weight initialisation. Placing
    it before the activation function ensures that ReLU receives zero-centred inputs,
    which reduces the "dying ReLU" problem.

MaxPool(2×2) for spatial downsampling
    Max pooling retains the strongest local activation, providing spatial translation
    invariance. Stride-2 pooling halves both spatial dimensions, giving a 4× compression
    per pooling step and reducing the computational cost of subsequent layers.

Block 4 uses AdaptiveAvgPool(4×4) instead of MaxPool
    The final block uses adaptive average pooling to obtain a fixed 4×4 feature map
    regardless of the input resolution. This decouples the fully-connected head size from
    the input image dimensions, making the architecture more portable. Average pooling
    here aggregates spatial information more smoothly than max pooling, which is
    appropriate before a global summary layer.

Dropout(p) on the fully-connected layer
    Dropout (Srivastava et al., 2014) is the primary regularisation mechanism for the
    classification head. During training, it randomly zeroes a fraction p of neurons,
    preventing co-adaptation and acting as an implicit ensemble. The default p=0.5 is
    the canonical value for fully-connected layers.

Label smoothing (CrossEntropyLoss, ε=0.1)
    Replaces hard one-hot targets with soft labels (1−ε for the correct class, ε/(K−1)
    for others). This prevents overconfident predictions and has been shown to improve
    generalisation and calibration (Müller et al., 2019).

AdamW optimiser
    AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the adaptive learning
    rate update, correcting a flaw in the original Adam where L2 regularisation does not
    behave as intended. It converges reliably on image classification tasks.

CosineAnnealingLR schedule
    Smoothly decays the learning rate from lr_max to lr_min following a cosine curve,
    without requiring manual step-decay schedules. This avoids abrupt learning rate drops
    that can cause instability.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reusable building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Convolutional block: Conv → BatchNorm → ReLU → Pooling.

    Args:
        in_channels:   Input channel count.
        out_channels:  Output channel count.
        pool:          Pooling module to append. Pass None to omit pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(pool)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Full CNN
# ---------------------------------------------------------------------------

class MarineCNN(nn.Module):
    """Custom CNN trained from scratch for marine species classification.

    Args:
        num_classes:   Number of output classes (known species only).
        dropout_rate:  Dropout probability applied to the FC layer. Default 0.5.
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — learns low-level edges and colour blobs
            ConvBlock(3,   32,  pool=nn.MaxPool2d(2, 2)),       # → 32 × 112 × 112
            # Block 2 — combines low-level features into textures
            ConvBlock(32,  64,  pool=nn.MaxPool2d(2, 2)),       # → 64 ×  56 ×  56
            # Block 3 — learns part-level representations
            ConvBlock(64,  128, pool=nn.MaxPool2d(2, 2)),       # → 128 × 28 × 28
            # Block 4 — learns high-level organism patterns; adaptive pool for fixed output
            ConvBlock(128, 256, pool=nn.AdaptiveAvgPool2d((4, 4))),  # → 256 × 4 × 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                         # → 4096
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming (He) initialisation for Conv layers; zero-bias for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Hyperparameter configurations (§4.2)
# ---------------------------------------------------------------------------

CONFIGS = {
    "config_1": {
        "lr": 1e-3,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout_rate": 0.5,
        "description": "Baseline — higher LR, smaller batch",
    },
    "config_2": {
        "lr": 3e-4,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout_rate": 0.5,
        "description": "Lower LR with larger batch — more stable gradient estimates",
    },
}


def build_model_a(num_classes: int, config_name: str = "config_1") -> MarineCNN:
    """Instantiate MarineCNN with the dropout rate from the named config.

    Args:
        num_classes:  Number of known species classes.
        config_name:  One of "config_1" or "config_2" (see CONFIGS).

    Returns:
        Initialised MarineCNN ready for training.
    """
    cfg = CONFIGS[config_name]
    model = MarineCNN(num_classes=num_classes, dropout_rate=cfg["dropout_rate"])
    return model


def build_optimizer_and_scheduler(
    model: nn.Module,
    config_name: str = "config_1",
    epochs: int = 30,
) -> tuple:
    """Return (optimizer, scheduler, criterion) for the given config.

    Args:
        model:        The model whose parameters will be optimised.
        config_name:  One of the keys in CONFIGS.
        epochs:       Total training epochs (used to set scheduler period).

    Returns:
        Tuple of (AdamW optimizer, CosineAnnealingLR scheduler, CrossEntropyLoss).
    """
    cfg = CONFIGS[config_name]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    return optimizer, scheduler, criterion
