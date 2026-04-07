"""
model_b.py — Transfer learning wrappers (Model B).

Three strategies are implemented as required by §4.3:

  Strategy A — Feature Extraction
      The backbone is entirely frozen. Only the newly added classification head
      is trained. This is the fastest strategy and works well when the target
      domain is similar to ImageNet (which underwater imagery partially is —
      textures and shapes overlap). Suitable when labelled data is scarce.

  Strategy B — Partial Fine-Tuning
      The uppermost backbone block (layer4 in ResNet-50) is unfrozen and trained
      jointly with the head using differential learning rates (backbone LR 10×
      lower than the head). This allows the network to adapt high-level features
      to the target domain while keeping low-level features stable.

  Strategy C — Full Fine-Tuning
      The entire backbone is unfrozen with a very low LR (1e-5) to avoid
      catastrophic forgetting of ImageNet representations. A linear warmup for
      the first few epochs is used to prevent large gradient updates early in
      training from destabilising pretrained weights.

Backbone choice: ResNet-50
    ResNet-50 (He et al., 2016) is chosen as the default because:
    - Its residual connections make it robust to gradient vanishing during
      fine-tuning of all layers.
    - The 2048-d final feature vector is rich enough for fine-grained species
      discrimination.
    - It is computationally manageable on a single GPU.
    EfficientNet-B0 is also supported via the `backbone` argument for comparison.

Classification head
    Linear(backbone_out → 512) → ReLU → Dropout(0.3) → Linear(512 → num_classes)
    Dropout rate is lower (0.3 vs 0.5 for Model A) because the pretrained backbone
    already acts as a strong feature regulariser.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torchvision import models

# Supported backbone identifiers
BackboneID = Literal["resnet50", "efficientnet_b0"]

# Transfer learning strategy names
Strategy = Literal["feature_extraction", "partial_finetuning", "full_finetuning"]


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

def _build_head(in_features: int, num_classes: int, dropout: float = 0.3) -> nn.Sequential:
    """Two-layer FC head replacing the backbone's original classifier."""
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),
    )


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _load_backbone(backbone_id: BackboneID) -> tuple[nn.Module, int]:
    """Load a pretrained backbone and return (model, feature_dim).

    The final classification layer is removed — only the feature extractor is
    returned. The caller attaches a task-specific head.

    Args:
        backbone_id: One of "resnet50" or "efficientnet_b0".

    Returns:
        Tuple of (backbone_without_classifier, output_feature_dim).
    """
    if backbone_id == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features      # 2048
        backbone.fc = nn.Identity()                # remove original classifier
        return backbone, in_features

    elif backbone_id == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features   # 1280
        backbone.classifier = nn.Identity()
        return backbone, in_features

    else:
        raise ValueError(f"Unsupported backbone: {backbone_id!r}. "
                         f"Choose from 'resnet50' or 'efficientnet_b0'.")


# ---------------------------------------------------------------------------
# TransferModel — unified wrapper for all three strategies
# ---------------------------------------------------------------------------

class TransferModel(nn.Module):
    """Pretrained backbone + custom head for marine species classification.

    Args:
        num_classes:  Number of known species classes.
        backbone_id:  Which pretrained backbone to use.
        strategy:     Transfer learning strategy (controls which layers are frozen).
        dropout:      Dropout rate in the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_id: BackboneID = "resnet50",
        strategy: Strategy = "feature_extraction",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_id = backbone_id
        self.strategy = strategy

        self.backbone, in_features = _load_backbone(backbone_id)
        self.head = _build_head(in_features, num_classes, dropout)

        self._apply_strategy(strategy)

    def _apply_strategy(self, strategy: Strategy) -> None:
        """Freeze/unfreeze backbone layers according to the chosen strategy."""
        if strategy == "feature_extraction":
            # Freeze entire backbone — only head trains.
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif strategy == "partial_finetuning":
            # Freeze all backbone layers first, then unfreeze the last block.
            for param in self.backbone.parameters():
                param.requires_grad = False

            if self.backbone_id == "resnet50":
                # Unfreeze layer4 (the deepest residual block, learns task-specific features)
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif self.backbone_id == "efficientnet_b0":
                # Unfreeze the last 3 MBConv blocks (features[6], features[7], features[8])
                for block in list(self.backbone.features)[-3:]:
                    for param in block.parameters():
                        param.requires_grad = True

        elif strategy == "full_finetuning":
            # Unfreeze everything — entire backbone + head trains end-to-end.
            for param in self.backbone.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories  (one per strategy)
# ---------------------------------------------------------------------------

def build_optimizer_feature_extraction(
    model: TransferModel,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
) -> tuple:
    """Strategy A: single LR for the head only.

    Only head parameters have requires_grad=True so a standard optimizer
    call is sufficient — backbone gradients are never computed.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return optimizer, scheduler, criterion


def build_optimizer_partial_finetuning(
    model: TransferModel,
    backbone_id: BackboneID = "resnet50",
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
) -> tuple:
    """Strategy B: differential LRs — backbone block at 10× lower rate than head.

    Using separate parameter groups lets AdamW apply different adaptive learning
    rates per group, which is essential for partial fine-tuning: the pretrained
    layers need conservative updates while the head can learn faster.
    """
    if backbone_id == "resnet50":
        backbone_params = list(model.backbone.layer4.parameters())
    else:
        backbone_params = []
        for block in list(model.backbone.features)[-3:]:
            backbone_params.extend(block.parameters())

    head_params = list(model.head.parameters())

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return optimizer, scheduler, criterion


def build_optimizer_full_finetuning(
    model: TransferModel,
    head_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    warmup_epochs: int = 3,
) -> tuple:
    """Strategy C: full backbone fine-tuning with linear warmup.

    The very low backbone LR (1e-5) prevents catastrophic forgetting —
    pretrained weights are updated only slightly per step. Linear warmup
    for the first `warmup_epochs` further protects against large early updates
    that could collapse the pretrained feature space before the head is stable.

    Scheduler: LinearLR warmup → CosineAnnealingLR decay (SequentialLR).
    """
    param_groups = [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.head.parameters(),     "lr": head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=1e-7,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return optimizer, scheduler, criterion


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS: Dict[Strategy, dict] = {
    "feature_extraction": {
        "lr": 1e-3,
        "description": "Frozen backbone — head only. Fast; best when data is scarce.",
    },
    "partial_finetuning": {
        "head_lr": 1e-3,
        "backbone_lr": 1e-4,
        "description": "layer4 + head. Adapts high-level features to marine domain.",
    },
    "full_finetuning": {
        "head_lr": 1e-4,
        "backbone_lr": 1e-5,
        "description": "All layers. Highest capacity; requires most data and time.",
    },
}


def build_model_b(
    num_classes: int,
    strategy: Strategy,
    backbone_id: BackboneID = "resnet50",
    epochs: int = 20,
) -> tuple[TransferModel, object, object, nn.Module]:
    """Instantiate TransferModel and return (model, optimizer, scheduler, criterion).

    Args:
        num_classes:  Number of known species classes.
        strategy:     One of "feature_extraction", "partial_finetuning", "full_finetuning".
        backbone_id:  Pretrained backbone to use.
        epochs:       Training epochs (used to configure the scheduler).

    Returns:
        Tuple (model, optimizer, scheduler, criterion).
    """
    model = TransferModel(num_classes=num_classes, backbone_id=backbone_id,
                          strategy=strategy)

    if strategy == "feature_extraction":
        opt, sched, crit = build_optimizer_feature_extraction(model, epochs=epochs)
    elif strategy == "partial_finetuning":
        opt, sched, crit = build_optimizer_partial_finetuning(
            model, backbone_id=backbone_id, epochs=epochs
        )
    elif strategy == "full_finetuning":
        opt, sched, crit = build_optimizer_full_finetuning(model, epochs=epochs)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return model, opt, sched, crit
