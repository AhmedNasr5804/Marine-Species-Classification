"""
train.py — Generic training and evaluation loop.

This module is intentionally model-agnostic: it works with any nn.Module that
returns class logits from a forward pass. It is used for both Model A (CNN from
scratch) and Model B (transfer learning).

Returned history format
-----------------------
{
    "train_loss":  [float, ...],   # one entry per epoch
    "train_acc":   [float, ...],
    "val_loss":    [float, ...],
    "val_acc":     [float, ...],
}

This dict is passed directly to src/evaluate.py for curve plotting.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, float]:
    """Run one training epoch.

    Uses automatic mixed precision (AMP) if a GradScaler is provided. AMP
    reduces memory usage and speeds up training on modern GPUs without
    meaningful accuracy loss.

    Args:
        model:     The neural network in training mode.
        loader:    Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser instance.
        device:    CPU or CUDA device.
        scaler:    AMP GradScaler. Pass None to disable AMP.

    Returns:
        Tuple (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate *model* on *loader* without gradient computation.

    Args:
        model:     The neural network in eval mode.
        loader:    Validation or test DataLoader.
        criterion: Loss function (same as training, for comparable loss values).
        device:    CPU or CUDA device.

    Returns:
        Tuple (mean_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int,
    device: torch.device,
    use_amp: bool = True,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train *model* for *epochs* epochs and return the history dict.

    Features:
    - Optional AMP (automatic mixed precision) for GPU speed-up.
    - Gradient clipping (max_norm=1.0) to prevent exploding gradients.
    - Optional early stopping based on validation loss.
    - Optional best-model checkpointing (saves weights when val_loss improves).

    Args:
        model:                   The model to train.
        train_loader:            Training DataLoader.
        val_loader:              Validation DataLoader.
        criterion:               Loss function.
        optimizer:               Optimiser.
        scheduler:               LR scheduler (stepped per epoch). Pass None to disable.
        epochs:                  Maximum training epochs.
        device:                  torch.device to train on.
        use_amp:                 Enable AMP on CUDA. Ignored on CPU.
        early_stopping_patience: Stop if val_loss does not improve for this many
                                 consecutive epochs. Pass None to disable.
        checkpoint_path:         File path to save the best model weights (.pt).
                                 Convention: "deliverables/weights/<model>_best.pt".
                                 Pass None to skip saving.

    Returns:
        History dict with keys "train_loss", "train_acc", "val_loss", "val_acc".
        Each value is a list of floats with one entry per epoch.
    """
    model.to(device)

    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}  [{elapsed:.1f}s]"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpointing — save best model by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"  [early stop] val_loss did not improve for {patience_counter} epochs.")
            break

    # Restore best weights if we saved them
    if checkpoint_path:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"  [checkpoint] Restored best weights from {checkpoint_path}")
        except FileNotFoundError:
            pass

    return history


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
