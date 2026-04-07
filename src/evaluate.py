"""
evaluate.py — Evaluation metrics and visualisation helpers.

Produces all outputs required by §4.5:
  - Per-class precision, recall, F1 score.
  - Macro-averaged F1 and overall classification accuracy.
  - Confusion matrix as a labelled heatmap.
  - Training/validation loss and accuracy curves.
  - OOD detection precision-recall curve (used by ood.py results).

All figures are saved to a configurable output directory (default: figures/).
Functions return the figure objects so they can also be rendered inline in the
Jupyter notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Default figure output directory (relative to the notebook working directory).
# All real outputs go under deliverables/ so the submission folder is self-contained.
FIGURES_DIR = Path("deliverables/figures")


def _ensure_figures_dir(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on *loader* and collect all predictions and ground-truth labels.

    Args:
        model:   Trained classifier in eval mode.
        loader:  DataLoader for any split (val, test, or OOD).
        device:  Compute device.

    Returns:
        Tuple (y_true, y_pred) — both shape (N,) integer arrays.
    """
    model.eval()
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc="  predicting", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute the full suite of classification metrics required by §4.5.

    Args:
        y_true:       Ground-truth integer labels, shape (N,).
        y_pred:       Predicted integer labels, shape (N,).
        class_names:  Optional list of human-readable class names. If None,
                      integer indices are used.

    Returns:
        Dict with keys:
          "accuracy"        — overall accuracy (float)
          "macro_f1"        — macro-averaged F1 (float)
          "report"          — sklearn classification_report string
          "report_dict"     — per-class dict {class_name: {precision, recall, f1, support}}
          "confusion_matrix"— integer confusion matrix (num_classes × num_classes)
          "class_names"     — list of class names used
    """
    labels = list(range(int(y_true.max()) + 1))
    target_names = class_names if class_names else [str(i) for i in labels]

    acc       = float(accuracy_score(y_true, y_pred))
    report_str  = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names,
        output_dict=True, zero_division=0
    )
    macro_f1  = float(report_dict["macro avg"]["f1-score"])
    cm        = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy":         round(acc,      4),
        "macro_f1":         round(macro_f1, 4),
        "report":           report_str,
        "report_dict":      report_dict,
        "confusion_matrix": cm,
        "class_names":      target_names,
    }


def print_metrics(metrics: Dict, title: str = "") -> None:
    """Pretty-print the metrics dict returned by compute_metrics()."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print()
    print(metrics["report"])


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    metrics: Dict,
    title: str = "Confusion Matrix",
    figures_dir: Path = FIGURES_DIR,
    filename: Optional[str] = None,
    normalise: bool = True,
) -> plt.Figure:
    """Plot a labelled confusion matrix heatmap and save to disk.

    Args:
        metrics:      Output of compute_metrics().
        title:        Plot title.
        figures_dir:  Directory to save the figure.
        filename:     File name (without path). Defaults to a slugified title.
        normalise:    If True, normalise rows to [0, 1] (shows recall per class).

    Returns:
        matplotlib Figure object.
    """
    _ensure_figures_dir(figures_dir)

    cm = metrics["confusion_matrix"].astype(float)
    class_names = metrics["class_names"]

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2f"
        cbar_label = "Recall (row-normalised)"
    else:
        fmt = "d"
        cm = cm.astype(int)
        cbar_label = "Count"

    n = len(class_names)
    fig_width  = max(8, n * 0.7)
    fig_height = max(6, n * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()

    if filename is None:
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  [saved] {figures_dir / filename}")
    return fig


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    figures_dir: Path = FIGURES_DIR,
    filename: Optional[str] = None,
) -> plt.Figure:
    """Plot loss and accuracy curves for a single training run.

    Args:
        history:      Dict returned by src.train.train() with keys
                      "train_loss", "val_loss", "train_acc", "val_acc".
        title:        Suptitle for the figure.
        figures_dir:  Directory to save the figure.
        filename:     File name. Defaults to a slugified title.

    Returns:
        matplotlib Figure object.
    """
    _ensure_figures_dir(figures_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    ax_loss.plot(epochs, history["train_loss"], label="Train loss", linewidth=1.8)
    ax_loss.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=1.8,
                 linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # Accuracy subplot
    ax_acc.plot(epochs, history["train_acc"], label="Train acc", linewidth=1.8)
    ax_acc.plot(epochs, history["val_acc"],   label="Val acc",   linewidth=1.8,
                linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if filename is None:
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  [saved] {figures_dir / filename}")
    return fig


def plot_comparison_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "val_acc",
    title: str = "Validation Accuracy Comparison",
    figures_dir: Path = FIGURES_DIR,
    filename: Optional[str] = None,
) -> plt.Figure:
    """Overlay multiple training runs on a single axes for comparison.

    Args:
        histories:  Dict mapping run_name → history dict.
        metric:     Which history key to plot ("val_acc", "val_loss", etc.).
        title:      Plot title.
        figures_dir: Directory to save the figure.
        filename:   File name.

    Returns:
        matplotlib Figure object.
    """
    _ensure_figures_dir(figures_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, history in histories.items():
        values = history[metric]
        ax.plot(range(1, len(values) + 1), values, label=run_name, linewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    if "acc" in metric:
        ax.set_ylim(0, 1)
    fig.tight_layout()

    if filename is None:
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  [saved] {figures_dir / filename}")
    return fig


# ---------------------------------------------------------------------------
# OOD detection visualisation
# ---------------------------------------------------------------------------

def plot_ood_curves(
    ood_results: Dict,
    title: str = "OOD Detection",
    figures_dir: Path = FIGURES_DIR,
    filename: Optional[str] = None,
) -> plt.Figure:
    """Plot three OOD visualisation panels:
       (1) Confidence score histograms (ID vs OOD),
       (2) Precision-Recall curve over threshold sweep,
       (3) F1 score over threshold sweep with the chosen threshold marked.

    Args:
        ood_results:  Dict returned by src.ood.run_ood_pipeline().
        title:        Figure suptitle.
        figures_dir:  Save directory.
        filename:     File name.

    Returns:
        matplotlib Figure object.
    """
    _ensure_figures_dir(figures_dir)

    curve       = ood_results["curve"]
    threshold   = ood_results["threshold"]
    test_metrics = ood_results["test_metrics"]

    # Use test confidences for the histogram
    id_conf  = ood_results["test_id_conf"]
    ood_conf = ood_results["test_ood_conf"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Panel 1: Confidence histogram ---
    ax = axes[0]
    ax.hist(id_conf,  bins=40, alpha=0.65, label="In-distribution", color="steelblue",
            density=True)
    ax.hist(ood_conf, bins=40, alpha=0.65, label="OOD",             color="tomato",
            density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: Precision-Recall curve ---
    ax = axes[1]
    ax.plot(curve["recall"], curve["precision"], linewidth=1.8, color="darkorange")
    # Mark the chosen threshold
    idx = int(np.argmin(np.abs(curve["thresholds"] - threshold)))
    ax.scatter(curve["recall"][idx], curve["precision"][idx],
               color="black", zorder=5, s=60,
               label=f"Chosen (P={test_metrics['precision']:.2f}, "
                     f"R={test_metrics['recall']:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOD Precision-Recall Curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: F1 over thresholds ---
    ax = axes[2]
    ax.plot(curve["thresholds"], curve["f1"], linewidth=1.8, color="mediumseagreen")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Best thr={threshold:.3f}  F1={test_metrics['f1']:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title("OOD F1 vs Threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if filename is None:
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  [saved] {figures_dir / filename}")
    return fig


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def build_results_table(
    results: Dict[str, Dict],
) -> str:
    """Format a markdown summary table from multiple experiment results.

    Args:
        results: Dict mapping experiment_name → metrics dict (from compute_metrics()).

    Returns:
        Markdown-formatted table string suitable for printing in a notebook cell.
    """
    header = "| Experiment | Accuracy | Macro F1 |"
    sep    = "|---|---|---|"
    rows   = [header, sep]
    for name, m in results.items():
        rows.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    return "\n".join(rows)
