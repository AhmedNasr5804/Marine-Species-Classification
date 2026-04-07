"""
ood.py — Out-of-Distribution (OOD) detection via Maximum Softmax Probability (MSP).

Method
------
Hendrycks & Gimpel (2017) "A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks" (ICLR 2017) showed that a
simple threshold on the maximum softmax confidence is a surprisingly strong
OOD detector that outperforms many more complex approaches.

Algorithm:
    1. Run a forward pass to get class logits.
    2. Apply softmax to obtain a probability distribution over known classes.
    3. Compute confidence = max(softmax_probs).
    4. Flag as OOD if confidence < threshold.

Rationale for MSP as the baseline (§4.4):
    - No architectural changes or auxiliary training required.
    - Works with any trained classifier without retraining.
    - Interpretable: confidence directly reflects the model's certainty.
    - Limitation: overconfident softmax on far-OOD inputs is known (ODIN paper),
      but MSP remains the accepted baseline to beat.

Threshold selection:
    The threshold is tuned on the validation set by sweeping over candidate values
    and selecting the one that maximises the F1 score for OOD detection. This is
    the standard practice to avoid using the test set for hyperparameter decisions.

Evaluation outputs (§4.4):
    - Precision and recall for OOD class detection at the chosen threshold.
    - Precision-Recall curve over the full threshold sweep for visualisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_confidences(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on *loader* and return per-sample MSP confidences and labels.

    Args:
        model:   Trained classifier in eval mode.
        loader:  DataLoader (can be in-distribution test or OOD loader).
        device:  Compute device.

    Returns:
        Tuple (confidences, labels):
          - confidences: shape (N,), max softmax probability per sample.
          - labels:      shape (N,), ground-truth integer labels
                         (-1 for OOD samples).
    """
    model.eval()
    all_confidences: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc="  extracting confidences", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.cpu().numpy()
        all_confidences.append(confidence)
        all_labels.append(labels.numpy())

    return np.concatenate(all_confidences), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    id_confidences: np.ndarray,
    ood_confidences: np.ndarray,
    n_thresholds: int = 200,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Sweep thresholds and return the one maximising OOD-detection F1.

    The sweep evaluates every candidate threshold on a combined pool of
    in-distribution (ID) and OOD samples. A sample is predicted OOD when its
    confidence falls below the threshold.

    Args:
        id_confidences:  MSP scores for in-distribution (known-class) samples.
        ood_confidences: MSP scores for OOD (unknown-class) samples.
        n_thresholds:    Number of candidate thresholds to evaluate.

    Returns:
        Tuple (best_threshold, curve_dict):
          - best_threshold: float that maximises OOD F1.
          - curve_dict: {
                "thresholds":  array of candidate thresholds,
                "precision":   OOD precision at each threshold,
                "recall":      OOD recall at each threshold,
                "f1":          OOD F1 at each threshold,
                "fpr":         false positive rate (ID samples flagged as OOD),
            }
    """
    # Binary labels: 1 = OOD, 0 = in-distribution
    scores = np.concatenate([id_confidences,  ood_confidences])
    y_true = np.concatenate([
        np.zeros(len(id_confidences),  dtype=int),
        np.ones( len(ood_confidences), dtype=int),
    ])

    lo = scores.min()
    hi = scores.max()
    thresholds = np.linspace(lo, hi, n_thresholds)

    precisions, recalls, f1s, fprs = [], [], [], []

    for thr in thresholds:
        y_pred = (scores < thr).astype(int)   # predict OOD if confidence < threshold

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr)

    f1s_arr = np.array(f1s)
    best_idx = int(f1s_arr.argmax())
    best_threshold = float(thresholds[best_idx])

    curve = {
        "thresholds": thresholds,
        "precision":  np.array(precisions),
        "recall":     np.array(recalls),
        "f1":         f1s_arr,
        "fpr":        np.array(fprs),
    }
    return best_threshold, curve


# ---------------------------------------------------------------------------
# Evaluation at a fixed threshold
# ---------------------------------------------------------------------------

def evaluate_ood(
    id_confidences: np.ndarray,
    ood_confidences: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute OOD detection metrics at a fixed threshold.

    Args:
        id_confidences:  MSP scores for in-distribution samples.
        ood_confidences: MSP scores for OOD samples.
        threshold:       Decision boundary (flag as OOD if confidence < threshold).

    Returns:
        Dict with keys: "precision", "recall", "f1", "fpr",
                        "accuracy", "threshold".
    """
    scores = np.concatenate([id_confidences, ood_confidences])
    y_true = np.concatenate([
        np.zeros(len(id_confidences),  dtype=int),
        np.ones( len(ood_confidences), dtype=int),
    ])
    y_pred = (scores < threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
        "accuracy":  round(accuracy,  4),
        "threshold": round(threshold, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# High-level convenience function (used directly in the notebook)
# ---------------------------------------------------------------------------

def run_ood_pipeline(
    model: nn.Module,
    val_id_loader: DataLoader,
    val_ood_loader: DataLoader,
    test_id_loader: DataLoader,
    test_ood_loader: DataLoader,
    device: torch.device,
    n_thresholds: int = 200,
) -> Dict:
    """Full OOD evaluation pipeline.

    Step 1: Extract MSP confidences on val ID + val OOD splits.
    Step 2: Tune threshold on val set (maximise OOD F1).
    Step 3: Evaluate at the chosen threshold on the held-out test splits.
    Step 4: Return all results + PR curve data for plotting.

    Args:
        model:            Trained classifier (eval mode expected).
        val_id_loader:    DataLoader for validation in-distribution samples.
        val_ood_loader:   DataLoader for validation OOD samples.
        test_id_loader:   DataLoader for test in-distribution samples.
        test_ood_loader:  DataLoader for test OOD (unknown) samples.
        device:           Compute device.
        n_thresholds:     Threshold sweep resolution.

    Returns:
        Dict with keys:
          "threshold"      — tuned threshold value
          "val_metrics"    — OOD metrics on the val set at tuned threshold
          "test_metrics"   — OOD metrics on the test set at tuned threshold
          "curve"          — PR/F1/FPR curve over threshold sweep (for plotting)
          "val_id_conf"    — raw val ID confidences
          "val_ood_conf"   — raw val OOD confidences
          "test_id_conf"   — raw test ID confidences
          "test_ood_conf"  — raw test OOD confidences
    """
    print("[OOD] Extracting confidences on validation set …")
    val_id_conf,  _ = extract_confidences(model, val_id_loader,  device)
    val_ood_conf, _ = extract_confidences(model, val_ood_loader, device)

    print("[OOD] Tuning threshold on validation set …")
    threshold, curve = tune_threshold(val_id_conf, val_ood_conf, n_thresholds)
    val_metrics = evaluate_ood(val_id_conf, val_ood_conf, threshold)

    print(f"[OOD] Best threshold = {threshold:.4f}  "
          f"(val F1={val_metrics['f1']:.4f}  "
          f"prec={val_metrics['precision']:.4f}  "
          f"rec={val_metrics['recall']:.4f})")

    print("[OOD] Extracting confidences on test set …")
    test_id_conf,  _ = extract_confidences(model, test_id_loader,  device)
    test_ood_conf, _ = extract_confidences(model, test_ood_loader, device)

    print("[OOD] Evaluating on test set …")
    test_metrics = evaluate_ood(test_id_conf, test_ood_conf, threshold)

    print(f"[OOD] Test  — F1={test_metrics['f1']:.4f}  "
          f"prec={test_metrics['precision']:.4f}  "
          f"rec={test_metrics['recall']:.4f}  "
          f"FPR={test_metrics['fpr']:.4f}")

    return {
        "threshold":     threshold,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "curve":         curve,
        "val_id_conf":   val_id_conf,
        "val_ood_conf":  val_ood_conf,
        "test_id_conf":  test_id_conf,
        "test_ood_conf": test_ood_conf,
    }
