"""
src/evaluate.py
===============
Test-set evaluation, metric computation, and result visualisation.

Responsibilities:
  - get_predictions(): run the best-saved model over the test DataLoader
    and collect all predictions and probabilities.
  - compute_metrics(): calculate accuracy, precision, recall, F1, ROC-AUC,
    and a full sklearn classification report.
  - save_metrics(): write all metrics to a CSV file.
  - plot_confusion_matrix(): save confusion matrix as a PNG.
  - plot_roc_curve(): save ROC curve as a PNG.
  - plot_training_history(): save loss and accuracy curves from training.
  - evaluate(): high-level function that does all of the above in one call.

IMPORTANT: All metrics are computed from ACTUAL model predictions.
No values are hard-coded or fabricated.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils import CLASS_NAMES, PLOTS_DIR, REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: get_predictions
# ─────────────────────────────────────────────────────────────────────────────
def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode:   str = "image_only",
) -> tuple:
    """
    Run the model over an entire DataLoader and collect results.

    Parameters
    ----------
    model  : trained RAClassifier (loaded from checkpoint)
    loader : DataLoader (typically the test loader)
    device : cpu or cuda
    mode   : 'image_only' or 'multimodal'

    Returns
    -------
    all_labels : np.ndarray int   — ground-truth labels
    all_preds  : np.ndarray int   — predicted class indices
    all_probs  : np.ndarray float — probability of class 1 (RA), shape (N,)
    """
    model.eval()

    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(device)
            meta   = meta.to(device)
            labels = labels.to(device)

            # Forward pass
            if mode == "image_only":
                logits = model(images)
            else:
                logits = model(images, meta)

            # Convert logits → probabilities using softmax
            probs = torch.softmax(logits, dim=1)   # shape: (B, 2)

            # Predicted class = index of highest probability
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of RA (class 1)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: compute_metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
) -> dict:
    """
    Compute all classification metrics from predictions.

    Because the dataset is heavily imbalanced (14:1), we report:
      - per-class precision, recall, F1
      - macro average    (treats each class equally regardless of size)
      - weighted average (weighted by support / class size)
      - ROC-AUC          (area under the ROC curve; threshold-independent)

    Note on metric choice:
      - Accuracy alone is misleading here: a model that always predicts 'RA'
        would get 93.3% accuracy but be clinically useless.
      - F1 for the minority class (Non-RA, label=0) is the key indicator of
        whether the model can actually detect Non-RA patients.
      - ROC-AUC measures the model's discriminative ability across all
        probability thresholds.

    Parameters
    ----------
    labels : ground-truth integer labels
    preds  : predicted integer labels
    probs  : probability of class 1 (RA), shape (N,)

    Returns
    -------
    dict of metric name → value
    """
    # ── Scalar metrics ────────────────────────────────────────────────────────
    acc      = accuracy_score(labels, preds)

    prec_macro   = precision_score(labels, preds, average="macro",    zero_division=0)
    rec_macro    = recall_score(   labels, preds, average="macro",    zero_division=0)
    f1_macro     = f1_score(       labels, preds, average="macro",    zero_division=0)

    prec_weighted = precision_score(labels, preds, average="weighted", zero_division=0)
    rec_weighted  = recall_score(   labels, preds, average="weighted", zero_division=0)
    f1_weighted   = f1_score(       labels, preds, average="weighted", zero_division=0)

    prec_nonra = precision_score(labels, preds, average=None, zero_division=0)[0]
    rec_nonra  = recall_score(   labels, preds, average=None, zero_division=0)[0]
    f1_nonra   = f1_score(       labels, preds, average=None, zero_division=0)[0]

    prec_ra    = precision_score(labels, preds, average=None, zero_division=0)[1]
    rec_ra     = recall_score(   labels, preds, average=None, zero_division=0)[1]
    f1_ra      = f1_score(       labels, preds, average=None, zero_division=0)[1]

    # ROC-AUC requires probability scores (not just binary predictions)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # Edge case: if only one class is present in labels (shouldn't happen)
        auc = float("nan")
        print("  WARNING: ROC-AUC could not be computed (only one class in labels).")

    metrics = {
        "accuracy"            : round(acc,          4),
        "precision_macro"     : round(prec_macro,   4),
        "recall_macro"        : round(rec_macro,    4),
        "f1_macro"            : round(f1_macro,     4),
        "precision_weighted"  : round(prec_weighted,4),
        "recall_weighted"     : round(rec_weighted, 4),
        "f1_weighted"         : round(f1_weighted,  4),
        "precision_NonRA"     : round(prec_nonra,   4),
        "recall_NonRA"        : round(rec_nonra,    4),
        "f1_NonRA"            : round(f1_nonra,     4),
        "precision_RA"        : round(prec_ra,      4),
        "recall_RA"           : round(rec_ra,       4),
        "f1_RA"               : round(f1_ra,        4),
        "roc_auc"             : round(auc,          4),
        "n_samples"           : int(len(labels)),
        "n_NonRA"             : int((labels == 0).sum()),
        "n_RA"                : int((labels == 1).sum()),
        "n_pred_NonRA"        : int((preds  == 0).sum()),
        "n_pred_RA"           : int((preds  == 1).sum()),
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: save_metrics
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(metrics: dict, save_path: Path = None) -> None:
    """Save the metrics dictionary to a CSV file."""
    if save_path is None:
        save_path = REPORTS_DIR / "test_metrics.csv"
    save_path = Path(save_path)
    pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(save_path)
    print(f"[evaluate] Test metrics saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: print_metrics
# ─────────────────────────────────────────────────────────────────────────────
def print_metrics(labels: np.ndarray, preds: np.ndarray, metrics: dict) -> None:
    """Print a human-readable metrics summary to the console."""
    print()
    print("=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples : {metrics['n_samples']}  "
          f"(Non-RA: {metrics['n_NonRA']}, RA: {metrics['n_RA']})")
    print()
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print()
    print("  Per-class metrics:")
    print(f"    Non-RA  Prec={metrics['precision_NonRA']:.4f}  "
          f"Rec={metrics['recall_NonRA']:.4f}  F1={metrics['f1_NonRA']:.4f}")
    print(f"    RA      Prec={metrics['precision_RA']:.4f}  "
          f"Rec={metrics['recall_RA']:.4f}  F1={metrics['f1_RA']:.4f}")
    print()
    print(f"  Macro avg   F1={metrics['f1_macro']:.4f}  "
          f"Prec={metrics['precision_macro']:.4f}  Rec={metrics['recall_macro']:.4f}")
    print(f"  Weighted avg F1={metrics['f1_weighted']:.4f}  "
          f"Prec={metrics['precision_weighted']:.4f}  Rec={metrics['recall_weighted']:.4f}")
    print()
    print("  Full sklearn classification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: plot_confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    labels:    np.ndarray,
    preds:     np.ndarray,
    save_path: Path = None,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    The confusion matrix shows:
        Rows    = actual labels
        Columns = predicted labels
    Each cell [i, j] shows how many samples of class i were predicted as class j.
    """
    if save_path is None:
        save_path = PLOTS_DIR / "confusion_matrix.png"

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: plot_roc_curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(
    labels:    np.ndarray,
    probs:     np.ndarray,
    save_path: Path = None,
) -> None:
    """
    Plot and save the ROC (Receiver Operating Characteristic) curve.

    The ROC curve plots True Positive Rate vs False Positive Rate at every
    possible classification threshold. The AUC (Area Under Curve) summarises
    the overall discrimination ability of the model.
    AUC = 0.5 means the model is no better than random guessing.
    AUC = 1.0 means the model is perfectly discriminating.
    """
    if save_path is None:
        save_path = PLOTS_DIR / "roc_curve.png"

    try:
        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        auc_score = roc_auc_score(labels, probs)
    except ValueError as e:
        print(f"  WARNING: Cannot plot ROC curve: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            lw=1.5, label="Random classifier (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] ROC curve saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: plot_training_history
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(
    history_csv: Path = None,
    plots_dir:   Path = None,
) -> None:
    """
    Load the training history CSV and save loss and accuracy curve plots.

    Saves four separate plots (as required by the specification):
        training_loss.png
        validation_loss.png
        training_accuracy.png
        validation_accuracy.png

    Also saves a combined overview plot:
        training_history.png
    """
    if history_csv is None:
        history_csv = REPORTS_DIR / "training_history.csv"
    if plots_dir is None:
        plots_dir = PLOTS_DIR

    history_csv = Path(history_csv)
    if not history_csv.exists():
        print(f"  WARNING: training_history.csv not found at {history_csv}. Skipping plots.")
        return

    df     = pd.read_csv(history_csv)
    epochs = df["epoch"].tolist()

    # ── Individual plots ──────────────────────────────────────────────────────
    def _save_plot(x, y, ylabel, title, filename, color):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, color=color, lw=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        p = plots_dir / filename
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[evaluate] Plot saved → {p}")

    _save_plot(epochs, df["train_loss"], "Loss",     "Training Loss",        "training_loss.png",      "#4C72B0")
    _save_plot(epochs, df["val_loss"],   "Loss",     "Validation Loss",      "validation_loss.png",    "#C44E52")
    _save_plot(epochs, df["train_acc"],  "Accuracy", "Training Accuracy",    "training_accuracy.png",  "#55A868")
    _save_plot(epochs, df["val_acc"],    "Accuracy", "Validation Accuracy",  "validation_accuracy.png","#DD8452")

    # ── Combined overview (4 subplots) ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training History — RA Classification", fontsize=14, fontweight="bold")

    axes[0][0].plot(epochs, df["train_loss"], "#4C72B0", lw=2)
    axes[0][0].set_title("Training Loss");   axes[0][0].set_ylabel("Loss")

    axes[0][1].plot(epochs, df["val_loss"], "#C44E52", lw=2)
    axes[0][1].set_title("Validation Loss"); axes[0][1].set_ylabel("Loss")

    axes[1][0].plot(epochs, df["train_acc"], "#55A868", lw=2)
    axes[1][0].set_title("Training Accuracy"); axes[1][0].set_ylabel("Accuracy")

    axes[1][1].plot(epochs, df["val_acc"], "#DD8452", lw=2)
    axes[1][1].set_title("Validation Accuracy"); axes[1][1].set_ylabel("Accuracy")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    combined_path = plots_dir / "training_history.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Combined history plot → {combined_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    mode:       str  = "image_only",
    save_dir:   Path = None,
) -> dict:
    """
    High-level evaluation function: runs the model on the test set and
    produces all metrics, plots, and CSV outputs in one call.

    Parameters
    ----------
    model    : trained RAClassifier (must already have weights loaded)
    loader   : test DataLoader
    device   : cpu or cuda
    mode     : 'image_only' or 'multimodal'
    save_dir : directory for CSV outputs (defaults to REPORTS_DIR)

    Returns
    -------
    metrics : dict of all computed metrics
    """
    if save_dir is None:
        save_dir = REPORTS_DIR

    print("[evaluate] Running test-set inference...")
    labels, preds, probs = get_predictions(model, loader, device, mode)

    print("[evaluate] Computing metrics...")
    metrics = compute_metrics(labels, preds, probs)
    print_metrics(labels, preds, metrics)

    # Save
    save_metrics(metrics, Path(save_dir) / "test_metrics.csv")
    plot_confusion_matrix(labels, preds)
    plot_roc_curve(labels, probs)
    plot_training_history()

    return metrics
