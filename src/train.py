"""
src/train.py
============
Training and validation loop functions.

Responsibilities:
  - train_one_epoch(): runs one full pass over the training DataLoader,
    updating model weights via backpropagation.
  - validate_one_epoch(): runs one full pass over the validation DataLoader
    without updating weights (torch.no_grad()).
  - train(): the outer training loop that calls both functions epoch by epoch,
    handles early stopping, best-model checkpointing, and history logging.
  - compute_class_weights(): computes inverse-frequency class weights to pass
    to CrossEntropyLoss, addressing the 14:1 RA/Non-RA imbalance.

All functions accept a 'mode' argument ('image_only' or 'multimodal') so the
same code works for both the baseline image-only model and the future multimodal
extension.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import MetricsLogger, BEST_MODEL, REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: compute_class_weights
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(
    class_counts: dict,
    device:       torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Formula: weight[c] = total_samples / (num_classes * count[c])

    This gives higher weight to the minority class (Non-RA) so the loss
    function penalises missing a Non-RA sample more than missing an RA sample.

    Example with this dataset:
        Non-RA: 74 samples in train  → weight = 793 / (2 * 74)  = 5.36
        RA    : 719 samples in train → weight = 793 / (2 * 719) = 0.55

    Parameters
    ----------
    class_counts : dict {0: n_non_ra, 1: n_ra}
    device       : torch.device

    Returns
    -------
    weights : torch.Tensor of shape (num_classes,)
    """
    num_classes   = len(class_counts)
    total_samples = sum(class_counts.values())

    weights = []
    for cls_idx in sorted(class_counts.keys()):
        count  = class_counts[cls_idx]
        w      = total_samples / (num_classes * count)
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print("[train] Class weights for CrossEntropyLoss:")
    class_names = ["Non-RA", "RA"]
    for i, w in enumerate(weights):
        print(f"  class {i} ({class_names[i]}): "
              f"count={class_counts[i]:4d}  weight={w:.4f}")

    return weight_tensor


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: train_one_epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    mode:       str = "image_only",
) -> tuple:
    """
    Run one training epoch: forward pass → compute loss → backprop → update.

    Parameters
    ----------
    model     : the RAClassifier
    loader    : DataLoader for the training split
    criterion : loss function (weighted CrossEntropyLoss)
    optimizer : e.g. Adam or SGD
    device    : cpu or cuda
    mode      : 'image_only' or 'multimodal'

    Returns
    -------
    avg_loss : float  — mean loss across all batches
    accuracy : float  — fraction of correctly classified samples
    """
    model.train()   # set to training mode (enables dropout, batch norm updates)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for batch_idx, (images, meta, labels) in enumerate(loader):
        # ── Move data to the compute device ───────────────────────────────────
        images = images.to(device)
        meta   = meta.to(device)
        labels = labels.to(device)

        # ── Zero the gradients from the previous batch ────────────────────────
        # PyTorch accumulates gradients by default; we must clear them each step.
        optimizer.zero_grad()

        # ── Forward pass ──────────────────────────────────────────────────────
        if mode == "image_only":
            logits = model(images)        # shape: (B, 2)
        else:
            logits = model(images, meta)  # shape: (B, 2)

        # ── Compute loss ──────────────────────────────────────────────────────
        loss = criterion(logits, labels)

        # ── Backward pass: compute gradients ──────────────────────────────────
        loss.backward()

        # ── Update weights using the computed gradients ───────────────────────
        optimizer.step()

        # ── Accumulate metrics ────────────────────────────────────────────────
        total_loss += loss.item() * images.size(0)   # loss.item() is mean over batch

        predicted      = logits.argmax(dim=1)        # class with the highest logit
        correct       += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: validate_one_epoch
# ─────────────────────────────────────────────────────────────────────────────
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    mode:      str = "image_only",
) -> tuple:
    """
    Run one validation epoch (no weight updates).

    torch.no_grad() disables gradient computation entirely, saving memory
    and making inference faster.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()   # set to eval mode (disables dropout, uses running BN stats)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(device)
            meta   = meta.to(device)
            labels = labels.to(device)

            if mode == "image_only":
                logits = model(images)
            else:
                logits = model(images, meta)

            loss           = criterion(logits, labels)
            total_loss    += loss.item() * images.size(0)
            predicted      = logits.argmax(dim=1)
            correct       += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: train
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    criterion:       nn.Module,
    optimizer:       torch.optim.Optimizer,
    device:          torch.device,
    num_epochs:      int   = 20,
    patience:        int   = 7,
    mode:            str   = "image_only",
    checkpoint_path: Path  = None,
    scheduler:       Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> MetricsLogger:
    """
    Full training loop with:
      - epoch-by-epoch loss and accuracy logging
      - best-model checkpointing (saved whenever val_loss improves)
      - early stopping (stops if val_loss has not improved for `patience` epochs)
      - optional learning rate scheduler

    Parameters
    ----------
    model           : RAClassifier
    train_loader    : DataLoader for training split
    val_loader      : DataLoader for validation split
    criterion       : weighted CrossEntropyLoss
    optimizer       : Adam or SGD
    device          : cpu or cuda
    num_epochs      : maximum training epochs
    patience        : early-stopping patience (epochs without val improvement)
    mode            : 'image_only' or 'multimodal'
    checkpoint_path : where to save the best model weights
    scheduler       : optional LR scheduler (e.g. ReduceLROnPlateau)

    Returns
    -------
    logger : MetricsLogger containing the full training history
    """
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger           = MetricsLogger(REPORTS_DIR / "training_history.csv")
    best_val_loss    = float("inf")
    epochs_no_improve = 0

    print("=" * 65)
    print(f"Training for up to {num_epochs} epochs  |  patience={patience}")
    print(f"Best model → {checkpoint_path}")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):

        # ── Training ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mode
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, mode
        )

        # ── Learning rate scheduler step ──────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            # ReduceLROnPlateau needs the metric; other schedulers take no arg
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ── Log metrics ───────────────────────────────────────────────────────
        logger.log(
            epoch      = epoch,
            train_loss = train_loss,
            val_loss   = val_loss,
            train_acc  = train_acc,
            val_acc    = val_acc,
            lr         = current_lr,
        )

        # ── Best-model checkpointing ───────────────────────────────────────────
        # We checkpoint whenever validation loss improves.
        # Saving state_dict() only stores the weights, not the whole object,
        # which is safer and more portable.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "val_loss"   : val_loss,
                    "val_acc"    : val_acc,
                },
                checkpoint_path,
            )
            print(f"  [checkpoint] New best val_loss={val_loss:.4f} — model saved.")
        else:
            epochs_no_improve += 1

        # ── Early stopping ─────────────────────────────────────────────────────
        # If validation loss has not improved for `patience` consecutive epochs,
        # we stop training to avoid wasting time and overfitting.
        if epochs_no_improve >= patience:
            print(f"\n[train] Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Save the complete training history CSV ─────────────────────────────────
    logger.save()

    print()
    print(f"[train] Training finished. Best val_loss = {best_val_loss:.4f}")
    print(f"[train] Best model saved at: {checkpoint_path}")

    return logger
