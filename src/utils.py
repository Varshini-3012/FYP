"""
src/utils.py
============
Utility helpers shared across all modules.

Responsibilities:
  - Set a global random seed for reproducibility.
  - Define canonical project paths so every module agrees on where files live.
  - Provide a simple console + CSV logger for training metrics.
  - Provide a small helper to select the compute device (GPU if available, else CPU).
"""

import os
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pandas as pd


# ── Project root ──────────────────────────────────────────────────────────────
# This file lives at  FYP/src/utils.py
# So two levels up  (src → FYP)  gives the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Canonical sub-paths ───────────────────────────────────────────────────────
DATASET_DIR   = PROJECT_ROOT / "Dataset" / "ra" / "Segmentation"
REPORTS_DIR   = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR     = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR    = PROJECT_ROOT / "models"

# Manifest CSVs produced by the preprocessing notebooks
TRAIN_CSV     = REPORTS_DIR / "train_manifest.csv"
VAL_CSV       = REPORTS_DIR / "val_manifest.csv"
TEST_CSV      = REPORTS_DIR / "test_manifest.csv"
NORM_STATS    = REPORTS_DIR / "normalisation_stats.csv"

# Default model checkpoint path
BEST_MODEL    = MODELS_DIR / "ra_resnet18_best.pth"

# Ensure output directories exist (safe to call multiple times)
for _d in [REPORTS_DIR, PLOTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Class constants ───────────────────────────────────────────────────────────
# The two classes in the dataset, in label order (index 0 = Non-RA, index 1 = RA)
CLASS_NAMES  = ["Non-RA", "RA"]
NUM_CLASSES  = 2

# ── Image preprocessing constants ─────────────────────────────────────────────
IMG_SIZE     = (224, 224)   # (height, width) — standard for ImageNet backbones


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: set_seed
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """
    Fix every relevant random-number generator so results are reproducible.

    PyTorch, NumPy, and Python's own random module all maintain separate
    internal states, so we must seed all three.

    Parameters
    ----------
    seed : int
        The integer seed to use everywhere. Default 42.
    """
    random.seed(seed)                          # Python built-in RNG
    np.random.seed(seed)                       # NumPy RNG
    torch.manual_seed(seed)                    # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU RNG (no-op if no GPU)

    # Make CuDNN deterministic (slower but reproducible).
    # Only relevant when CUDA is available, but safe to set always.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: get_device
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Return the best available compute device.

    Checks for CUDA (NVIDIA GPU) first, then falls back to CPU.
    On this machine, CUDA is not available so this will return cpu.

    Returns
    -------
    torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Using device: {device}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: load_norm_stats
# ─────────────────────────────────────────────────────────────────────────────
def load_norm_stats() -> tuple:
    """
    Load the per-channel mean and standard deviation computed from the
    training images in the preprocessing notebook (Notebook 04).

    These values are used to standardise every image before it enters
    the neural network.

    Returns
    -------
    mean : list[float]   e.g. [0.2504, 0.2504, 0.2504]
    std  : list[float]   e.g. [0.2531, 0.2531, 0.2531]
    """
    assert NORM_STATS.exists(), (
        f"Normalisation stats file not found at {NORM_STATS}.\n"
        f"Please run notebooks/04_preprocessing.ipynb first."
    )
    df   = pd.read_csv(NORM_STATS)
    mean = df["mean"].tolist()
    std  = df["std"].tolist()
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: MetricsLogger
# ─────────────────────────────────────────────────────────────────────────────
class MetricsLogger:
    """
    Logs per-epoch training and validation metrics to the console and to a
    CSV file so you can inspect or plot them later.

    Usage
    -----
        logger = MetricsLogger(REPORTS_DIR / "training_history.csv")
        logger.log(epoch=1, train_loss=0.5, val_loss=0.4,
                   train_acc=0.8, val_acc=0.85)
        logger.save()   # writes the CSV
    """

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.records   = []   # list of dicts, one per epoch

    def log(self, **kwargs) -> None:
        """
        Record one row of metrics. Pass any keyword arguments.
        Also prints the values to the console so you see progress in real time.
        """
        # Add a timestamp so you know when each epoch ran
        kwargs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.records.append(kwargs)

        # Console output
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in kwargs.items() if k != "timestamp"]
        print("  [log] " + "  |  ".join(parts))

    def save(self) -> None:
        """Write all logged rows to the CSV file."""
        df = pd.DataFrame(self.records)
        df.to_csv(self.save_path, index=False)
        print(f"[utils] Training history saved → {self.save_path}")

    def to_dataframe(self) -> pd.DataFrame:
        """Return logged records as a DataFrame (useful for plotting)."""
        return pd.DataFrame(self.records)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: check_patient_leakage
# ─────────────────────────────────────────────────────────────────────────────
def check_patient_leakage(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> dict:
    """
    Explicitly check whether any patient ID (Normalized PatientID) appears
    in more than one split.

    The dataset inspection (Notebook 01) found 17 patients spanning splits.
    This function reports that overlap clearly so it is never invisible.

    IMPORTANT: This function REPORTS but does NOT change the split.
    The original split is preserved for reproducibility.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames loaded from the manifest CSVs.

    Returns
    -------
    dict with keys:
        train_val_overlap  : set of patient IDs in both train and val
        train_test_overlap : set of patient IDs in both train and test
        val_test_overlap   : set of patient IDs in both val and test
    """
    col = "Normalized PatientID"

    train_pts = set(train_df[col].unique())
    val_pts   = set(val_df[col].unique())
    test_pts  = set(test_df[col].unique())

    tv_overlap  = train_pts & val_pts
    tt_overlap  = train_pts & test_pts
    vt_overlap  = val_pts   & test_pts

    print("=" * 60)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 60)
    print(f"  Train patients : {len(train_pts)}")
    print(f"  Val   patients : {len(val_pts)}")
    print(f"  Test  patients : {len(test_pts)}")
    print()
    print(f"  Train ∩ Val  overlap : {len(tv_overlap)} patients")
    print(f"  Train ∩ Test overlap : {len(tt_overlap)} patients")
    print(f"  Val   ∩ Test overlap : {len(vt_overlap)} patients")
    print()

    if tt_overlap:
        print(f"  WARNING: {len(tt_overlap)} patient(s) appear in BOTH train AND test.")
        print(f"  Patient IDs: {sorted(tt_overlap)}")
        print()
        print("  EXPLANATION: The original dataset split from the authors is")
        print("  preserved as-is for reproducibility. This means a small number")
        print("  of patients have exams in multiple splits. This can cause")
        print("  'patient-level leakage': the model may have seen the same patient")
        print("  at a different time point during training, making test performance")
        print("  slightly optimistic.")
        print()
        print("  RECOMMENDATION: For rigorous evaluation, re-split using a strict")
        print("  patient-stratified GroupShuffleSplit. See the OPTIONAL section")
        print("  at the end of Notebook 05 for a ready-to-use example.")
    else:
        print("  Train and Test have NO patient overlap — evaluation is clean.")

    print("=" * 60)

    return {
        "train_val_overlap" : tv_overlap,
        "train_test_overlap": tt_overlap,
        "val_test_overlap"  : vt_overlap,
    }
