"""
src/dataset.py
==============
PyTorch Dataset class and image transforms for RA classification.

Key design decisions:
  - Images are BMP hand X-rays, originally grayscale but stored in variable
    colour modes. We always convert to RGB (3 channels) so the same ResNet-18
    backbone works regardless of the source mode.
  - Augmentation is applied ONLY to training images, never to val/test.
    This is enforced by passing different transform objects.
  - The Dataset class is deliberately simple: it reads one row of the
    manifest CSV per __getitem__ call, opens the image from disk, and
    returns (image_tensor, label). No caching is done to keep memory use low.
  - Metadata features (Age, Sex, etc.) are included as a separate tensor
    so the model can optionally fuse them with the image features later.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: get_transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(
    split:    str,
    img_size: Tuple[int, int] = (224, 224),
    mean:     List[float]     = None,
    std:      List[float]     = None,
) -> T.Compose:
    """
    Return the torchvision transform pipeline for a given split.

    Training transforms include data augmentation to help the model
    generalise better and reduce overfitting. Validation and test
    transforms are deterministic (no randomness).

    Augmentations chosen for hand X-rays:
      - RandomHorizontalFlip: left/right hands are anatomically symmetric
      - RandomRotation(±10°): slight rotational variance from patient positioning
      - ColorJitter(brightness, contrast): simulate different X-ray exposures
      - RandomAffine(translate): small translations mimic centering differences
    We intentionally avoid aggressive augmentations (e.g. large crops, vertical
    flips) that would destroy clinically relevant anatomy.

    Parameters
    ----------
    split    : 'train', 'val', or 'test'
    img_size : (H, W) target size; default 224×224
    mean     : per-channel mean for normalisation (list of 3 floats)
    std      : per-channel std  for normalisation (list of 3 floats)

    Returns
    -------
    torchvision.transforms.Compose
    """
    # Use the training-set statistics we computed in preprocessing.
    # If not provided, fall back to the values we know from Notebook 04.
    if mean is None:
        mean = [0.2504, 0.2504, 0.2504]
    if std is None:
        std  = [0.2531, 0.2531, 0.2531]

    # Normalise: converts [0,1] → zero-centred with unit variance
    normalize = T.Normalize(mean=mean, std=std)

    if split == "train":
        # ── Training: resize + augment + convert to tensor + normalise ────────
        return T.Compose([
            T.Resize(img_size),                      # resize to 224×224
            T.RandomHorizontalFlip(p=0.5),           # flip left↔right with 50% chance
            T.RandomRotation(degrees=10),            # rotate ±10 degrees
            T.ColorJitter(brightness=0.2,            # vary brightness ±20%
                          contrast=0.2),             # vary contrast ±20%
            T.RandomAffine(degrees=0,                # no extra rotation
                           translate=(0.05, 0.05)),  # translate up to 5% of image
            T.ToTensor(),                            # uint8 PIL → float32 [0,1] tensor
            normalize,                               # standardise channels
        ])
    else:
        # ── Val / Test: resize + convert to tensor + normalise (no augmentation) ─
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: RADataset
# ─────────────────────────────────────────────────────────────────────────────
class RADataset(Dataset):
    """
    PyTorch Dataset for the RAM-H1200-v1 Rheumatoid Arthritis dataset.

    Each item returned by __getitem__ is a tuple:
        (image_tensor, meta_tensor, label)

    Where:
        image_tensor : float32 tensor of shape (3, 224, 224)
        meta_tensor  : float32 tensor of shape (n_meta_features,)
                       Contains: age_norm, pixel_spacing, Sex_enc,
                                 laterality_enc, LR_enc
        label        : int64 scalar tensor  — 0 = Non-RA, 1 = RA

    Parameters
    ----------
    manifest_csv : path to one of the split CSVs from Notebook 04
    transform    : torchvision Compose pipeline (use get_transforms())
    meta_cols    : which metadata columns to include as numeric features
    """

    # Default metadata columns to include as extra features.
    # PatientID and StudyID are explicitly excluded — they are identifiers.
    DEFAULT_META_COLS = [
        "age_norm",        # z-score normalised age
        "pixel_spacing",   # mm/pixel (acquisition parameter)
        "Sex_enc",         # 0=F, 1=M, 2=O (label-encoded)
        "laterality_enc",  # 0=L, 1=R (encoded from filename suffix)
        "LR_enc",          # 0/1/2 — from the metadata LR column
    ]

    def __init__(
        self,
        manifest_csv: str | Path,
        transform:    Optional[T.Compose] = None,
        meta_cols:    Optional[List[str]] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        assert self.manifest_csv.exists(), (
            f"Manifest CSV not found: {self.manifest_csv}\n"
            f"Run notebooks/04_preprocessing.ipynb first."
        )

        self.df        = pd.read_csv(self.manifest_csv)
        self.transform = transform
        self.meta_cols = meta_cols if meta_cols is not None else self.DEFAULT_META_COLS

        # Validate that all requested meta_cols exist in the manifest
        missing = [c for c in self.meta_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Requested meta_cols not found in manifest: {missing}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        # Convert label column to integer (safety: already int but explicit is better)
        self.df["isRA"] = self.df["isRA"].astype(int)

        # Fill any remaining NaN in meta columns with 0
        # (Notebook 04 should have no NaN, but this guards against edge cases)
        self.df[self.meta_cols] = self.df[self.meta_cols].fillna(0.0)

        print(f"[RADataset] Loaded {len(self.df)} images from {self.manifest_csv.name}")
        print(f"[RADataset] Class distribution: "
              f"Non-RA={int((self.df['isRA']==0).sum())}  "
              f"RA={int((self.df['isRA']==1).sum())}")

    # ── Required by PyTorch: total number of items ────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ── Required by PyTorch: return one item by index ─────────────────────────
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── 1. Load image ──────────────────────────────────────────────────────
        # open() is called inside a context manager to ensure the file handle
        # is closed immediately after we have the data in memory.
        try:
            with Image.open(row["full_path"]) as img:
                # Convert to RGB: ensures exactly 3 channels.
                # Grayscale BMP (mode='L') → RGB copies the single channel 3×.
                # The resulting image looks the same visually; the CNN gets 3
                # identical channels, which is fine for transfer learning.
                img = img.convert("RGB")

                # Apply the transform pipeline (resize, augment, to-tensor, normalise)
                if self.transform is not None:
                    image_tensor = self.transform(img)
                else:
                    # Fallback: just convert to tensor without augmentation
                    image_tensor = T.ToTensor()(img)

        except Exception as e:
            # If a file is unreadable (should not happen after validation),
            # return a black image rather than crashing the whole DataLoader.
            print(f"  WARNING: Could not load image '{row['full_path']}': {e}")
            print(f"  Returning zero tensor as placeholder.")
            image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        # ── 2. Build metadata feature vector ──────────────────────────────────
        meta_values  = row[self.meta_cols].values.astype(np.float32)
        meta_tensor  = torch.from_numpy(meta_values)   # shape: (n_meta_features,)

        # ── 3. Label ───────────────────────────────────────────────────────────
        label = torch.tensor(int(row["isRA"]), dtype=torch.long)

        return image_tensor, meta_tensor, label

    # ── Convenience: class counts ─────────────────────────────────────────────
    def get_class_counts(self) -> dict:
        """Return {0: n_non_ra, 1: n_ra} for use in class-weight calculations."""
        counts = self.df["isRA"].value_counts().to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def get_labels(self) -> List[int]:
        """Return all labels as a Python list. Used by WeightedRandomSampler."""
        return self.df["isRA"].tolist()
