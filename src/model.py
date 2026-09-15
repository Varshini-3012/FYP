"""
src/model.py
============
ResNet-18 model adapted for RA binary classification.

Design decisions:
  - We use transfer learning: start from ImageNet-pretrained weights.
    The network has already learned low-level features (edges, textures)
    from millions of natural images. Fine-tuning it on 793 X-rays is much
    more effective than training from scratch.
  - The final fully-connected layer is replaced to output 2 classes
    (Non-RA / RA) instead of the original 1000 ImageNet classes.
  - A metadata fusion head is included as an optional extension: clinical
    variables (age, sex, etc.) are concatenated with the image features
    before the final classifier. This is the 'multimodal' part of the FYP.
    The default mode ('image_only') skips metadata fusion so we can first
    establish a solid image-only baseline.
  - The model is structured so that Grad-CAM can be applied to the last
    convolutional layer (layer4) with no architecture changes.

GradCAM hook points:
    model.backbone.layer4   ← target layer for Grad-CAM heatmaps
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: RAClassifier
# ─────────────────────────────────────────────────────────────────────────────
class RAClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for RA vs Non-RA binary classification.

    Two operating modes:
      'image_only'  — image features → Linear(512, 2)
      'multimodal'  — image features + metadata → Linear(512 + n_meta, 256)
                      → ReLU → Dropout → Linear(256, 2)

    Parameters
    ----------
    num_classes    : int   — number of output classes (2 for RA/Non-RA)
    pretrained     : bool  — load ImageNet weights (strongly recommended)
    freeze_backbone: bool  — if True, only the classifier head is trained.
                             Useful for very small datasets or quick experiments.
                             Default False: all layers are trained (fine-tuning).
    mode           : str   — 'image_only' or 'multimodal'
    n_meta_features: int   — number of metadata features (only used in 'multimodal')
    dropout_rate   : float — dropout probability in the classifier head
    """

    def __init__(
        self,
        num_classes:     int   = 2,
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
        mode:            str   = "image_only",
        n_meta_features: int   = 5,
        dropout_rate:    float = 0.5,
    ):
        super().__init__()

        self.mode            = mode
        self.n_meta_features = n_meta_features
        self.num_classes     = num_classes

        # ── Load ResNet-18 backbone ────────────────────────────────────────────
        # 'pretrained=True' downloads ImageNet weights the first time (≈44 MB).
        # Set pretrained=False to train from scratch (not recommended here).
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # How many features come out of ResNet-18's last pooling layer?
        # The original ResNet-18 fc layer is: Linear(512, 1000)
        # So the feature dimension is 512.
        in_features = backbone.fc.in_features  # 512

        # ── Remove the original 1000-class head ───────────────────────────────
        # We keep everything up to (and including) the adaptive average pool,
        # then add our own head. Using nn.Identity() as a placeholder so the
        # backbone attribute structure stays intact for Grad-CAM.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ── Optionally freeze backbone weights ────────────────────────────────
        # If freeze_backbone=True, gradients will not flow back into ResNet
        # and only the classifier head's weights will be updated.
        # This is useful when you have very few training examples.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[model] Backbone frozen — only classifier head will be trained.")

        # ── Classifier head ───────────────────────────────────────────────────
        if mode == "image_only":
            # Simple single linear layer
            self.classifier = nn.Linear(in_features, num_classes)

        elif mode == "multimodal":
            # Image features (512) + metadata features (n_meta) → 256 → 2
            fused_dim = in_features + n_meta_features
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'image_only' or 'multimodal'."
            )

        print(f"[model] RAClassifier created")
        print(f"[model]   mode            = {mode}")
        print(f"[model]   pretrained      = {pretrained}")
        print(f"[model]   freeze_backbone = {freeze_backbone}")
        print(f"[model]   num_classes     = {num_classes}")
        if mode == "multimodal":
            print(f"[model]   n_meta_features = {n_meta_features}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        images: torch.Tensor,
        meta:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (B, 3, 224, 224) float32 — batch of preprocessed images
        meta   : (B, n_meta) float32      — batch of metadata features
                 Only used when mode='multimodal'.

        Returns
        -------
        logits : (B, num_classes) float32
                 Raw scores before softmax. Use with CrossEntropyLoss.
        """
        # ── 1. Extract image features ──────────────────────────────────────────
        # ResNet-18 outputs a (B, 512) feature vector after global average pooling.
        features = self.backbone(images)   # shape: (B, 512)

        # ── 2. Fuse with metadata (multimodal mode only) ───────────────────────
        if self.mode == "multimodal":
            if meta is None:
                raise ValueError(
                    "mode='multimodal' requires meta tensor in forward(). "
                    "Pass meta=None only when mode='image_only'."
                )
            features = torch.cat([features, meta], dim=1)  # shape: (B, 512 + n_meta)

        # ── 3. Classification head ─────────────────────────────────────────────
        logits = self.classifier(features)   # shape: (B, num_classes)
        return logits

    # ── Convenience: trainable parameter count ────────────────────────────────
    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Return total parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: build_model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    mode:            str   = "image_only",
    pretrained:      bool  = True,
    freeze_backbone: bool  = False,
    n_meta_features: int   = 5,
    dropout_rate:    float = 0.5,
    device:          torch.device = None,
) -> RAClassifier:
    """
    Convenience factory function that builds, prints, and moves the model
    to the target device in one call.

    Parameters
    ----------
    mode            : 'image_only' or 'multimodal'
    pretrained      : load ImageNet weights
    freeze_backbone : freeze ResNet layers
    n_meta_features : number of metadata features (multimodal only)
    dropout_rate    : dropout in classifier head
    device          : torch.device (cpu or cuda)

    Returns
    -------
    RAClassifier instance on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RAClassifier(
        num_classes     = 2,
        pretrained      = pretrained,
        freeze_backbone = freeze_backbone,
        mode            = mode,
        n_meta_features = n_meta_features,
        dropout_rate    = dropout_rate,
    )

    model = model.to(device)

    total     = model.count_all_parameters()
    trainable = model.count_parameters()
    print(f"[model] Total parameters    : {total:,}")
    print(f"[model] Trainable parameters: {trainable:,}")
    print(f"[model] Moved to device     : {device}")

    return model
