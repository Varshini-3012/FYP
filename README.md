# P63 – Multimodal Deep Learning for Autoimmune Disease Diagnosis

**Final Year Project** | SSN College of Engineering  
**Student:** Varshini | **Phase implemented:** Rheumatoid Arthritis (RA) classification

---

## Project Overview

This project develops a deep learning pipeline to assist in diagnosing autoimmune diseases from medical imaging and clinical metadata.

**Current phase:** Rheumatoid Arthritis (RA) — image-only classification using hand radiographs.  
**Future phases:** Multiple Sclerosis (MS) integration and multimodal fusion.

---

## Dataset

**RAM-H1200-v1** — Radiographic Assessment of the Multi-joint  
Source: [TokyoTechMagicYang/RAM-H1200-v1](https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1)

- 1200 hand X-ray images (BMP format, 6 imaging centres in Japan)
- Labels: RA (1120 images) vs Non-RA (80 images) — severe class imbalance (~14:1)
- Original train/val/test split: 793 / 140 / 267 images

> **Note:** The dataset is not included in this repository (12+ GB).  
> Download from Hugging Face and place in `Dataset/ra/`.

---

## Project Structure

```
FYP/
├── notebooks/
│   ├── 01_dataset_overview.ipynb        # Dataset structure & class distribution
│   ├── 02_image_analysis.ipynb          # Image validation & pixel statistics
│   ├── 03_metadata_analysis.ipynb       # Clinical metadata analysis
│   ├── 04_preprocessing.ipynb           # Preprocessing pipeline & split manifests
│   ├── 05_train_ra_cnn.ipynb            # ResNet-18 training (imbalanced baseline)
│   ├── 05B_train_ra_balanced_subsets.ipynb  # Balanced-subset experiments (12×)
│   └── 06_gradcam_ra.ipynb              # Grad-CAM explainability (post-training)
│
├── src/
│   ├── utils.py       # Paths, seeds, logging, leakage checks
│   ├── dataset.py     # RADataset, transforms, augmentation
│   ├── model.py       # ResNet-18 wrapper (image-only & multimodal modes)
│   ├── train.py       # Training loop, early stopping, checkpointing
│   └── evaluate.py    # Metrics, confusion matrix, ROC curve, plots
│
├── outputs/
│   ├── plots/         # All generated figures (PNG)
│   └── reports/       # CSV manifests, metrics, training history
│
├── models/            # Model checkpoints (*.pth — not in git, too large)
│   ├── ra_resnet18_best.pth
│   └── ra/balanced/   # Per-experiment balanced-subset checkpoints
│
├── app/               # Streamlit demo (future)
└── Dataset/           # Not in git — download separately
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision pandas numpy pillow scikit-learn matplotlib seaborn openpyxl
```

### 2. Download the dataset

```python
from datasets import load_dataset
ds = load_dataset("TokyoTechMagicYang/RAM-H1200-v1")
```

Place in `Dataset/ra/Segmentation/{train,val,test}/`.

### 3. Run notebooks in order

```
01 → 02 → 03 → 04 → 05 (or 05B) → 06
```

Each notebook is self-contained, saves its outputs, and passes them to the next.

---

## Results (Notebook 05 — Imbalanced Baseline)

| Metric | Value |
|---|---|
| Best checkpoint | Epoch 1 (early stopped after epoch 8) |
| Best val loss | 0.4002 |
| Best val accuracy | 89.3% |
| Test accuracy | 84.6% |
| Test ROC-AUC | 0.8698 |
| Test F1 macro | 0.4815 |
| Test Non-RA F1 | 0.0465 ⚠️ |

**Key finding:** The model overfits from epoch 2 onward (training loss 0.15, val loss 0.66 at stop).  
The 14:1 class imbalance and only 2 Non-RA images in the test set make standard metrics unreliable.

---

## Balanced-Subset Experiments (Notebook 05B)

12 experiments, each training on ~54 Non-RA + ~49–54 RA images (≈100–108 total).  
All experiments use the same unchanged val and test sets.

See `outputs/reports/05B_experiment_results.csv` for results after running.

---

## Architecture

- **Model:** ResNet-18 (pretrained ImageNet, fine-tuned)
- **Input:** 224 × 224 RGB (grayscale BMPs converted to RGB)
- **Output:** 2-class logits (Non-RA / RA)
- **Optimiser:** Adam, lr=1e-4, weight_decay=1e-4
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping:** patience=7 (baseline), 10 (balanced experiments)
- **GradCAM target:** `model.backbone.layer4`

---

## Known Limitations

1. **Severe class imbalance** — 14:1 RA:Non-RA in the original split
2. **Only 2 Non-RA test images** — Non-RA test metrics are statistically unreliable
3. **10 patients span train and test** — mild leakage inherited from authors' split
4. **CPU-only training** — no GPU available; training times are long
5. **Small training sets in 05B** — 100–108 images per experiment; high variance expected

---

## Technologies

Python 3.12 · PyTorch 2.14 · torchvision 0.29 · scikit-learn 1.7 · pandas · NumPy · Pillow · Matplotlib · Seaborn
