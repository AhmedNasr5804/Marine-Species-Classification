# Marine Species Classification
Deep Learning project for marine organism image classification using **FathomNet** data, with both:
- **Model A:** custom CNN trained from scratch
- **Model B:** transfer learning with ResNet-50 strategies

The project also includes **out-of-distribution (OOD) detection** using Maximum Softmax Probability (MSP), plus full evaluation utilities (accuracy, macro F1, per-class metrics, confusion matrices, and training curves).

## Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Pipeline](#data-pipeline)
- [Modeling Approach](#modeling-approach)
- [OOD Detection](#ood-detection)
- [Evaluation and Results](#evaluation-and-results)
- [How to Run](#how-to-run)
- [Reproducibility Notes](#reproducibility-notes)
- [Future Improvements](#future-improvements)

## Project Overview
This project builds a complete image-classification pipeline for marine species:
1. Prepare and split marine image crops from FathomNet.
2. Train and compare a custom CNN and transfer-learning strategies.
3. Evaluate classification performance with detailed metrics and visualizations.
4. Detect unknown/OOD samples using a confidence-threshold baseline.

Core implementation is in `src/`, while `notebook.ipynb` runs the full experimental workflow end-to-end.

## Repository Structure
```text
.
├── notebook.ipynb
├── report.pdf
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model_a.py
│   ├── model_b.py
│   ├── ood.py
│   ├── seed.py
│   ├── train.py
│   └── transforms.py
├── figures/
│   ├── model_a_config1_confusion_matrix.png
│   ├── model_a_config1_curves.png
│   ├── model_a_config2_confusion_matrix.png
│   ├── model_a_config2_curves.png
│   ├── model_a_config_comparison.png
│   ├── model_b_feature_extraction_confusion_matrix.png
│   ├── model_b_feature_extraction_curves.png
│   ├── model_b_full_finetuning_confusion_matrix.png
│   ├── model_b_full_finetuning_curves.png
│   ├── model_b_partial_finetuning_confusion_matrix.png
│   ├── model_b_partial_finetuning_curves.png
│   ├── model_b_strategy_comparison.png
│   └── ood_detection_results.png
└── weights/
    ├── model_a_config1_best.pt
    ├── model_a_config2_best.pt
    ├── model_b_feature_extraction_best.pt
    ├── model_b_full_finetuning_best.pt
    └── model_b_partial_finetuning_best.pt
```

> Note: the runtime data directory (`data/`) is expected locally when running experiments, but is typically not committed due size.

## Environment Setup
### 1) Create and activate virtual environment
**PowerShell (Windows):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

Main dependencies include:
- `torch`, `torchvision`
- `scikit-learn`, `torchmetrics`
- `numpy`, `Pillow`, `requests`, `tqdm`
- `fathomnet`
- `matplotlib`, `seaborn`
- `jupyter`, `ipykernel`, `ipywidgets`

## Data Pipeline
Implemented in `src/dataset.py`.

### Supported flow
- `load_from_disk(...)`: scan existing `data/` class folders (no API calls)
- `download_dataset(...)`: query FathomNet and download/crop images
- `build_splits(...)`: create reproducible train/val/test/OOD splits
- `get_dataloaders(...)`: build PyTorch DataLoaders with train/eval transforms

### Class setup (default)
- **Known classes:** 12
- **OOD classes:** 3 (held out from classifier training)

Known and OOD concept names are defined as constants in `src/dataset.py`:
- `KNOWN_CONCEPTS`
- `OOD_CONCEPTS`
- `ALL_CONCEPTS`

Known classes:
- `Actiniaria`
- `Crinoidea`
- `Holothuroidea`
- `Ophiuroidea`
- `Pennatulacea`
- `Porifera`
- `Sebastolobus`
- `Nanomia`
- `Paragorgia arborea`
- `Dosidicus gigas`
- `Beroe abyssicola`
- `Bathochordaeus stygius`

OOD classes:
- `Aegina citrea`
- `Pleuroncodes planipes`
- `Acanthogorgia`

### Split strategy
- Known classes: **70/15/15** train/val/test (stratified via seeded shuffle)
- OOD classes: stored fully in OOD split with label `-1`
- Splits can be saved/loaded via `data/splits.json`

### Preprocessing
In `src/transforms.py`:
- Train: random resized crop, horizontal flip, color jitter, ImageNet normalization
- Val/Test/OOD: deterministic resize + center crop + ImageNet normalization

## Modeling Approach
### Model A (Custom CNN from Scratch)
Implemented in `src/model_a.py`.

Architecture:
- 4 convolutional blocks (`Conv + BN + ReLU + pooling`)
- Adaptive average pooling to fixed `4x4`
- MLP head with dropout

Training/eval loop is handled by `src/train.py`:
- optional AMP
- gradient clipping
- LR scheduler
- early stopping
- best-checkpoint save/restore

Two configs are compared in notebook:
- **Config 1:** `lr=1e-3`, `batch_size=32`
- **Config 2:** `lr=3e-4`, `batch_size=64`

### Model B (Transfer Learning, ResNet-50)
Implemented in `src/model_b.py`.

Three strategies:
1. **Feature Extraction** (freeze backbone, train head only)
2. **Partial Fine-Tuning** (unfreeze top block/layers + head with differential LRs)
3. **Full Fine-Tuning** (unfreeze full backbone with conservative LR + warmup)

## OOD Detection
Implemented in `src/ood.py`.

Method:
- Maximum Softmax Probability (MSP)
- Confidence threshold tuned on validation ID+OOD set (maximize F1)
- Evaluate tuned threshold on test ID+OOD set

Returns:
- threshold
- precision, recall, F1, FPR
- curve data for plotting
- raw confidence arrays

Visualization is generated via `plot_ood_curves(...)` in `src/evaluate.py`.

## Evaluation and Results
Evaluation utilities are in `src/evaluate.py`:
- `collect_predictions(...)`
- `compute_metrics(...)`
- `plot_confusion_matrix(...)`
- `plot_training_curves(...)`
- `plot_comparison_curves(...)`
- `build_results_table(...)`

### Reported classification results (from notebook run)
| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Model A — Config 1 | 0.6304 | 0.6010 |
| Model A — Config 2 | 0.6139 | 0.5913 |
| Model B — Feature Extraction | 0.7954 | 0.7836 |
| Model B — Partial Fine-Tuning | **0.8647** | **0.8608** |
| Model B — Full Fine-Tuning | 0.8515 | 0.8452 |

### Reported OOD results (from notebook run)
- Tuned threshold (validation): **0.7606**
- Validation: `F1=0.7727`, `Precision=0.6821`, `Recall=0.8909`
- Test: `F1=0.7686`, `Precision=0.6759`, `Recall=0.8909`, `FPR=0.4653`

## How to Run
### Full workflow (recommended)
Run `notebook.ipynb` top-to-bottom.

Optional quick smoke run:
```powershell
$env:QUICK_RUN="true"
jupyter notebook notebook.ipynb
```

By default in notebook:
- `EPOCHS_A = 30`
- `EPOCHS_B = 20`

### Typical notebook sequence
1. Set seed and detect device
2. Load/download data and build splits
3. Build dataloaders with transforms
4. Train Model A configs
5. Train Model B strategies
6. Run OOD pipeline
7. Compute metrics and save figures

## Reproducibility Notes
- `set_seed(42)` in `src/seed.py` seeds Python/NumPy/PyTorch and configures deterministic cuDNN settings.
- Splits can be persisted to JSON and reused.
- Saved checkpoints for all major experiments are included under `weights/`.

## Future Improvements
- Stronger OOD baselines (e.g., temperature scaling / ODIN / energy-based scores)
- Class-imbalance handling and richer augmentation policies
- More lightweight deployment model variants
- Automated experiment tracking and configuration management

