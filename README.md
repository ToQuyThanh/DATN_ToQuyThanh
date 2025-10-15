# DATN_ToQuyThanh

HAUI Graduation Thesis repository delivering a full pipeline for learning-based document unwarping and invoice normalization. The project centers on a light-weight cascaded U-Net (TinyDocUNet) to predict displacement fields that remap warped inputs to rectified outputs. This repository includes machine learning training scripts, evaluation utilities, pre-trained checkpoints, logs, and packaged web application artifacts.

## Overview
- Objective: predict 2-channel displacement maps for document images, enabling geometric unwarping via `cv2.remap`.
- Core model: TinyDocUNet (a cascade of two TinyUNets) with CMRF blocks for multi-receptive fields and efficient feature aggregation.
- Training: AdamW optimizer, cosine annealing with warm restarts, mixed precision (AMP), gradient clipping and accumulation.
- Evaluation: MS-SSIM for structural similarity and Average Distance (AD) for displacement fidelity.

## Repository Structure
```
MachineLearning/
  ModelResult/            # Pre-trained checkpoints (.pth)
  TrainingLogs/           # Metrics and plots (JSON, PNG)
  TrainingScripts/
    config.py             # Training/paths configuration
    docunet_model.py      # TinyDocUNet (baseline)
    docunet_model_c.py    # TinyDocUNet (optimized/compiled variant)
    docunet_loss.py       # Loss functions for DocUNet variants
    eval_scores.py        # MS-SSIM and AD metric computation
    invoice_dataset.py    # Dataset (classic)
    invoice_dataset_norm.py # Dataset with normalization
    GenData.ipynb         # Data generation utilities/notebook
    ProcessNorm.ipynb     # Displacement normalization and remapping demo
    TrainNotebook.ipynb   # End-to-end training pipeline notebook

Web/
  backend_invoice.zip     # Backend package (zipped)
  haui-invoice-main.zip   # Frontend package (zipped)

README.md                 # This document
```

## Machine Learning Component

### Data Layout
- Dataset roots are configured in `TrainingScripts/config.py`:
  - `trainroot = 'train_gen'`
  - `testroot  = 'test_gen'`
- Each root must contain two subfolders:
  - `images/` — input images (`.jpg`)
  - `labels/` — ground-truth displacement maps (`.npy`), sharing basenames with the images.
- Example:
  - `train_gen/images/000001.jpg`
  - `train_gen/labels/000001.npy`

### Displacement Map Format
- Predicted target is a 2-channel field `[map_x, map_y]` of shape `[2, H, W]` representing (x, y) coordinates in the source image for each pixel in the rectified output.
- In notebooks, maps can be normalized to `[-1, 1]` for model stability; convert back to pixel coordinates before remapping.

### Dataset Implementations
- `invoice_dataset_norm.ImageData` applies default normalization:
  - Images: `ToTensor()` then `Normalize(mean=[0.5]*C, std=[0.5]*C)` → input in `[-1, 1]`.
  - Labels: `ToTensor()` (expected scaled to `[0, 1]` or appropriate numeric range).
- Both datasets assume RGB (`C=3`) by default and resize to `image_size=(512, 512)`.

### Model Architecture
- `TinyDocUnet` (see `docunet_model.py` / `docunet_model_c.py`):
  - Two-stage cascade: `U_net1` produces a first displacement estimate `y1`; features are concatenated with `y1` and fed into `U_net2` to produce refined `y2`.
  - Building blocks: `CMRF` (Cascade Multi-Receptive Fields) combining pointwise and depth-wise convolutions with residual connections.
  - Input: `[N, 3, 512, 512]` by default. Output: two predictions, each `[N, 2, 512, 512]`.
- `docunet_model_c.py` includes minor graph-safety tweaks and optional `torch.compile` usage for performance.

### Loss Functions
- `DocUnetLoss`: MSE(y, label) with regularization on the mean displacement (penalizes global bias), applied to both `y1` and `y2`.
- Variants:
  - `DocUnetLoss_DL`, `DocUnetLoss_DL_batch`: single-stage versions with batch-aware reductions.
  - `DocUnetLossPow`: squared variants of the regularization terms.
  - `DocUnetLossB`, `DocUnetLoss_DL1`: add constraints (e.g., positivity) via min-based penalties.

### Evaluation Metrics
- Implemented in `eval_scores.py`:
  - `MS-SSIM`: structural similarity, computed via `pytorch_msssim`; expects inputs scaled to `[0, 1]`.
  - `AD` (Average Distance): mean L2 distance across channels per pixel.
- The training notebook periodically reports and tracks `val_loss`, `val_ms_ssim`, and `val_ad`.

## Training

### Configuration (`TrainingScripts/config.py`)
- Paths: `trainroot`, `testroot`, `output_dir`.
- Hardware: `gpu_id`, `workers`.
- Schedule: `epochs=100`, cosine warm restarts; `eval_interval=5`.
- Optimization: `lr=1e-3`, `AdamW` with `weight_decay=1e-4`.
- Stabilization:
  - Mixed precision: `use_amp=True` (if CUDA available).
  - Gradient clipping: `grad_clip=1.0`.
  - Gradient accumulation: `accumulation_steps=4`.
  - Optional `torch.compile`: `use_compile=True`.

### Running Training
- Recommended: open and run `TrainingScripts/TrainNotebook.ipynb` (the notebook contains the complete training loop, logging, checkpointing, and validation).
- Training artifacts are written to `output/` by default: periodic checkpoints, `training_metrics.json`, and `training_curves.png`.

## Pre-trained Models and Logs
- Provided checkpoints under `MachineLearning/ModelResult/`:
  - `best_ms_ssim.pth`, `best_ad.pth`, `best_val_loss.pth`, and intermediate `checkpoint_epoch_XXX.pth` files.
- Logs under `MachineLearning/TrainingLogs/`:
  - `training_curves.png`: plots of training/validation metrics.
  - `training_metrics.json`: serialized metrics and best scores.

## Inference and Unwarping
Below is a minimal example to load a checkpoint, run a forward pass, and remap an image using a predicted displacement field. Adjust paths according to your setup.

```python
import os
import cv2
import numpy as np
import torch
from MachineLearning.TrainingScripts.docunet_model_c import TinyDocUnet
from MachineLearning.TrainingScripts.invoice_dataset_norm import ImageData

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TinyDocUnet(input_channels=3, n_classes=2).to(device)

# Load checkpoint (example: full checkpoint dict)
ckpt = torch.load('MachineLearning/ModelResult/best_ms_ssim.pth', map_location=device)
state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

# Single image forward (ensure preprocessing matches training)
img_path = 'test_gen/images/000001.jpg'
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (512, 512))
tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
tensor = tensor.unsqueeze(0).to(device)

with torch.no_grad():
    y1, y2 = model(tensor)
    disp = y2.squeeze(0).cpu().numpy()  # [2, H, W]

# If disp is in [-1, 1], convert back to pixel coordinates
H, W = disp.shape[1], disp.shape[2]
map_x = ((disp[0] + 1) / 2.0) * (W - 1)
map_y = ((disp[1] + 1) / 2.0) * (H - 1)

# Remap to obtain unwarped image
map_x = map_x.astype(np.float32)
map_y = map_y.astype(np.float32)
unwarped = cv2.remap(image_resized, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
cv2.imwrite('unwarped_000001.png', cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR))
```

## Environment Setup
Use Python 3.10+ and install the following dependencies. GPU with CUDA is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install torch torchvision
pip install opencv-python numpy natsort matplotlib thop torchsummary tensorboardX pytorch_msssim pillow colorlog
```

## Web Application Artifacts
- `Web/backend-haui-invoice.zip`: backend service (unzip and follow its internal README or entry-point instructions).
- `Web/frontend-haui-invoice.zip`: frontend application (unzip and follow its internal README).
- These packages are provided as-is for deployment and demonstration; they are not covered by the training scripts above.

## Reproducibility Checklist
- Verify dataset layout under `train_gen/` and `test_gen/` with `images/` and `labels/`.
- Inspect and adjust `TrainingScripts/config.py` (paths, epochs, batch size, learning rate).
- Ensure consistent normalization between training and inference.
- Use provided notebooks for end-to-end runs and monitor `output/` for artifacts.

## Notes
- This repository is organized to present thesis results, with training performed mainly via notebooks.
- Checkpoint formats may vary; when loading, prefer reading the dictionary and extracting `state_dict` if present.
- If you encounter CUDA Graph or compilation issues, disable `torch.compile` (`use_compile=False`) and/or run in pure float32 (disable AMP).

This repository is used to deliver the results of the HAUI Graduation Thesis.
