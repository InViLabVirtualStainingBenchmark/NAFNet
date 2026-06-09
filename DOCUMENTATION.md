--- 
# NAFNet: BCI Virtual Staining
Local setup and initial training for H&E → IHC translation · BCI & MIST Datasets · April 2026

> **Note:** This documents the local development setup and smoke training on Windows 11 (CPU only).
> For HPC training, inference, and evaluation on the CalcUA cluster, see [HPC-Instruction.md](HPC-INSTRUCTION.md).
> See the [official NAFNet repository](https://github.com/megvii-research/NAFNet) for the original codebase.

---

## 1. Project Goal

This adapts NAFNet (Nonlinear Activation Free Network, originally designed for image denoising and deblurring) to perform **virtual histological staining**:

- **Input:** H&E stained histology tiles
- **Output:** IHC stained equivalents

Trained and evaluated on the **BCI** and **MIST** datasets as part of the InViLab Virtual Staining Benchmark.

---

## 2. Environment Setup

Local setup ran on **Windows 11, CPU only** — no GPU available on the development machine.

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.11.0+cpu |
| TorchVision | 0.26.0+cpu |
| BasicSR | 1.2.0+efac49d (local install via `setup.py develop`) |
| OpenCV | 4.13.0 |

Create a virtual environment and activate it:

```powershell
python -m venv nafnet_env        # name it whatever you prefer
nafnet_env\Scripts\activate
```

---

## 3. Installation

```powershell
git clone https://github.com/InViLabVirtualStainingBenchmark/NAFNet
cd NAFNet

# 1. Remove opencv-python from requirements.txt (conflicts with HPC system OpenCV)
# 2. Install dependencies
pip install -r requirements.txt

# 3. Install NAFNet as local package
python setup.py develop --no_cuda_ext

# 4. Apply PyTorch 2.x compatibility patches
python patch_torch2.py

# 5. Fix TorchVision deprecation — see Fix 3 below

# 6. Create missing __init__.py
New-Item basicsr/__init__.py -ItemType File

# 7. Verify
python -c "import basicsr; print(basicsr.__file__)"
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

---

## 4. Compatibility Fixes & Modifications

NAFNet ships with BasicSR 1.x which requires several patches for modern Python/PyTorch environments. These are applied once during setup and committed to the repo.

### Fix 1 — Missing `basicsr/__init__.py`

The NAFNet repo ships without `basicsr/__init__.py`. Without it, Python treats `basicsr/` as a namespace package and cannot resolve `create_model`:

```powershell
New-Item basicsr/__init__.py -ItemType File
```

> After setup, verify `import basicsr` resolves to `...\NAFNet\basicsr\__init__.py` and not `site-packages`. If it resolves to site-packages, run `pip uninstall basicsr -y` then reinstall via `python setup.py develop --no_cuda_ext`.

### Fix 2 — PyTorch 2.x Compatibility (`patch_torch2.py`)

BasicSR was written for PyTorch 1.x. Two breaking changes in PyTorch 2.x need patching:

| Location | Old Code | New Code |
|----------|----------|----------|
| `basicsr/models/*.py` | `torch.cuda.amp.autocast` | `torch.amp.autocast('cuda')` |
| `basicsr/utils/*.py` | `from torch._six import string_classes` | `string_classes = str` |

Run once after setup — edits files in place:
```powershell
python patch_torch2.py
```

### Fix 3 — TorchVision `functional_tensor` Removed

TorchVision 0.26 removed `functional_tensor`. Fix in `basicsr/data/degradations.py` line 8:

```python
# FROM
from torchvision.transforms.functional_tensor import rgb_to_grayscale
# TO
from torchvision.transforms.functional import rgb_to_grayscale
```

### Fix 4 — pip basicsr Overrides Local basicsr

Installing `basicsr==1.4.2` via pip shadows the local NAFNet `basicsr/` folder:

```powershell
pip uninstall basicsr -y
python setup.py develop --no_cuda_ext
```

### Modification — `basicsr/models/image_restoration_model.py`

Added saving of the H&E input image (`_he.png`) alongside prediction and ground truth during inference. The `lq` tensor is extracted from `visuals` before `del self.lq` is called, then saved with the `_he` suffix.

---

## 5. Dataset Preparation

BasicSR's `PairedImageDataset` pairs images strictly **by filename** — if `train_HE/` contains `00001.png`, `train_IHC/` must also contain `00001.png`. Any count or filename mismatch causes an `AssertionError` at startup.

Local smoke training used a small subset (20 images) with this structure:

```
datasets/BCI/
├── train_HE/        ← H&E input tiles
├── train_IHC/       ← IHC ground truth tiles
├── val_HE/
└── val_IHC/

datasets/MIST/
├── ER/
│   ├── train_HE/
│   ├── train_IHC/
│   ├── val_HE/
│   └── val_IHC/
├── HER2/   (same structure)
├── Ki67/   (same structure)
└── PR/     (same structure)
```

> For HPC training, datasets are stored as SquashFS images with a neutral `HE/` / `IHC/` structure and symlinked at runtime. See [HPC-Instruction.md](HPC-INSTRUCTION.md).

---

## 6. Network Architecture

NAFNet uses a U-Net style architecture with nonlinear-activation-free blocks. The width64 variant was used throughout the benchmark:

| Parameter | Value |
|-----------|-------|
| `width` | 64 |
| `enc_blk_nums` | [2, 2, 4, 8] |
| `middle_blk_num` | 12 |
| `dec_blk_nums` | [2, 2, 2, 2] |

> width32 (~17M params) was considered but width64 (~67M params) was chosen for benchmark quality consistency with other models.

---

## 7. Training

Local smoke training uses the `-local-smoke` configs (20 iterations, CPU):

```powershell
# BCI
python basicsr/train.py -opt options/train/BCI/NAFNet-BCI-local-smoke.yml

# MIST
python basicsr/train.py -opt options/train/MIST/NAFNet-MIST-ER-local-smoke.yml
python basicsr/train.py -opt options/train/MIST/NAFNet-MIST-HER2-local-smoke.yml
python basicsr/train.py -opt options/train/MIST/NAFNet-MIST-Ki67-local-smoke.yml
python basicsr/train.py -opt options/train/MIST/NAFNet-MIST-PR-local-smoke.yml
```

Full benchmark training (100k iterations) runs on HPC — see [HPC-Instruction.md](HPC-INSTRUCTION.md).

---

## 8. Inference

Inference uses `basicsr/test.py` with the corresponding test config:

```powershell
# BCI
python basicsr/test.py -opt options/test/BCI/NAFNet-BCI-local-smoke.yml

# MIST
python basicsr/test.py -opt options/test/MIST/NAFNet-MIST-ER-local-smoke.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-MIST-HER2-local-smoke.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-MIST-Ki67-local-smoke.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-MIST-PR-local-smoke.yml
```

Results are saved to `results/` (created automatically on first run) with three images per sample:

| File | Content |
|------|---------|
| `{name}.png` | Predicted IHC |
| `{name}_gt.png` | Ground truth IHC |
| `{name}_he.png` | H&E input |

To view side-by-side comparisons locally:

```powershell
python scripts/view_results.py --dataset BCI
```

---

## 9. Smoke Test Results

### Inference — Pretrained SIDD Model (Untrained Baseline)

> The SIDD pretrained model was trained on camera sensor noise — not pathology images. These numbers are the zero-shot baseline before any domain-specific training.

| Dataset | PSNR | SSIM |
|---------|------|------|
| BCI (20 test images) | 14.34 dB | 0.3157 |
| MIST HER2 (20 test images) | 5.68 dB | 0.0170 |
| MIST PR (20 test images) | 6.10 dB | 0.0126 |

### Training — After 20 Iterations

| Dataset | PSNR | SSIM | vs Baseline |
|---------|------|------|-------------|
| BCI | 16.08 dB | 0.2367 | +1.74 dB |
| MIST HER2 | 15.51 dB | 0.0831 | +9.83 dB |
| MIST PR | 11.64 dB | 0.1712 | +5.54 dB |
| MIST Ki67 | 14.75 dB | 0.1139 | — |
| MIST ER | 11.60 dB | 0.0913 | — |

> Even 20 iterations produced significant improvement — especially MIST HER2 (+9.83 dB). Full HPC training results (100k iters) are in [HPC-Instruction.md](HPC-INSTRUCTION.md).

---

## 10. Pretrained Models

| Model | Original Task | PSNR | Location |
|-------|--------------|------|----------|
| `NAFNet-SIDD-width64.pth` | Image Denoising (SIDD) | 40.30 dB | `experiments/pretrained_models/` |

Download from the gdrive links in the original NAFNet README.

To run the pretrained denoising demo as a sanity check:

```powershell
python basicsr/demo.py `
  -opt options/test/SIDD/NAFNet-width64.yml `
  --input_path ./demo/noisy.png `
  --output_path ./results/test_output.png
```