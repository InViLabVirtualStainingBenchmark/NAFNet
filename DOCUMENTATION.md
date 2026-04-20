# NAFNet — Smoke Test Documentation

---

## Model Info

- **Model name:** NAFNet + Baseline
- **Upstream repo URL:** https://github.com/megvii-research/NAFNet
- **Upstream last commit date:** 2022-08-02
- **Paper / citation:** Chen et al., "Simple Baselines for Image Restoration", ECCV 2022
- **MIST dataset citation:** Li et al., "Adaptive supervised PatchNCE loss for learning H&E-to-IHC stain translation with inconsistent groundtruth image pairs", MICCAI 2023, pp. 632–641, Springer
- **Paired or unpaired assumption:** Paired
- **Intended staining task:** H&E to IHC (BCI dataset); generalised to H&E → HER2 / ER / Ki67 / PR IHC (MIST dataset)

---

## Environment Claimed by Authors

- **Python version:** 3.9.5
- **PyTorch version:** 1.11.0
- **CUDA version:** 11.3
- **Requirements file present:** `requirements.txt`

---

## Environment Actually Used

- **Python version:** 3.11.9
- **PyTorch version:** 2.11.0
- **TorchVision version:** 0.26.0
- **CUDA version:** N/A — Apple MPS backend
- **OS:** macOS (Apple Silicon M2)
- **Date tested:** 2026-04-20
- **Hardware:** Apple M2 (MPS, `num_gpu: 1`)

Exact working package versions are in `requirements_frozen.txt`.

---

## Dataset Preparation

NAFNet is a paired model, it expects separate folders for input (LQ) and target (GT) images.

- **BCI download:** https://bupt-ai-cz.github.io/BCI/
- **MIST download:** https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MIST-HER2.md

### BCI

- **Format expected by model:** separate `HE/` and `IHC/` folders, filenames matching exactly
- **Conversion applied:** none — folders used directly
- **Smoke test subset:** 10 train pairs + 5 test pairs copied into project

```
datasets/BCI/
├── train/
│   ├── HE/    # H&E input images
│   └── IHC/   # IHC target images
└── test/
    ├── HE/
    └── IHC/
```

### MIST

- **Format expected by model:** separate `trainA/` (H&E) and `trainB/` (stained) folders, filenames matching exactly
- **Note:** local `valA/valB` folders serve as the test split
- **Conversion applied:** none — folders used directly
- **Smoke test subset:** 10 train pairs + 5 val pairs per stain copied into project

```
datasets/MIST/
├── ER/    trainA/ trainB/ valA/ valB/
├── HER2/  trainA/ trainB/ valA/ valB/
├── Ki67/  trainA/ trainB/ valA/ valB/
└── PR/    trainA/ trainB/ valA/ valB/
```

---

## Smoke Test Commands

### Installation

```bash
pip install torch torchvision
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### BCI

```bash
# Train
python basicsr/train.py -opt options/train/BCI/NAFNet-width32.yml
python basicsr/train.py -opt options/train/BCI/Baseline-width32.yml

# Test
python basicsr/test.py -opt options/test/BCI/NAFNet-width32.yml
python basicsr/test.py -opt options/test/BCI/Baseline-width32.yml
```

### MIST (repeat for each stain)

```bash
# Train
python basicsr/train.py -opt options/train/MIST/NAFNet-width32-HER2.yml
python basicsr/train.py -opt options/train/MIST/Baseline-width32-HER2.yml

# Test
python basicsr/test.py -opt options/test/MIST/NAFNet-width32-HER2.yml
python basicsr/test.py -opt options/test/MIST/Baseline-width32-HER2.yml
```

Replace `HER2` with `ER`, `Ki67`, `PR` for other stains.

---

## Changes Made to Original Code

No model architecture, loss functions, or training logic was changed. All changes are infrastructure fixes to make the code run outside the original CUDA environment.

| File | Change | Reason |
|---|---|---|
| `basicsr/models/base_model.py` line 25 | `cuda if num_gpu != 0 else cpu` → CUDA → MPS → CPU priority | Hardcoded CUDA fails on backends without CUDA support (e.g., Apple Silicon); order ensures CUDA is used first when available |
| `basicsr/train.py` lines 170–173 | `torch.cuda.current_device()` + `.cuda()` map_location → device-agnostic `map_location=device` | Hardcoded CUDA device mapping fails when loading checkpoints on systems without CUDA |
| `basicsr/metrics/psnr_ssim.py` lines 189–192 | `.cuda()` on kernel and tensors → `.to(device)` with CUDA → MPS → CPU detection | Hardcoded CUDA in SSIM 3D kernel fails during validation on systems without CUDA |
| `basicsr/models/image_restoration_model.py` line 295 | `torch.cuda.empty_cache()` → conditional on `torch.cuda.is_available()` | Avoids unnecessary CUDA calls and potential unintended initialization when CUDA is not available |
| `basicsr/models/image_restoration_model.py` line 366 | `torch.distributed.reduce()` → conditional on `self.opt['dist']` | Called unconditionally even in non-distributed mode; fails because the process group is not initialized |

> **Note on `dist_params.backend`:** `nccl` is CUDA-only. Since we use `--launcher none` (no distributed training), this setting is unused during smoke tests. Set to `gloo` as a safe default and switch to `nccl` for multi-GPU training on the HPC.

> **Note on `io_backend`:** `disk` reads raw image files directly — simple but slower for large datasets. `lmdb` reads from a pre-packed binary database (created via `scripts/data_preparation/`). Convert datasets to LMDB before running on the HPC.

---

## Summary

**Overall result: PASS**

NAFNet smoke test completed on 2026-04-20. The full pipeline (train → test) ran successfully on macOS M2 (MPS) for both BCI and all four MIST stains (HER2, ER, Ki67, PR), for both NAFNet and Baseline models. Infrastructure fixes were applied to resolve macOS and non-CUDA compatibility issues; no model logic was changed. Frozen environment saved to `requirements_frozen.txt`.