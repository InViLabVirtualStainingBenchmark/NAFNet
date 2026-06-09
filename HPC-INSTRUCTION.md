--- 
# NAFNet: HPC Virtual Staining Benchmark
H&E → IHC Translation · BCI & MIST Datasets · CalcUA HPC (Vaughan A100)

> This documents HPC training, inference, and evaluation for NAFNet as part of the InViLab Virtual Staining Benchmark. For local setup and initial BCI experiments, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Table of Contents
- [Overview](#overview)
- [Environment](#environment)
- [Cluster Structure](#cluster-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Modifications](#modifications)
- [Notes](#notes)

---

## Overview

NAFNet was originally designed for image denoising and restoration. In this benchmark, it is repurposed for **H&E → IHC virtual staining** by treating the task as paired image-to-image translation with pixel-level supervision using PSNRLoss.

**Datasets:**

| Dataset | Task | Train | Val | Test |
|---------|------|-------|-----|------|
| BCI | H&E → IHC | 3896 | 488 | 489 |
| MIST ER | H&E → ER IHC | 4153 | 500 | 500 |
| MIST HER2 | H&E → HER2 IHC | 4642 | 500 | 500 |
| MIST Ki67 | H&E → Ki67 IHC | 4361 | 500 | 500 |
| MIST PR | H&E → PR IHC | 4139 | 500 | 500 |

**Key training settings:**

| Parameter | Value |
|-----------|-------|
| Architecture | NAFNet width64 |
| Input crop size | 512 × 512 |
| Batch size | 1 |
| Total iterations | 100,000 (2 × 50k chained jobs) |
| Loss function | PSNRLoss |
| Optimizer | AdamW (lr=1e-3, betas=[0.9, 0.9]) |
| LR scheduler | TrueCosineAnnealingLR (T_max=100k) |

---

## Environment

Training runs inside an Apptainer container on the **CalcUA Vaughan cluster** (NVIDIA A100 40GB, `ampere_gpu` partition).

**Container:** `basicsr_nvidia.sif`
- Base image: `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`
- PyTorch: 2.1.2+cu121
- BasicSR: 1.2.0 (modified — see [Modifications](#modifications))
- Python: 3.9

**Container location:**
```
$VSC_SCRATCH/containers/basicsr_nvidia.sif
```

> ⚠️ The container was built with a modified version of BasicSR with custom fixes. Do not replace it with a standard BasicSR container.

---

## Cluster Structure

**Compute nodes used:**

| Partition | Node | GPU | Used for |
|-----------|------|-----|----------|
| ampere_gpu | nvam1.vaughan | 4× A100 40GB | BCI + MIST training (primary) |

**Key paths:**
```
$VSC_DATA/projects/code/NAFNet/          ← repository
$VSC_DATA/projects/jobs/                 ← SLURM job scripts
$VSC_DATA/projects/logs/                 ← job logs
$VSC_DATA/projects/outputs/              ← training checkpoints
$VSC_SCRATCH/containers/                 ← Apptainer containers
$VSC_SCRATCH/nafnet_checkpoints/         ← intermediate checkpoints (temporary)
/scratch/antwerpen/grp/ap_invilab_td_thesis/   ← shared group storage
```

---

## Dataset Preparation

All datasets are stored as **SquashFS images** (`.sqsh`) for fast HPC I/O using a neutral folder structure:

```
dataset.sqsh (mounted at /data)
├── train/
│   ├── HE/        ← H&E input images
│   └── IHC/       ← IHC ground truth images
├── val/
│   ├── HE/
│   └── IHC/
└── test/
    ├── HE/
    └── IHC/
```

**Squashfs locations (shared group storage):**
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_ER_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_HER2_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_Ki67_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_PR_neutral.sqsh
```

**Runtime symlinks:**

BasicSR's `PairedImageDataset` expects `train_HE/`, `train_IHC/` etc. Since the neutral squashfs uses `HE/` and `IHC/`, job scripts create symlinks at runtime inside the container:

```bash
mkdir -p /tmp/bci
ln -s /data/train/HE  /tmp/bci/train_HE
ln -s /data/train/IHC /tmp/bci/train_IHC
ln -s /data/val/HE    /tmp/bci/val_HE
ln -s /data/val/IHC   /tmp/bci/val_IHC
ln -s /data/test/HE   /tmp/bci/test_HE
ln -s /data/test/IHC  /tmp/bci/test_IHC
```

The training configs then point to `/tmp/bci/train_HE` etc.

---

## Training

### How Training Works

Training is split into **two chained SLURM jobs** (50k iters each = 100k total) because a single job would exceed the 23-hour wall time limit at `gt_size=512`.

- **Part 1**: Trains from scratch, iter 0 → 50k. Saves `net_g_best.pth` when validation PSNR improves.
- **Part 2**: Resumes from the `50000.state` checkpoint, continues iter 50k → 100k. The learning rate schedule spans the full 100k range (`T_max=100000`) across both parts so the cosine annealing is correct end-to-end.

### Job Scripts

```
$VSC_DATA/projects/jobs/
├── submit_nafnet_BCI_512_100k.sh           ← chains part1 + part2 with --dependency=afterok
├── train_nafnet_BCI_512_100k_part1.sh
├── train_nafnet_BCI_512_100k_part2.sh
├── submit_nafnet_MIST_ER_512_100k.sh
├── train_nafnet_MIST_ER_512_100k_part1.sh
├── train_nafnet_MIST_ER_512_100k_part2.sh
└── ... (HER2, Ki67, PR — same pattern)
```

### Submitting Training

**BCI:**
```bash
bash $VSC_DATA/projects/jobs/submit_nafnet_BCI_512_100k.sh
```

**MIST (all 4 biomarkers):**
```bash
for marker in ER HER2 Ki67 PR; do
    bash $VSC_DATA/projects/jobs/submit_nafnet_MIST_${marker}_512_100k.sh
done
```

### Output Structure

The best validation checkpoint is saved to `$VSC_DATA`:

```
$VSC_DATA/projects/outputs/nafnet_BCI_512_100k/
├── checkpoints/
│   └── net_g_best.pth      ← best validation checkpoint (used for inference)
├── gpu_usage_p1.csv
├── gpu_usage_p2.csv
└── train_log.txt
```

Intermediate checkpoints are stored in `$VSC_SCRATCH` due to VSC_DATA quota constraints (25G limit):

```
$VSC_SCRATCH/nafnet_checkpoints/
├── net_g_50000.pth / 50000.state   ← end of part1, used by part2 to resume
├── net_g_75000.pth / 75000.state
├── net_g_90000.pth / 90000.state
└── nafnet_BCI_part1_net_g_latest.pth
```

> Note: scratch storage is not permanent. The only checkpoint that matters long-term is `net_g_best.pth` in `$VSC_DATA`.

### Monitoring

```bash
# Check running jobs
squeue -u vsc21216 --format="%.18i %.35j %.8T %.10M %R"

# Watch training progress
tail -f $VSC_DATA/projects/logs/nafnet_BCI_512_100k_p1_<JOBID>.out

# Check validation PSNR
grep "Validation" $VSC_DATA/projects/logs/nafnet_BCI_512_100k_p1_<JOBID>.out

# Check quota
myquota
```

---

## Inference

Inference runs the trained model on **full 1024×1024 test images** (no cropping). BasicSR handles full-resolution inference automatically.

The modified `image_restoration_model.py` saves three images per test sample:

| File | Content |
|------|---------|
| `{name}.png` | Predicted IHC |
| `{name}_gt.png` | Ground truth IHC |
| `{name}_he.png` | H&E input |

### Inference Config

```
options/test/BCI/NAFNet-BCI-512-100k-test.yml
```

Key settings:
- `dataroot_gt: /tmp/bci/test_IHC`
- `dataroot_lq: /tmp/bci/test_HE`
- `pretrain_network_g: /checkpoint/net_g_best.pth`
- `save_img: true`

### Running Inference

Benchmark inference uses the unified `benchmark_inference.py` script:

```bash
sbatch $VSC_DATA/projects/jobs/run_benchmark_BCI_nafnet.sh
```

Results are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_inference/nafnet_BCI/
├── comparison/      ← side-by-side PNGs (HE | predicted | GT)
├── predicted/       ← predicted IHC only
├── metrics.csv      ← per-image PSNR and SSIM
└── summary.txt      ← average PSNR and SSIM
```

---

## Evaluation

Evaluation uses the shared `evaluate.py` script from the InViLab benchmark repository, run inside the `evaluate_nvidia.sif` container on the `broadwell` (CPU) partition of Leibniz.

**Metrics computed:** PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID

Results are appended to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_results.csv
```

---

## Results

### BCI Dataset

| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|-----------|--------------|-------------|-------|-------|
| NAFNet (512 crop, 100k iters) | **23.13 dB** | **0.6121** | — | — | — | — | — |

*MS-SSIM, LPIPS, MAE, FID will be updated after evaluation completes.*

### MIST Dataset

| Model | Marker | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | FID ↓ |
|-------|--------|--------|--------|--------------|-------|
| NAFNet | ER | — | — | — | — |
| NAFNet | HER2 | — | — | — | — |
| NAFNet | Ki67 | — | — | — | — |
| NAFNet | PR | — | — | — | — |

*MIST training in progress. Results will be updated after inference and evaluation complete.*

---

## Modifications

### `basicsr/models/image_restoration_model.py`

Added saving of the H&E input image (`_he.png`) alongside prediction and ground truth during inference. The `lq` tensor is extracted from `visuals` before `del self.lq` is called, then saved with the `_he` suffix.

### `basicsr/__init__.py`

Created — missing from the original repo. Required for Python to treat `basicsr/` as a proper package rather than a namespace package.

---

## Notes

- **Loss values are negative** during training — `PSNRLoss` returns `-PSNR`, so minimizing the loss maximizes PSNR. A loss of `-25` corresponds to ~25 dB PSNR.
- **Validation uses full images** (1024×1024), not patches. Only training applies the 512×512 random crop.
- **Job chaining** is required because 100k iterations at ~0.18 sec/iter (~5 hours per 50k) with validation overhead exceeds safe margins for a single 23-hour job.
- **Always use the neutral squashfs** (`BCI.sqsh`, `MIST_*_neutral.sqsh`) with the runtime symlink pattern. Do not use the old `BCI_basicsr_split.sqsh` — it has been deleted.
- The `--launcher none` flag in `test_nafnet.sh` is required to disable distributed inference when running on a single GPU.
- **VSC_DATA quota** is 25G. Only `net_g_best.pth` lives there permanently — intermediate checkpoints are kept in `$VSC_SCRATCH/nafnet_checkpoints/` which is not backed up.