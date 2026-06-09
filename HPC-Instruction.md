# NAFNet — Virtual Staining Benchmark
### H&E → IHC Translation · BCI & MIST Datasets · CalcUA HPC (Vaughan A100)
 
---
 
> This repository adapts **NAFNet** (Nonlinear Activation Free Network) for virtual staining — translating H&E histology images to IHC-stained equivalents. It is part of the **InViLab Virtual Staining Benchmark** comparing transformer-based image restoration models across two datasets: BCI and MIST.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Environment](#environment)
- [Repository Structure](#repository-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Modifications](#modifications)
- [Notes](#notes)
---
 
## Overview
 
NAFNet was originally designed for image denoising and restoration. In this benchmark, we repurpose it for **unpaired H&E → IHC virtual staining** by treating the task as image-to-image translation with pixel-level supervision.
 
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
| Input crop size | 512 × 512 |
| Batch size | 1 |
| Total iterations | 100,000 (2 × 50k chained jobs) |
| Loss function | PSNRLoss |
| Optimizer | AdamW (lr=1e-3) |
| LR scheduler | TrueCosineAnnealingLR (T_max=100k) |
 
---
 
## Environment
 
Training runs inside a Singularity/Apptainer container on the **CalcUA Vaughan cluster** (NVIDIA A100 40GB, `ampere_gpu` partition).
 
**Container:** `basicsr_nvidia.sif`
- Base image: `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`
- PyTorch: 2.1.2+cu121
- BasicSR: 1.2.0
- Python: 3.9
**Container location on cluster:**
```
$VSC_SCRATCH/containers/basicsr_nvidia.sif
```
 
> ⚠️ The container was built with a modified version of BasicSR with custom fixes. Do not replace it with a standard BasicSR container.
 
---
 
## Repository Structure
 
```
NAFNet/
├── basicsr/                        ← Modified BasicSR library
│   ├── train.py                    ← Main training entry point
│   ├── test.py                     ← Main inference entry point
│   ├── models/
│   │   ├── image_restoration_model.py  ← Modified to save HE + pred + GT
│   │   └── losses/
│   │       └── losses.py           ← PSNRLoss implementation
│   └── data/
│       └── paired_image_dataset.py ← Dataset loader (512 crop during train, full image during val/test)
├── options/
│   ├── train/
│   │   ├── BCI/
│   │   │   ├── NAFNet-BCI-512-100k-part1.yml   ← Training config part 1 (iter 0→50k)
│   │   │   └── NAFNet-BCI-512-100k-part2.yml   ← Training config part 2 (iter 50k→100k)
│   │   └── MIST/
│   │       ├── NAFNet-MIST-ER-512-100k-part1.yml
│   │       ├── NAFNet-MIST-ER-512-100k-part2.yml
│   │       └── ... (HER2, Ki67, PR)
│   └── test/
│       └── BCI/
│           └── NAFNet-BCI-512-100k-test.yml    ← Inference config
├── train_nafnet.sh                 ← Training wrapper (sets PYTHONPATH, calls basicsr/train.py)
└── test_nafnet.sh                  ← Inference wrapper (sets PYTHONPATH, calls basicsr/test.py)
```
 
---
 
## Dataset Preparation
 
### Squashfs Format
 
All datasets are stored as **SquashFS images** (`.sqsh`) for fast HPC I/O. Each squashfs uses a **neutral folder structure**:
 
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
 
### Runtime Symlinks
 
Since BasicSR expects `train_HE/`, `train_IHC/` etc., the job scripts create symlinks at runtime inside the container:
 
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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/cad04d4d-073b-4a9f-9cb7-62f95e780ef4" />
 
---
 
## Training
 
### How Training Works
 
Training is split into **two chained SLURM jobs** (50k iters each = 100k total) because a single job would exceed the 23-hour wall time limit at `gt_size=512`.
 
- **Part 1**: Trains from scratch, iter 0 → 50k. Saves checkpoints at 15k, 30k, 45k, 50k.
- **Part 2**: Resumes from the `50000.state` checkpoint, continues iter 50k → 100k. The learning rate schedule spans the full 100k range (`T_max=100000`) across both parts so the cosine annealing is correct.
### Job Scripts
 
```
$VSC_DATA/projects/jobs/
├── train_nafnet_BCI_512_100k_part1.sh
├── train_nafnet_BCI_512_100k_part2.sh
├── submit_nafnet_BCI_512_100k.sh          ← Submits both parts with --dependency=afterok
├── train_nafnet_MIST_ER_512_100k_part1.sh
├── train_nafnet_MIST_ER_512_100k_part2.sh
├── submit_nafnet_MIST_ER_512_100k.sh
└── ... (HER2, Ki67, PR)
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
 
```
$VSC_DATA/projects/outputs/nafnet_BCI_512_100k/
├── checkpoints/
│   ├── 15000.pth / 15000.state
│   ├── 30000.pth / 30000.state
│   ├── 45000.pth / 45000.state
│   ├── 50000.pth / 50000.state    ← Used by part2 to resume
│   ├── 100000.pth / 100000.state  ← Final checkpoint
│   └── latest.pth / latest.state
├── logs/
│   └── NAFNet-BCI-512-100k-part1/  ← TensorBoard logs
└── gpu_usage_p1.csv
```
 
### Monitoring
 
```bash
# Check running jobs
squeue -u vsc21216
 
# Watch training progress
tail -f $VSC_DATA/projects/logs/nafnet_BCI_512_100k_p1_<JOBID>.out
 
# Check validation PSNR
grep "Validation" $VSC_DATA/projects/logs/nafnet_BCI_512_100k_p1_<JOBID>.out
```
 
---
 
## Inference
 
### How Inference Works
 
Inference runs the trained model on the **full 1024×1024 test images** (no cropping). BasicSR handles full-resolution inference automatically. The modified `image_restoration_model.py` saves three images per test sample:
 
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
 
```bash
sbatch $VSC_DATA/projects/jobs/infer_nafnet_BCI_512_100k.sh
```
 
Predictions are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/transformer_prediction/NAFNet_BCI_512/
```
 
> `[ADD IMAGE: side-by-side example of HE input, predicted IHC, ground truth IHC]`
 
---
 
## Evaluation
 
Evaluation uses the shared `evaluate.py` script from the InViLab benchmark repository, run inside the `evaluate_nvidia.sif` container on the `broadwell` (CPU) partition of Leibniz.
 
**Metrics computed:**
- PSNR, SSIM, MS-SSIM
- LPIPS (AlexNet and VGG)
- MAE
- FID
- Cellpose cell detection F1 (optional, `--cellpose`)
**Results are appended to:**
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_results.csv
```
 
---
 
## Results
 
### BCI Dataset
 
| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|-----------|--------------|-------------|-------|-------|
| NAFNet (512 crop, 100k iters) | — | — | — | — | — | — | — |
 
*Results will be updated after training completes.*
 
### MIST Dataset
 
| Model | Marker | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | FID ↓ |
|-------|--------|--------|--------|--------------|-------|
| NAFNet | ER | — | — | — | — |
| NAFNet | HER2 | — | — | — | — |
| NAFNet | Ki67 | — | — | — | — |
| NAFNet | PR | — | — | — | — |
 
---
 
## Modifications
 
The following files were modified from the original NAFNet repository:
 
### `basicsr/models/image_restoration_model.py`
 
Added saving of the H&E input image (`_he.png`) alongside the prediction and ground truth during inference. The `lq` tensor is extracted from `visuals` before `del self.lq` is called, then saved with the suffix `_he.png`.
 
---
 
## Notes
 
- **Loss values are negative** during training. This is expected — `PSNRLoss` returns `-PSNR` (negative PSNR in dB), so minimizing the loss maximizes PSNR. A loss of `-25` corresponds to ~25 dB PSNR.
- **Validation uses full images** (1024×1024), not patches. Only training applies the 512×512 random crop.
- **Job chaining** is required because 100k iterations at 0.18 sec/iter (~5 hours per 50k) with validation overhead exceeds safe margins for a single 23-hour job.
- **Always use the neutral squashfs** (`BCI.sqsh`, `MIST_*_neutral.sqsh`) with the runtime symlink pattern. Do not use the old `BCI_basicsr_split.sqsh` or `MIST_*_Uformer.sqsh` files — they have been deleted.
- The `--launcher none` flag in `test_nafnet.sh` is required to disable distributed inference since we use a single GPU.
