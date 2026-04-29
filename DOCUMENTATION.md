# NAFNet Virtual Staining — Repository Documentation
**Project:** Virtual Staining Benchmark | KdG Hogeschool | April 2026  
**Status:** Inference + Smoke Training confirmed working on BCI and MIST datasets
 
---
 
## 1. Repository Profile
 
| Field | Value |
|---|---|
| Model Name | NAFNet (Nonlinear Activation Free Network) |
| Original Task | Image Restoration (Denoising / Deblurring) |
| Adapted Task | Virtual Staining (Image-to-Image Translation) |
| Pairing Mode | Paired — `PairedImageDataset` (BasicSR) |
| Scale | 1 (same resolution in/out — no upsampling) |
| Framework | BasicSR 1.2.0 (local install via setup.py develop) |
| Repo | InViLab fork of official NAFNet repository |
 
---
 
## 2. Environment Setup
 
### Local Machine (Windows 11 — CPU only)
 
| Component | Version |
|---|---|
| Python | 3.11 (venv via `nafnet_env/`) |
| PyTorch | 2.11.0+cpu |
| TorchVision | 0.26.0+cpu |
| BasicSR | 1.2.0+efac49d (local install) |
| OpenCV | 4.13.0 (pip) |
| num_gpu | 1 in all configs (NAFNet falls back to CPU gracefully) |
 
### Activation Command
```powershell
cd C:\Users\edobo\IdeaProjects\NAFNet
nafnet_env\Scripts\activate
```
 
### HPC (UA Tier2 — future)
> When moving to HPC: load modules instead of pip installing torch/opencv.  
> The same configs work — only `num_gpu`, `num_worker_per_gpu`, and dataset paths need updating.  
> Datasets must be transferred separately via rsync (excluded from git).
 
```bash
module load Python/3.9.25
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib
```
 
---
 
## 3. Critical Setup Fixes Applied
 
### Fix 1 — Missing `basicsr/__init__.py`
The NAFNet repo ships without `basicsr/__init__.py`. Without it, Python treats `basicsr/` as a namespace package and cannot resolve `create_model`.
 
```powershell
New-Item basicsr/__init__.py -ItemType File
```
 
### Fix 2 — PyTorch 2.x Compatibility (`patch_torch2.py`)
BasicSR was written for PyTorch 1.x. Two breaking changes in PyTorch 2.x need patching across `basicsr/` source files:
 
| Location | Old Code | New Code |
|---|---|---|
| `basicsr/models/*.py` | `torch.cuda.amp.autocast` | `torch.amp.autocast('cuda')` |
| `basicsr/utils/*.py` | `from torch._six import string_classes` | `string_classes = str` |
 
Run `patch_torch2.py` once after setup — it edits the files in place.
 
### Fix 3 — TorchVision `functional_tensor` Removed
TorchVision 0.26 removed `functional_tensor`. Fix in:
```
nafnet_env\Lib\site-packages\basicsr\data\degradations.py  (line 8)
```
```python
# FROM
from torchvision.transforms.functional_tensor import rgb_to_grayscale
# TO
from torchvision.transforms.functional import rgb_to_grayscale
```
 
### Fix 4 — opencv-python Conflict on HPC
`requirements.txt` ships with `opencv-python`. On HPC this conflicts with the system OpenCV module. Remove it:
```txt
# opencv-python  ← removed, HPC provides OpenCV/4.8.1-foss-2023a-contrib
```
 
### Fix 5 — pip basicsr Overrides Local basicsr
Installing `basicsr==1.4.2` via pip shadows the local NAFNet `basicsr/` folder. Solution:
```powershell
pip uninstall basicsr -y
python setup.py develop --no_cuda_ext
python -c "import basicsr; print(basicsr.__file__)"
# Must print: ...\NAFNet\basicsr\__init__.py (NOT site-packages)
```
 
---
 
## 4. Installation Order
 
```powershell
git clone https://github.com/InViLabVirtualStainingBenchmark/NAFNet
cd NAFNet
 
# 1. Fix requirements.txt — remove opencv-python line
# 2. Install deps
pip install -r requirements.txt
 
# 3. Install NAFNet as local package
python setup.py develop --no_cuda_ext
 
# 4. Apply PyTorch 2.x patches
python patch_torch2.py
 
# 5. Fix degradations.py (torchvision)
# Edit nafnet_env\Lib\site-packages\basicsr\data\degradations.py line 8
 
# 6. Create missing __init__.py
New-Item basicsr/__init__.py -ItemType File
 
# 7. Verify
python -c "import basicsr; print(basicsr.__file__)"
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```
 
---
 
## 5. Dataset Handling
 
### How PairedImageDataset Works
BasicSR's `PairedImageDataset` pairs images strictly **by filename**. If `input/` contains `00001.png`, it expects `target/` to also contain `00001.png`. Any mismatch in count or filename causes an `AssertionError` at startup.
 
---
 
### A. BCI Dataset — H&E to IHC
 
| Field | Value |
|---|---|
| Input (lq) | HE — Hematoxylin & Eosin stained |
| Target (gt) | IHC — Immunohistochemistry stained |
| Test pairs used | 20 matched images |
| Train pairs used | 20 (same as test — smoke only) |
| Issue | First download was incomplete — HE and IHC had different filenames |
| Fix | Redownloaded full dataset — filenames now match exactly |
 
**Folder structure used:**
```
datasets/BCI/
    input/        ← dataroot_lq (HE images)
    target/       ← dataroot_gt (IHC images)
    train/
        HE/       ← dataroot_lq (training)
        IHC/      ← dataroot_gt (training)
```
 
---
 
### B. MIST Dataset — H&E to Protein Marker
 
| Field | Value |
|---|---|
| Modalities | ER, HER2, Ki67, PR |
| Input (A) | H&E stained tissue |
| Target (B) | Specific protein marker stain |
| Pairs used per modality | 20 val images (smoke test) |
 
**Folder structure used:**
```
datasets/MIST/
    HER2/
        trainA/   ← dataroot_lq (training input)
        trainB/   ← dataroot_gt (training target)
        valA/     ← dataroot_lq (validation input)
        valB/     ← dataroot_gt (validation target)
    ER/    (same structure)
    Ki67/  (same structure)
    PR/    (same structure)
 
datasets/MiST_HER2/    ← flat structure used for test-only inference
    input/
    target/
```
 
---
 
## 6. Configuration Files
 
### Network Architecture Parameters
 
| Parameter | Value |
|---|---|
| `width` | 64 |
| `enc_blk_nums` | [2, 2, 4, 8] |
| `middle_blk_num` | 12 |
| `dec_blk_nums` | [2, 2, 2, 2] |
 
> width32 = ~17M params (faster, lower quality)  
> width64 = ~67M params (slower, better quality) — used throughout
 
### Training Config — Key Fields
 
| Field | Local Value | HPC Value |
|---|---|---|
| `num_gpu` | 1 (falls back to CPU) | 1 (real GPU) |
| `num_worker_per_gpu` | 1 | 4 |
| `batch_size_per_gpu` | 1 | 4 |
| `gt_size` | 256 | 256 |
| `total_iter` | 20 (smoke) | 200,000 (full) |
| `val_freq` | 20 | 5,000 |
| `save_img` | false | true |
| Optimizer | AdamW, lr=1e-3, betas=[0.9, 0.9] | same |
| Loss | PSNRLoss | same |
| Scheduler | TrueCosineAnnealingLR | same |
| `dist_params backend` | gloo | nccl |
 
### All Config Files
 
| File | Location | Purpose |
|---|---|---|
| `NAFNet-width64.yml` | `options/test/BCI/` | BCI inference |
| `NAFNet-width64-HER2.yml` | `options/test/MIST/` | MIST HER2 inference |
| `NAFNet-width64-PR.yml` | `options/test/MIST/` | MIST PR inference |
| `NAFNet-width64-ER.yml` | `options/test/MIST/` | MIST ER inference |
| `NAFNet-width64-Ki67.yml` | `options/test/MIST/` | MIST Ki67 inference |
| `NAFNet-width64.yml` | `options/train/BCI/` | BCI training |
| `NAFNet-width64-HER2.yml` | `options/train/MIST/` | MIST HER2 training |
| `NAFNet-width64-PR.yml` | `options/train/MIST/` | MIST PR training |
| `NAFNet-width64-ER.yml` | `options/train/MIST/` | MIST ER training |
| `NAFNet-width64-Ki67.yml` | `options/train/MIST/` | MIST Ki67 training |
 
---
 
## 7. Execution Commands
 
### Activate Environment
```powershell
cd C:\Users\edobo\IdeaProjects\NAFNet
nafnet_env\Scripts\activate
```
 
### Pretrained Model Demo (Sanity Check)
```powershell
python basicsr/demo.py `
  -opt options/test/SIDD/NAFNet-width64.yml `
  --input_path ./demo/noisy.png `
  --output_path ./results/test_output.png
```
 
### BCI
```powershell
# Inference (pretrained SIDD model — baseline)
python basicsr/test.py -opt options/test/BCI/NAFNet-width64.yml
 
# Smoke training (20 iters)
python basicsr/train.py -opt options/train/BCI/NAFNet-width64.yml
```
 
### MIST
```powershell
# Inference
python basicsr/test.py -opt options/test/MIST/NAFNet-width64-HER2.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-width64-PR.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-width64-ER.yml
python basicsr/test.py -opt options/test/MIST/NAFNet-width64-Ki67.yml
 
# Training (run sequentially)
python basicsr/train.py -opt options/train/MIST/NAFNet-width64-HER2.yml; `
python basicsr/train.py -opt options/train/MIST/NAFNet-width64-PR.yml; `
python basicsr/train.py -opt options/train/MIST/NAFNet-width64-ER.yml; `
python basicsr/train.py -opt options/train/MIST/NAFNet-width64-Ki67.yml
```
 
### View Results (Side-by-Side Comparison)
```powershell
python -c "
from PIL import Image, ImageDraw
import os
 
vis_path = './results/NAFNet-BCI-width64-test/visualization/BCI_test'
input_path = './datasets/BCI/input'
target_path = './datasets/BCI/target'
 
files = os.listdir(vis_path)[:3]
for f in files:
    name = os.path.splitext(f)[0].replace('_gt', '')
    input_img = Image.open(os.path.join(input_path, name + '.png'))
    output_img = Image.open(os.path.join(vis_path, f))
    target_img = Image.open(os.path.join(target_path, name + '.png'))
    w, h = input_img.size
    combined = Image.new('RGB', (w*3 + 20, h + 30), (255,255,255))
    combined.paste(input_img, (0, 30))
    combined.paste(output_img, (w + 10, 30))
    combined.paste(target_img, (w*2 + 20, 30))
    draw = ImageDraw.Draw(combined)
    draw.text((w//2 - 20, 5), 'HE Input', fill='black')
    draw.text((w + w//2 - 10, 5), 'NAFNet Output', fill='black')
    draw.text((w*2 + w//2, 5), 'IHC Target', fill='black')
    combined.show()
"
```
 
---
 
## 8. Smoke Test Results
 
### Inference — Pretrained SIDD Model (Untrained Baseline)
 
> These numbers are expected to be low — the model was trained on camera noise, not pathology images. They are the baseline before any domain-specific training.
 
| Dataset | PSNR | SSIM |
|---|---|---|
| BCI (20 test images) | 14.34 dB | 0.3157 |
| MIST HER2 (20 test images) | 5.68 dB | 0.0170 |
| MIST PR (20 test images) | 6.10 dB | 0.0126 |
 
### Training — After 20 Iterations (Smoke)
 
| Dataset | PSNR | SSIM | vs Baseline |
|---|---|---|---|
| BCI | 16.08 dB | 0.2367 | +1.74 dB |
| MIST HER2 | 15.51 dB | 0.0831 | +9.83 dB |
| MIST PR | 11.64 dB | 0.1712 | +5.54 dB |
| MIST Ki67 | 14.75 dB | 0.1139 | — |
| MIST ER | 11.60 dB | 0.0913 | — |
 
> Even 20 iterations produced significant improvement — especially MiST HER2 (+9.83 dB). Full training on HPC (200k iters) is expected to push these much higher.
 
---
 
## 9. Pretrained Models
 
| Model | Task | PSNR | Location |
|---|---|---|---|
| `NAFNet-SIDD-width64.pth` | Image Denoising | 40.30 dB (SIDD) | `experiments/pretrained_models/` |
 
Download from: README gdrive links in the original NAFNet repo.
 
---
 
## 10. HPC Migration Checklist
 
Make these changes to **all config files** before running on HPC:
 
| Local Setting | HPC Setting |
|---|---|
| `num_gpu: 1` (CPU fallback) | `num_gpu: 1` (real GPU) |
| `num_worker_per_gpu: 1` | `num_worker_per_gpu: 4` |
| `batch_size_per_gpu: 1` | `batch_size_per_gpu: 4` |
| `total_iter: 20` | `total_iter: 200000` |
| `val_freq: 20` | `val_freq: 5000` |
| `dist_params backend: gloo` | `dist_params backend: nccl` |
 
### Dataset Transfer
```bash
rsync -avz --exclude='__pycache__' \
  datasets/BCI/  user@login.hpc.uantwerpen.be:~/NAFNet/datasets/BCI/
 
rsync -avz --exclude='__pycache__' \
  datasets/MIST/ user@login.hpc.uantwerpen.be:~/NAFNet/datasets/MIST/
 
rsync -avz \
  experiments/pretrained_models/ \
  user@login.hpc.uantwerpen.be:~/NAFNet/experiments/pretrained_models/
```
 
### HPC SLURM Job Script Template
```bash
#!/bin/bash
#SBATCH --job-name=nafnet_train_bci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
 
module load Python/3.9.25
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib
 
cd $SLURM_SUBMIT_DIR
 
python basicsr/train.py -opt options/train/BCI/NAFNet-width64.yml
```
 
---
 
## 11. Next Steps
 
| Step | Task |
|---|---|
| 1 | Push NAFNet repo — all configs + fixes + `.gitignore` |
| 2 | Transfer full datasets to HPC via rsync |
| 3 | Update all configs for HPC (num_gpu, workers, total_iter) |
| 4 | Submit SLURM jobs — one per dataset |
| 5 | Monitor training logs — check PSNR curves |
| 6 | Run test configs with fully trained models |
| 7 | Compare PSNR/SSIM: NAFNet vs HAT → benchmark table |
| 8 | CellPose integration on NAFNet outputs for downstream evaluation |