#!/bin/bash
#SBATCH --job-name=nafnet_MIST_Ki67_512_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p arcturus_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/nafnet_MIST_Ki67_512_100k_p2_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/nafnet_MIST_Ki67_512_100k_p2_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/basicsr_rocm.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_Ki67_neutral.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/nafnet_MIST_Ki67_512_100k"
CKPT_DIR="$VSC_SCRATCH/nafnet_checkpoints"

mkdir -p "$OUTPUT_DIR"

srun apptainer exec \
    --rocm \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$OUTPUT_DIR":/output \
    -B "$CKPT_DIR":/checkpoints \
    "$CONTAINER" \
    bash -c "
    mkdir -p /tmp/mist
    ln -sf /data/train/HE  /tmp/mist/train_HE
    ln -sf /data/train/IHC /tmp/mist/train_IHC
    ln -sf /data/val/HE    /tmp/mist/val_HE
    ln -sf /data/val/IHC   /tmp/mist/val_IHC
    ln -sf /data/test/HE   /tmp/mist/test_HE
    ln -sf /data/test/IHC  /tmp/mist/test_IHC
    bash /code/NAFNet/train_nafnet.sh \
    options/train/MIST/NAFNet-MIST-Ki67-512-100k-part2.yml
    "
