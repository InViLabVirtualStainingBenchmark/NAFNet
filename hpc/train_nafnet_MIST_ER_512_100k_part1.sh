#!/bin/bash
#SBATCH --job-name=nafnet_MIST_ER_512_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/nafnet_MIST_ER_512_100k_p1_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/nafnet_MIST_ER_512_100k_p1_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_ER_neutral.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/nafnet_MIST_ER_512_100k"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 > "$OUTPUT_DIR/gpu_usage_p1.csv" &
GPU_LOG_PID=$!

srun apptainer exec \
    --nv \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$OUTPUT_DIR":/output \
    "$CONTAINER" \
    bash -c "
    mkdir -p /tmp/mist
    ln -s /data/train/HE  /tmp/mist/train_HE
    ln -s /data/train/IHC /tmp/mist/train_IHC
    ln -s /data/val/HE    /tmp/mist/val_HE
    ln -s /data/val/IHC   /tmp/mist/val_IHC
    ln -s /data/test/HE   /tmp/mist/test_HE
    ln -s /data/test/IHC  /tmp/mist/test_IHC
    bash /code/NAFNet/train_nafnet.sh \
    options/train/MIST/NAFNet-MIST-ER-512-100k-part1.yml
    "

kill $GPU_LOG_PID || true
