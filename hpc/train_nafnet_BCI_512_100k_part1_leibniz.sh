#!/bin/bash
#SBATCH --job-name=nafnet_BCI_512_100k_p1_leib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/nafnet_BCI_512_100k_p1_leib_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/nafnet_BCI_512_100k_p1_leib_%j.err
set -euo pipefail
CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/nafnet_BCI_512_100k_leibniz"
mkdir -p "$OUTPUT_DIR"
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
    mkdir -p /tmp/bci
    ln -s /data/train/HE  /tmp/bci/train_HE
    ln -s /data/train/IHC /tmp/bci/train_IHC
    ln -s /data/val/HE    /tmp/bci/val_HE
    ln -s /data/val/IHC   /tmp/bci/val_IHC
    ln -s /data/test/HE   /tmp/bci/test_HE
    ln -s /data/test/IHC  /tmp/bci/test_IHC
    bash /code/NAFNet/train_nafnet.sh \
    options/train/BCI/NAFNet-BCI-512-100k-part1.yml
    "
kill $GPU_LOG_PID || true
