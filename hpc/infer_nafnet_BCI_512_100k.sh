#!/bin/bash
#SBATCH --job-name=infer_nafnet_BCI_512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/infer_nafnet_BCI_512_100k_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/infer_nafnet_BCI_512_100k_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh"
CHECKPOINT_DIR="$VSC_DATA/projects/outputs/nafnet_BCI_512_100k/checkpoints"
GRP_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis"
OUTPUT_DIR="$GRP_DIR/transformer_prediction/NAFNet_BCI_512"

mkdir -p "$OUTPUT_DIR"

srun apptainer exec \
    --nv \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$CHECKPOINT_DIR":/checkpoint \
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
    bash /code/NAFNet/test_nafnet.sh \
    options/test/BCI/NAFNet-BCI-512-100k-test.yml
    "
