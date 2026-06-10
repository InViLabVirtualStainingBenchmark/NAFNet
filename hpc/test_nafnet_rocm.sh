#!/bin/bash
#SBATCH --job-name=test_nafnet_rocm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -A ap_invilab
#SBATCH -p arcturus_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/test_nafnet_rocm_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/test_nafnet_rocm_%j.err
set -euo pipefail
CONTAINER="$VSC_SCRATCH/containers/basicsr_rocm.sif"
CODE_DIR="$VSC_DATA/projects/code"
OUTPUT_DIR="$VSC_DATA/projects/outputs/test_nafnet_rocm"
mkdir -p "$OUTPUT_DIR"
srun apptainer exec \
    --rocm \
    -B "$CODE_DIR":/code \
    -B "$OUTPUT_DIR":/output \
    "$CONTAINER" \
    python3 -c "
import sys
sys.path.insert(0, '/code/NAFNet')
import torch
print('ROCm available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)) + ' GB')
import basicsr
print('basicsr imported OK')
"
