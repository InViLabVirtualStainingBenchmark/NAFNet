#!/bin/bash
#SBATCH --job-name=infer_nafnet_MIST_PR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/infer_nafnet_MIST_PR_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/infer_nafnet_MIST_PR_%j.err

set -euo pipefail

SHARED=/scratch/antwerpen/grp/ap_invilab_td_thesis
SQSH=${SHARED}/MIST_PR_neutral.sqsh
CONTAINER=${VSC_SCRATCH}/containers/basicsr_nvidia.sif
CODE_DIR=${VSC_DATA}/projects/code
OUTPUT_BASE=${SHARED}/benchmark_inference

WEIGHTS=${VSC_DATA}/projects/code/NAFNet/experiments/NAFNet-MIST-PR-512-100k-part2/models/net_g_latest.pth

echo "========================================"
echo " Job     : ${SLURM_JOB_ID}"
echo " Model   : NAFNet"
echo " Stain   : PR"
echo " Weights : ${WEIGHTS}"
echo "========================================"

srun apptainer exec \
    --nv \
    -B ${SQSH}:/data:image-src=/ \
    -B ${CODE_DIR}:/code \
    -B ${OUTPUT_BASE}:/output \

    --env VSC_DATA=${VSC_DATA} \
    ${CONTAINER} \
    python3 /code/benchmark_inference.py \
        --model   nafnet \
        --dataset MIST_PR \
        --weights /code/NAFNet/experiments/NAFNet-MIST-PR-512-100k-part2/models/net_g_latest.pth \
        --output_base /output

echo "Done — results at: ${OUTPUT_BASE}/nafnet_MIST_PR"
