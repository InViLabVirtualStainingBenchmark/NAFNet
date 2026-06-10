#!/bin/bash
#SBATCH --job-name=eval_nafnet_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab
#SBATCH -p broadwell

#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/eval_nafnet_BCI_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/eval_nafnet_BCI_%j.err

set -euo pipefail

CONTAINER="/scratch/antwerpen/grp/ap_invilab_td_thesis/evaluate_nvidia.sif"
DATA_SQSH="$VSC_SCRATCH/BCI_basicsr_split.sqsh"
GRP_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis"

module purge
module load calcua/2026.1

srun apptainer exec --nv \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$GRP_DIR":/grp \
    -B "$VSC_DATA/evaluate":/evaluate \
    "$CONTAINER" python3 /evaluate/evaluate.py \
        --pred   /grp/transformer_prediction/NAFNet_BCI \
        --gt     /data/test_IHC \
        --model_name   NAFNet \
        --dataset_name BCI \
        --split_name   test \
        --match_by     stem \
        --output       /grp/benchmark_results.csv \
        --device       cpu
