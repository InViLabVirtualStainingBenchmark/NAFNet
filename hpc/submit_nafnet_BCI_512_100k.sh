#!/bin/bash
# Submits NAFNet BCI 512 100k training in two chained parts
# Part 2 starts automatically after Part 1 completes successfully

JOB1=$(sbatch --parsable $VSC_DATA/projects/jobs/train_nafnet_BCI_512_100k_part1.sh)
echo "Submitted Part 1: job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
  $VSC_DATA/projects/jobs/train_nafnet_BCI_512_100k_part2.sh)
echo "Submitted Part 2: job $JOB2 (starts after $JOB1 completes)"

echo ""
echo "Monitor with: squeue -u vsc21216"
