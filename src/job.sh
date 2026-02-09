#!/bin/bash
#SBATCH --job-name=Judge_job
#SBATCH --partition=l4
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --mail-type=FAIL

echo "Starting job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"


accelerate launch --num_processes=2 --mixed_precision=fp16  --main_process_port 0  llm_judge.py


echo "Job completed at $(date)"
