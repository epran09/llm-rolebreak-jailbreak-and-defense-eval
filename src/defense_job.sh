#!/bin/bash
#SBATCH --job-name=Defense_Job
#SBATCH --partition=l4
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
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


python defense.py


echo "Job completed at $(date)"
