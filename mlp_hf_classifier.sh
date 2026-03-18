#!/bin/bash
#SBATCH --job-name=mlp_hf_classifier
#SBATCH --output=mlp_hf_classifier_%j.out
#SBATCH --error=mlp_hf_classifier_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --gres=gpu:4

# Load modules or activate environment if needed
# module load python/3.10
# source ~/myenv/bin/activate

cd $SLURM_SUBMIT_DIR

python src/mlp_hf_classifier.py
