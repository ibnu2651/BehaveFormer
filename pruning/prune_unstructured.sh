#!/bin/bash
#SBATCH --job-name=prune_unstructured
#SBATCH --output=prune_unstructured_out.slurmlog
#SBATCH --error=prune_unstructured_out.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1

srun python prune_unstructured.py
srun python -u finetune_unstructured.py
srun python -u test_unstructured_finetune.py