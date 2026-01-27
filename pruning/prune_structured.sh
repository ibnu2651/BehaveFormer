#!/bin/bash
#SBATCH --job-name=prune_structured
#SBATCH --output=prune_structured_out.slurmlog
#SBATCH --error=prune_structured_out.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1

# srun python prune_structured.py
srun -u python finetune_structured.py
srun -u python test_structured_finetune.py
