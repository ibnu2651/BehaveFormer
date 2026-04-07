#!/bin/bash
#SBATCH --job-name=prune_structured
#SBATCH --output=quantise.slurmlog
#SBATCH --error=quantise.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --open-mode=append

# config="200_40_3"

# srun python prune_structured.py "$config"
# srun -u python finetune_structured.py "$config"
# srun -u python test_structured_finetune.py "$config"
# srun -u python export_to_onnx.py "$config"
srun -u python quantise_from_pruned.py