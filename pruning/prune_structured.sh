#!/bin/bash
#SBATCH --job-name=prune_structured
#SBATCH --output=prune_structured_iterative_rd1_out.slurmlog
#SBATCH --error=prune_structured_iterative_rd1_out.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --gres=gpu:a100-40:1
#SBATCH -p gpu



srun python prune_structured.py
srun -u python finetune_structured.py
srun -u python test_structured_finetune.py
# srun -u python export_to_onnx.py
