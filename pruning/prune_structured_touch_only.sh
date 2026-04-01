#!/bin/bash
#SBATCH --job-name=prune_structured
#SBATCH --output=prune_structured_touch_only_200_40_5.slurmlog
#SBATCH --error=prune_structured_touch_only_200_40_5.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
## SBATCH --open-mode=append

config="200_40_5"

srun python prune_structured_touch_only.py "$config"
srun -u python finetune_structured_touch_only.py "$config"
srun -u python test_structured_finetune_touch_only.py "$config"
# srun -u python export_to_onnx.py "$config"
