#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=preproc_out.slurmlog
#SBATCH --error=preproc_out.slurmlog
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1

srun python -m preprocessing.preprocess_humidb --config "configs/humi/preprocess/humi_scroll50downup_imu100.yaml" --ncpus 8
