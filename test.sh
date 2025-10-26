#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_out.slurmlog
#SBATCH --error=test_out.slurmlog
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1 -C cuda75

CUDA_VISIBLE_DEVICES=0 python test.py --dataname humi -c configs/humi/main/humi_scroll50down_imu100all_epoch500_enroll3_b128.yaml