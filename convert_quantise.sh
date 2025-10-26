#!/bin/bash
#SBATCH --job-name=convert_quantise
#SBATCH --output=convert_quantise_out.slurmlog
#SBATCH --error=convert_quantise_out.slurmlog
#SBATCH --time=01:00:00
#SBATCH --mem=16G

srun python convert_quantise.py