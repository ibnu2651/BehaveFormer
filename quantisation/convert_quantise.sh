#!/bin/bash
#SBATCH --job-name=convert_quantise
#SBATCH --output=quantise_static_out.slurmlog
#SBATCH --error=quantise_static_out.slurmlog
#SBATCH --time=01:00:00
#SBATCH --mem=16G

srun python quantise_static.py