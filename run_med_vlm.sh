#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 medvlm.py