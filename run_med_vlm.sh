#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate myModel
srun python3 medvlm.py