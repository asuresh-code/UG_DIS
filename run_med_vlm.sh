#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --mail-user=sureshanikesh@yahoo.co.uk

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate myModel

python --version

python3 -m pip install torch==2.6.0 torchvision

python3 -m pip install transformers==4.57.1

python3 -m pip install av --only-binary av

python3 -m pip install qwen_vl_utils

python3 -m pip install accelerate

python3 -m pip install numpy

python3 -m pip install pandas

python3 -m pip install matplotlib

pip show transformers

pip show torch

conda run -n myModel python medinvest.py