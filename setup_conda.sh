#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate myModel
python3 -m pip install torch torchvision

python3 -m pip install transformers

python3 -m pip install qwen_vl_utils