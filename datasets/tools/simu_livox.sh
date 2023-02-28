#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --mem=60G
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out

ml torchsparse
cd $HOME

python -u data_utils/livox/simu_livox.py
