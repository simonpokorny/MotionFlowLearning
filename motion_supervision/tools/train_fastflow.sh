#!/bin/bash
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --mem=60G
#SBATCH --partition=amdgpulong
#SBATCH --gres=gpu:1

#SBATCH --error=logs/train_priors_%a.out
#SBATCH --output=logs/train_priors_%a.out


ml torchsparse

cd $HOME/motion_supervision

# why not working?!?
python train.py
# this argument will then specify the config file or exp$SLURM_ARRAY_TASK_ID
