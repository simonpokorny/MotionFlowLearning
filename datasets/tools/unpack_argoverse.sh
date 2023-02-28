#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=20G
#SBATCH --partition=amdfast

#SBATCH --error=logs/argo2_%a.out
#SBATCH --output=logs/argo2_%a.out

#ml TensorFlow/2.6.0-foss-2021a #
ml torchsparse

cd $HOME

python -u my_datasets/argoverse/argoverse2.py $SLURM_ARRAY_TASK_ID
