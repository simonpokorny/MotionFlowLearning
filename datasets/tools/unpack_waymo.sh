#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=30G
#SBATCH --partition=amdfast

#SBATCH --error=logs/waymo_%a.out
#SBATCH --output=logs/waymo_%a.out

#ml TensorFlow/2.6.0-foss-2021a # for unpacking one_time_waymo
ml torchsparse

cd $HOME

python -u my_datasets/waymo/waymo.py $SLURM_ARRAY_TASK_ID
