#!/bin/bash

#SBATCH --partition=titan
#SBATCH --job-name=training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roto9457@colorado.edu
#SBATCH --output=src/slurm/out/training_%j.out
#SBATCH --error=src/slurm/out/training_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=6
#SBATCH --mem=32G
##SBATCH --gres=gpu:1

'''
This script is for training the model using SLURM job submission.
It will use all available data in the ../Data/ directory for now.
The output is... we don/t know yet!
'''

echo Starting training job %j on $(hostname) at $(date)

conda activate particle

python src/train.py

date
deactivate

echo Training job completed

