#!/bin/bash

BATCH --partition=titan
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=IdentiKey@colorado.edu
#SBATCH --output=/scratch/Shares/JLR_particle_csci6118/src/slurm/out/slurm_test_%j.out
#SBATCH --error=/scratch/Shares/JLR_particle_csci6118/src/slurm/out/slurm_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

## gather system information
hostname
date
uptime
df -h
pwd
## purge and load Fiji modules as needed
module purge
module load python/3.11.3
module list
## see which GPU device was assigned to your job
echo Assigned GPU via CUDA: $CUDA_VISIBLE_DEVICES
## hold for 30 seconds and finish
sleep 30
date
echo Finished
