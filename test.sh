#!/bin/bash

#SBATCH -A berzelius-2026-6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nithesh.chandher.karthikeyan@liu.se
#SBATCH --gpus 2
#SBATCH -t 3-00:00:00

# Define paths
export ZIP_DATA=/proj/dcdl/users/gabei62/rep-ldm-test/data/cifar10_subset.zip
export SCRATCH_DIR=/scratch/local/rep-ldm/data

# Create scratch directories
mkdir -p $SCRATCH_DIR 

# Extract ZIP files
echo "Extracting dataset ZIP..."
unzip -q $ZIP_DATA -d $SCRATCH_DIR

# Load Anaconda and activate environment
module load buildtool-easybuild/4.3.3-nscf4a947
module load Anaconda3/2021.05-nsc1
conda init bash
source ~/.bashrc
conda activate di

# Navigate to DINO-LDM directory
cd /proj/dcdl/users/gabei62/rep-ldm-test

# Start training
accelerate launch rep-dm.py --config="configs/cifar10/cifar10-subset-dino.yaml"