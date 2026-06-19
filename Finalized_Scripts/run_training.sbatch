#!/bin/bash
#SBATCH --job-name=flood_train
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=uvahydroinformatics            # Request 2 GPUs
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=200G               # Request 100 GB of RAM
#SBATCH --time=00:30:00         # Maximum time the job will run

# Navigate to your working directory
cd /home/dng3fe/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the python script
python Finalized_Scripts/model_training.py --input-file apparently-darling-gecko.parquet
