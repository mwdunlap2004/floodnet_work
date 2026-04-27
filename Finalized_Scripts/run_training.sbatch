#!/bin/bash
#SBATCH --job-name=flood_train
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --account=ds6050_sp26             # Request 2 GPUs
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=200G               # Request 100 GB of RAM
#SBATCH --time=2:00:00         # Maximum time the job will run

# Navigate to your working directory
cd /home/upw4ys/Documents/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the python script
python Finalized_Scripts/model_training.py --input-file rain_influenced_gages.parquet
