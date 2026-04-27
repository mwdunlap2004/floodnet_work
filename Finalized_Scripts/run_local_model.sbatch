#!/bin/bash
#SBATCH --job-name=flood_local
#SBATCH --output=local_train_%j.out
#SBATCH --error=local_train_%j.err
#SBATCH --partition=gpu
#SBATCH --account=ds6050_sp26 
#SBATCH --gres=gpu:2             
#SBATCH --cpus-per-task=4        
#SBATCH --mem=100G               
#SBATCH --time=12:00:00          

# Navigate to your working directory
cd /home/upw4ys/Documents/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the local model training script
# Note: Ensure you saved the script from the previous step as local_training.py inside Finalized_Scripts/
python Finalized_Scripts/local_training.py --input-file rain_influenced_gages.parquet