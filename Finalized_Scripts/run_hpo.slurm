#!/bin/bash
#SBATCH --job-name=flood_hpo
#SBATCH --output=hpo_%j.out
#SBATCH --error=hpo_%j.err
#SBATCH --partition=gpu        
#SBATCH --gres=gpu:2             
#SBATCH --cpus-per-task=4        
#SBATCH --mem=100G               
#SBATCH --time=23:00:00          

# Navigate to your working directory
cd /home/upw4ys/Documents/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the hyperparameter optimization script
python Finalized_Scripts/hpo_search.py --input-file rain_influenced_gages.parquet
