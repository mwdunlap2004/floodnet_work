#!/bin/bash
#SBATCH --job-name=flood_hpo
#SBATCH --output=hpo_%j.out
#SBATCH --error=hpo_%j.err
#SBATCH --partition=interactive
#SBATCH --account=ds6050_sp26 
#SBATCH --gres=gpu:1        
#SBATCH --cpus-per-task=8      
#SBATCH --mem=100G               
#SBATCH --time=1:00:00          

# Navigate to your working directory
cd /home/upw4ys/Documents/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the hyperparameter optimization script
python Finalized_Scripts/hpo_search.py
