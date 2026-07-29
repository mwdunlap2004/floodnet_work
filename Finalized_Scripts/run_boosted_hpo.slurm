#!/bin/bash
#SBATCH --job-name=flood_hpo
#SBATCH --output=hpo_%j.out
#SBATCH --error=hpo_%j.err
#SBATCH --partition=gpu
#SBATCH --exclude=udc-an33-37,udc-an33-38
#SBATCH --account=uvahydroinformatics
#SBATCH --gres=gpu:2   
#SBATCH --cpus-per-task=8      
#SBATCH --mem=100G               
#SBATCH --time=3-00:00:00          

# Navigate to your working directory
cd /home/dng3fe/floodnet_work

# Activate your virtual environment
source .venv/bin/activate 

# Run the hyperparameter optimization script
python Finalized_Scripts/boosted_hpo_search.py
