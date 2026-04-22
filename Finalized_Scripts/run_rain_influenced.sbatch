#!/bin/bash
#SBATCH --job-name=flood_rain
#SBATCH --output=flood_rain_%j.out
#SBATCH --error=flood_rain_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00

set -euo pipefail

# Usage:
#   sbatch Finalized_Scripts/run_rain_influenced.sbatch                  # defaults to "both"
#   sbatch --export=STEP=hpo Finalized_Scripts/run_rain_influenced.sbatch
#   sbatch --export=STEP=train Finalized_Scripts/run_rain_influenced.sbatch
#   sbatch --export=STEP=both Finalized_Scripts/run_rain_influenced.sbatch
#   sbatch --export=STEP=all  Finalized_Scripts/run_rain_influenced.sbatch
#
# Optional overrides:
#   --export=PROJECT_ROOT=/home/upw4ys/Documents/floodnet_work
#   --export=DATA_FILE=rain_influenced_gages.parquet

PROJECT_ROOT="${PROJECT_ROOT:-/home/upw4ys/Documents/floodnet_work}"
STEP="${STEP:-both}"
DATA_FILE="${DATA_FILE:-rain_influenced_gages.parquet}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "STEP=$STEP"
echo "DATA_FILE=$DATA_FILE"

case "$STEP" in
  hpo)
    python Finalized_Scripts/hpo_search.py --input-file "$DATA_FILE"
    ;;
  train)
    python Finalized_Scripts/model_training.py --input-file "$DATA_FILE"
    ;;
  both)
    python Finalized_Scripts/hpo_search.py --input-file "$DATA_FILE"
    python Finalized_Scripts/model_training.py --input-file "$DATA_FILE"
    ;;
  all)
    python Finalized_Scripts/select_precip_influenced_gages.py --output "Data_Files/$DATA_FILE"
    python Finalized_Scripts/hpo_search.py --input-file "$DATA_FILE"
    python Finalized_Scripts/model_training.py --input-file "$DATA_FILE"
    ;;
  *)
    echo "Invalid STEP: $STEP"
    echo "Valid values: hpo, train, both, all"
    exit 1
    ;;
esac
