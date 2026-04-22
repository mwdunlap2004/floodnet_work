# FloodNet Work

Hydroinformatics workflows for NYC FloodNet sensor data, including:
- API extraction and parquet checkpointing
- dataset joining and spatial weather enrichment
- storm delineation and event-level EDA
- rain-influenced gage selection for modeling
- flood-duration analysis and model training

## Project Structure

- `Finalized_Scripts/`: primary notebooks and Python scripts for the core pipeline
- `Test_Scripts/`: exploratory notebooks and standalone experiments
- `Data_Files/`: parquet datasets and training databases
- `Images_or_plots/`: exported figures
- `results/`: run logs and presentation outputs

## 1. Prerequisites

- Python `3.10+` (recommended: `3.11`)
- `pip`
- Jupyter Notebook or JupyterLab
- Optional for R scripts: R `4.2+`

## 2. Quickstart (5 Minutes)

```bash
git clone https://github.com/mwdunlap2004/floodnet_work.git
cd floodnet_work
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Windows (PowerShell):

```powershell
git clone https://github.com/mwdunlap2004/floodnet_work.git
cd floodnet_work
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Notes:
- `geopandas` may require system GIS libraries on some machines.
- `torch` install can vary by OS/GPU. If needed, use the official selector: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 3. Data Inputs (Expected Files)

Most finalized notebooks/scripts expect files under `Data_Files/`:

Primary shared data location:
- Google Drive folder: [FloodNet project data](https://drive.google.com/drive/folders/1N-w9N6aYujfRRqcTfod1GltTaggaVSnx?usp=share_link)
- Download/sync these files into local `Data_Files/` before running the notebooks.

| File | Produced By | Used By |
|---|---|---|
| `floodnet_parquet_data/*.parquet` | `Finalized_Scripts/parquet_floodnet_download.py` | `Finalized_Scripts/joining_parquets.ipynb` |
| `floodnet_full_dataset_merged.parquet` | `Finalized_Scripts/joining_parquets.ipynb` | intermediate checks |
| `floodnet_floods_only.parquet` | `Finalized_Scripts/joining_parquets.ipynb` | `Finalized_Scripts/flood_duration.ipynb` |
| `floodnet_full_dataset_merged_with_weather.parquet` | `Finalized_Scripts/spatial_join.ipynb` | `Finalized_Scripts/select_precip_influenced_gages.py` |
| `rain_influenced_sites_raw.parquet` | `Finalized_Scripts/select_precip_influenced_gages.py` | `Finalized_Scripts/delineate_filtered_storms.py` |
| `rain_influenced_gages.parquet` | `Finalized_Scripts/delineate_filtered_storms.py` | `Finalized_Scripts/rain_influenced_EDA.ipynb`, `Finalized_Scripts/hpo_search.py`, `Finalized_Scripts/model_training.py` |
| `delineated_storms.parquet` | `Finalized_Scripts/updated_method_storm_seperation.ipynb` | `Finalized_Scripts/floodnet_eda.ipynb` |
| `floodnet_hpo_newfilter.db` | `Finalized_Scripts/hpo_search.py` | `Finalized_Scripts/model_training.py` |

## 4. Notebook Run Order

Use this order for a clean, reproducible workflow:

1. `Finalized_Scripts/parquet_floodnet_download.py`
2. `Finalized_Scripts/joining_parquets.ipynb`
3. `Finalized_Scripts/spatial_join.ipynb`
4. `Finalized_Scripts/select_precip_influenced_gages.py` (top-10 gages, 30-day tidal gate; creates `Data_Files/rain_influenced_sites_raw.parquet`)
5. `Finalized_Scripts/delineate_filtered_storms.py` (creates final training input `Data_Files/rain_influenced_gages.parquet`)
6. `Finalized_Scripts/rain_influenced_EDA.ipynb` (rain-coupling diagnostics for selected/delineated gages)
7. `Finalized_Scripts/hpo_search.py` (default input: `rain_influenced_gages.parquet`)
8. `Finalized_Scripts/model_training.py` (default input: `rain_influenced_gages.parquet`)
9. Optional full-network analysis notebooks: `Finalized_Scripts/updated_method_storm_seperation.ipynb`, `Finalized_Scripts/floodnet_eda.ipynb`, `Finalized_Scripts/flood_duration.ipynb`, `Finalized_Scripts/finalized_lstm_modeling.ipynb`

## 5. SLURM Submission

- Unified job script:
  - `Finalized_Scripts/run_rain_influenced.sbatch`
- Examples:

```bash
sbatch Finalized_Scripts/run_rain_influenced.sbatch
sbatch --export=STEP=hpo Finalized_Scripts/run_rain_influenced.sbatch
sbatch --export=STEP=train Finalized_Scripts/run_rain_influenced.sbatch
sbatch --export=STEP=all Finalized_Scripts/run_rain_influenced.sbatch
```

- Legacy scripts are also aligned to the new input file:
  - `Finalized_Scripts/run_hpo.slurm`
  - `Finalized_Scripts/run_training.sbatch`
## 6. Outputs

- `Images_or_plots/`: static figures (for example model comparison and duration plots)
- `results/`: model run artifacts/logs
- `results/presentation_figures/`: exported presentation figures from `floodnet_eda.ipynb`
- `checkpoints/`: trained model checkpoints and scalers from training scripts

## 7. Optional R Dependencies

If you plan to run `.R` scripts in `Test_Scripts/`:

```r
install.packages(c("tidyverse", "jsonlite", "leaflet", "htmltools", "lubridate"))
```

## Troubleshooting

- `ModuleNotFoundError`: install missing packages into the active `.venv`.
- Jupyter kernel mismatch: select the `.venv` kernel in notebook UI.
- GeoPandas install issues: try conda/mamba for geospatial dependencies.
- Slow API pulls: the downloader includes retries and checkpointed parquet outputs.
