# FloodNet Work

Hydroinformatics workflows for NYC FloodNet sensor data, including:
- API extraction and parquet checkpointing
- dataset joining and spatial weather enrichment
- storm delineation and event-level EDA
- rain-influenced gage selection for modeling
- single-gage focused modeling on `apparently-darling-gecko`
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
| `apparently-darling-gecko.parquet` | `Finalized_Scripts/apparently_darling_gecko.py` | `Finalized_Scripts/hpo_search.py`, `Finalized_Scripts/model_training.py` |
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
6. `Finalized_Scripts/apparently_darling_gecko.py` (isolates the single modeling gage and creates `Data_Files/apparently-darling-gecko.parquet`)
7. `Finalized_Scripts/rain_influenced_EDA.ipynb` (rain-coupling diagnostics for selected/delineated gages)
8. `Finalized_Scripts/hpo_search.py` (default input: `apparently-darling-gecko.parquet`)
9. `Finalized_Scripts/model_training.py` (default input: `apparently-darling-gecko.parquet`)
10. Optional full-network analysis notebooks: `Finalized_Scripts/updated_method_storm_seperation.ipynb`, `Finalized_Scripts/floodnet_eda.ipynb`, `Finalized_Scripts/flood_duration.ipynb`, `Finalized_Scripts/finalized_lstm_modeling.ipynb`

## 5. Modeling Scope: Why One Gage

For final model development, the workflow uses one high-reliability FloodNet deployment: `apparently-darling-gecko` (written as `apparently_darling_gecko` in text and `apparently-darling-gecko` in file/script IDs).  
`Finalized_Scripts/apparently_darling_gecko.py` filters storm records to that deployment and writes `Data_Files/apparently-darling-gecko.parquet`, which is the default training input used by both `hpo_search.py` and `model_training.py`.

## 6. Verified Metrics (Accuracy Check)

The current checked metrics are from `results/run_log.json` with timestamp `2026-05-03T12:00:36.337321`.

Test-set skill scores:

| Model | KGE | NSE | RMSE (in) | PBIAS (%) | PeakNSE |
|---|---:|---:|---:|---:|---:|
| Log-Ridge | -0.0246 | 0.1502 | 1.3014 | 55.83 | -0.4895 |
| Res-ANN | 0.0772 | 0.1806 | 1.2779 | 30.07 | -0.4044 |
| Attn-LSTM | -0.0477 | 0.0843 | 1.3695 | 45.18 | -0.5499 |

Train-set skill scores:

| Model | KGE | NSE | RMSE (in) | PBIAS (%) | PeakNSE |
|---|---:|---:|---:|---:|---:|
| Log-Ridge | 0.4419 | 0.1626 | 0.4988 | 3.54 | -1.1352 |
| Res-ANN | 0.5348 | 0.2275 | 0.4790 | 1.98 | -0.5224 |
| Attn-LSTM | 0.6899 | 0.5888 | 0.3548 | -2.62 | -0.1798 |

## 7. SLURM Submission

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
## 8. Outputs

- `Images_or_plots/`: static figures (for example model comparison and duration plots)
- `results/`: model run artifacts/logs
- `results/presentation_figures/`: exported presentation figures from `floodnet_eda.ipynb`
- `checkpoints/`: trained model checkpoints and scalers from training scripts

## 9. Optional R Dependencies

If you plan to run `.R` scripts in `Test_Scripts/`:

```r
install.packages(c("tidyverse", "jsonlite", "leaflet", "htmltools", "lubridate"))
```

## Troubleshooting

- `ModuleNotFoundError`: install missing packages into the active `.venv`.
- Jupyter kernel mismatch: select the `.venv` kernel in notebook UI.
- GeoPandas install issues: try conda/mamba for geospatial dependencies.
- Slow API pulls: the downloader includes retries and checkpointed parquet outputs.
