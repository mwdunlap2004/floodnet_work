# FloodNet Work

Hydroinformatics project files for working with NYC FloodNet sensor data, including:
- API data extraction
- parquet joining and spatial workflows
- EDA and flood-duration analysis
- LSTM modeling notebooks
- Python and R plotting scripts

## Project Structure

- `Finalized_Scripts/`: primary project notebooks (cleaner/final workflow versions)
- `Test_Scripts/`: exploratory notebooks and standalone Python/R scripts
- `Images_or_plots/`: saved output figures

## 1. Prerequisites

Install:
- Python `3.10+` (recommended: `3.11`)
- `pip`
- Jupyter Notebook or JupyterLab
- Optional for R scripts: R `4.2+`

## 2. Clone And Enter The Project

```bash
git clone <(https://github.com/mwdunlap2004/floodnet_work.git)>
cd floodnet_work
```

## 3. Create A Python Virtual Environment

Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4. Install Python Dependencies

This repo currently does not include a pinned requirements file, so install the packages used by the notebooks/scripts:

```bash
pip install \
  jupyterlab notebook ipykernel \
  requests urllib3 numpy pandas pyarrow duckdb \
  matplotlib seaborn folium \
  geopandas shapely \
  scikit-learn statsmodels optuna \
  torch
```

Notes:
- `geopandas` may require system GIS libraries on some machines.
- `torch` install can vary by OS/GPU. If needed, install from the official PyTorch selector: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 5. (Optional) Install R Dependencies

If you plan to run `.R` scripts in `Test_Scripts/`:

```r
install.packages(c("tidyverse", "jsonlite", "leaflet", "htmltools", "lubridate"))
```

## 6. Run The Project

### Option A: Use Notebooks (recommended)

Start Jupyter:
```bash
jupyter lab
```
or
```bash
jupyter notebook
```

Then open notebooks from `Finalized_Scripts/`, for example:
- `Finalized_Scripts/floodnet_eda.ipynb`
- `Finalized_Scripts/joining_parquets.ipynb`
- `Finalized_Scripts/spatial_join.ipynb`
- `Finalized_Scripts/flood_duration.ipynb`
- `Finalized_Scripts/finalized_lstm_modeling.ipynb`
- `Finalized_Scripts/updated_method_storm_seperation.ipynb`

### Option B: Run Python scripts directly

1) Download FloodNet depth data to parquet files:
```bash
python Test_Scripts/parquet_floodnet_download.py
```
This creates `floodnet_parquet_data/` in your current working directory.

2) Generate a flood-depth map figure (requires input CSV):
```bash
python Test_Scripts/hw3_floodnet_graphs.py
```
Expected input file in working directory:
- `full_dataset.csv`

Expected output:
- `floodnet_max_depth_fig.png`

### Option C: Run R plotting scripts

From R or RStudio, run:
- `Test_Scripts/plotting_water_data.R`
- `Test_Scripts/new_coding_plots.R`

Expected CSVs for these scripts include:
- `got_rain.csv`
- `full_data.csv`
- `master_data.csv`

## 7. Typical Workflow (Step-by-Step)

1. Set up environment (Sections 1-5).
2. Pull FloodNet data via `Test_Scripts/parquet_floodnet_download.py` (or notebook equivalents).
3. Join/prepare data in `Finalized_Scripts/joining_parquets.ipynb`.
4. Perform spatial enrichment with `Finalized_Scripts/spatial_join.ipynb`.
5. Run analysis notebooks (`floodnet_eda.ipynb`, `flood_duration.ipynb`).
6. Run modeling notebooks (`finalized_lstm_modeling.ipynb`).
7. Save/inspect figures in `Images_or_plots/`.

## Troubleshooting

- `ModuleNotFoundError`: install missing package into the active `.venv`.
- Jupyter kernel mismatch: select the virtual environment kernel in notebook UI.
- GeoPandas install issues: install `geopandas` with conda/mamba if pip build fails.
- Slow API pulls: script includes retries; rerun safely (completed parquet files are checkpointed/skipped).
