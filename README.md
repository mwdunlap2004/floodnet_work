NYC FloodNet: Urban Flash-Flood Forecasting

Leveraging high-frequency ultrasonic sensor data and HRRR meteorological forcing to predict street-level flood depths.

University of Virginia | Hydroinformatics Final Project

Dataset Architecture

Filename

Size

Description

delineated_storms.parquet

353 MB

Primary modeling set (6h IETD).

floodnet_full_merged.parquet

2.63 GB

Master join (Sensor depths + Weather).

nyc_precip_master.parquet

104 MB

Cleaned hourly & max-intensity precip.

Data Engineering Highlights

Storm Delineation: Events isolated using a 6-hour dry-gap criterion (MIT standard) to ensure independent meteorological events.

Lead/Lag Buffering: Includes 2h antecedent moisture lead-up and 6h flood recession limb for full event capture.

Temporal Integrity: 70/15/15 splits performed strictly at the Storm ID level to prevent temporal data leakage.

Optimized I/O: Utilizes Apache Parquet with ZSTD for high-speed read/write and minimal disk footprint.

Modeling Pipeline

Models Tested

Baseline: Log-Ridge Regression

SOTA ANN: Res-ANN with LayerNorm

SOTA RNN: Attention-Bi-LSTM

Hardware Optimization

ANN Batch Size: 32,768

LSTM Batch Size: 2,048

Acceleration: Mixed Precision (torch.amp) Enabled

Reproducibility

Setup: Configure a Python 3.11+ environment with PyTorch and Optuna.

ETL: Run the provided DuckDB script to regenerate delineated storms.

Tune & Test: Execute the Optuna shootout and retrain the champion models.

Contact & License

Contact: Michael Dunlap 
Email: mwdunlap2004@icloud.com

References: NYC FloodNet/NYS Mesonet
