# %%
# %%
# %%─────────────────────────────────────────────────────────────────────────
# 02_final_training.py
# FloodNet — Final Model Training Script
# Loads best hyperparameters from Optuna DB (produced by 01_hpo_search.py),
# retrains all three models on Train+Val, evaluates on held-out Test, and
# saves checkpoints, scalers, metrics, and figures to disk.
# ─────────────────────────────────────────────────────────────────────────────
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 1 │ Imports & Hardware Setup
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import json
import gc
import warnings
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import optuna
from optuna.samplers import TPESampler
 
warnings.filterwarnings('ignore')

# %%
# ── Multi-GPU Setup ──────────────────────────────────────────────────────────
N_GPUS  = min(torch.cuda.device_count(), 2)
PRIMARY = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler_amp = GradScaler(device=PRIMARY.type) if torch.cuda.is_available() else None
 
print(f"🚀 Using {N_GPUS} GPU(s) | Primary: {PRIMARY}")
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"   [{i}] {p.name}  ({p.total_memory / 1e9:.1f} GB)")
 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 1b │ VRAM Safeguard Utilities
# ─────────────────────────────────────────────────────────────────────────────
 
def vram_free_gb(device: int = 0):
    """Returns (free_gb, total_gb) for a given CUDA device."""
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return free / 1e9, total / 1e9
 
 
def require_vram(gb_needed: float, label: str = ""):
    """
    Checks that at least `gb_needed` GB is free on the primary GPU.
    Attempts a cache flush first; raises MemoryError if still insufficient.
    """
    free, total = vram_free_gb()
    print(f"   VRAM check [{label}]: {free:.1f} GB free / {total:.1f} GB total")
    if free < gb_needed:
        torch.cuda.empty_cache()
        gc.collect()
        free, _ = vram_free_gb()
        if free < gb_needed:
            raise MemoryError(
                f"[{label}] Need ≥{gb_needed:.1f} GB free, only {free:.1f} GB available. "
                "Reduce hidden_size, n_layers, or batch size in best_params."
            )
 
 
def safe_batch_size(model: nn.Module, sample_input: torch.Tensor,
                    starting_batch: int = 32768, min_batch: int = 512) -> int:
    """
    Binary-searches the largest batch size that fits on the primary GPU
    without an OutOfMemoryError.  Starts from `starting_batch` and halves
    until either a size succeeds or `min_batch` is reached.
    """
    batch = starting_batch
    model.eval()
    while batch >= min_batch:
        try:
            torch.cuda.empty_cache()
            dummy = sample_input[:batch].to(PRIMARY)
            with torch.no_grad(), autocast(device_type='cuda'):
                _ = model(dummy)
            del dummy
            torch.cuda.empty_cache()
            print(f"   ✅ Safe batch size: {batch:,}")
            return batch
        except torch.cuda.OutOfMemoryError:
            batch //= 2
            print(f"   ⚠️  OOM — reducing batch to {batch:,}")
    raise MemoryError(
        f"Even batch size {min_batch} causes OOM. "
        "Consider reducing hidden_size or n_layers."
    )
 
 
def train_step(model: nn.Module, opt: optim.Optimizer,
               amp_scaler: GradScaler, bx: torch.Tensor,
               by: torch.Tensor, loss_fn: nn.Module,
               clip_grad: float | None = None) -> float | None:
    """
    Executes one training step with AMP and optional gradient clipping.
    Catches OOM, clears cache, and returns None so the caller can skip
    the batch rather than crash.
    """
    try:
        opt.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            loss = loss_fn(model(bx), by)
        amp_scaler.scale(loss).backward()
        if clip_grad is not None:
            amp_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        amp_scaler.step(opt)
        amp_scaler.update()
        return loss.item()
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print("   ⚠️  OOM on batch — skipped and cache cleared.")
        return None


class WeightedDepthLoss(nn.Module):
    """
    Depth-weighted regression loss:
      base_loss(yhat, y) * (1 + lambda_weight * y_true_depth)
    The weighting uses inverse-scaled target depth, so wet/peak timesteps
    contribute more strongly than dry/background timesteps.
    """
    def __init__(self, base: str = "huber", lambda_weight: float = 2.0):
        super().__init__()
        self.base = str(base).lower()
        self.lambda_weight = float(lambda_weight)
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true_scaled: torch.Tensor) -> torch.Tensor:
        y_true_depth = descale(y_true_scaled)
        weight = 1.0 + self.lambda_weight * torch.clamp(y_true_depth, min=0.0)
        if self.base == "mse":
            base_loss = (y_pred - y_true_scaled) ** 2
        else:
            base_loss = self.huber(y_pred, y_true_scaled)
        return (base_loss * weight).mean()


def build_loss_fn(params: dict, model_name: str = "model") -> nn.Module:
    """Builds weighted/unweighted MSE or Huber loss from trial params."""
    loss_name = str(params.get("loss_fn", "huber")).lower()
    use_weighted = bool(params.get("use_weighted_loss", True))
    lambda_w = float(params.get("loss_lambda", 2.0))

    if use_weighted:
        print(f"   🎯 {model_name}: weighted {loss_name} loss (lambda={lambda_w:.3f})")
        return WeightedDepthLoss(base=loss_name, lambda_weight=lambda_w)

    print(f"   🎯 {model_name}: unweighted {loss_name} loss")
    return nn.MSELoss() if loss_name == "mse" else nn.HuberLoss()


def resolve_batch_size(model: nn.Module,
                       sample_input: torch.Tensor,
                       params: dict,
                       default_start: int,
                       default_min: int,
                       model_name: str = "model") -> int:
    """
    Allows explicit batch-size control from Optuna params while preserving
    OOM-safe probing fallback.
    """
    start_bs = int(params.get("batch_size", default_start))
    min_bs = int(params.get("min_batch_size", default_min))
    if start_bs < min_bs:
        start_bs = min_bs
    print(f"   📦 {model_name}: batch probe start={start_bs:,}, min={min_bs:,}")
    return safe_batch_size(model, sample_input, starting_batch=start_bs, min_batch=min_bs)

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 2 │ Paths, Config, and Directory Setup
# ─────────────────────────────────────────────────────────────────────────────
try:
    current_location = Path(__file__).resolve().parent
except NameError:
    current_location = Path.cwd().resolve()
 
if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"]:
    PROJECT_ROOT = current_location.parent
else:
    PROJECT_ROOT = current_location
 
DATA_DIR       = PROJECT_ROOT / "Data_Files"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"
FIGURES_DIR    = PROJECT_ROOT / "Images_or_plots"
 
for d in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)
 
# Configuration and Feature Selection
FEATURES = [
    'precip_1hr [inch]', 
    'precip_max_intensity [inch/hour]', 
    'temp_2m [degF]', 
    # 'soil_moisture_05cm [m^3/m^3]',  <-- REMOVE THIS
    'elevation [feet]'
]
TARGET   = 'depth_inches'
TV_SPLIT = (0.70, 0.15, 0.15)

# CLI flags (parse known args to avoid notebook-injected flags)
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--input-file",
    default="rain_influenced_gages.parquet",
    help="Parquet filename under Data_Files/ or absolute path.",
)
args, _unknown = parser.parse_known_args()
 
HPO_DB_NAME = "floodnet_hpo_newfilter.db"
DB = f"sqlite:///{PROJECT_ROOT}/Data_Files/{HPO_DB_NAME}"
db_path = PROJECT_ROOT / "Data_Files" / HPO_DB_NAME

if not db_path.exists():
    raise FileNotFoundError(
        f"Optuna database not found at {db_path}. "
        "Run hpo_search.py first."
    )

 
study_lr   = optuna.load_study(study_name="log_ridge", storage=DB)
study_ann  = optuna.load_study(study_name="res_ann",   storage=DB)
study_lstm = optuna.load_study(study_name="attn_lstm", storage=DB)
 
bp_lr   = study_lr.best_params
bp_ann  = study_ann.best_params
bp_lstm = study_lstm.best_params
 
# ── Unified best_params dict (used in run_log and checkpoint saves) ───────────
# FIX: was referenced in Blocks 9/10/11 but never defined, causing NameError.
best_params = {
    "log_ridge": bp_lr,
    "res_ann":   bp_ann,
    "attn_lstm": bp_lstm,
}
 
print(f"✅ Loaded studies from {db_path}")
print(f"   Log-Ridge : best NSE {study_lr.best_value:.4f}  | {bp_lr}")
print(f"   Res-ANN   : best NSE {study_ann.best_value:.4f}  | {bp_ann}")
print(f"   Attn-LSTM : best NSE {study_lstm.best_value:.4f}  | {bp_lstm}")

# Fit-boosted search/retrain policy
TOP_N_CANDIDATES = 3
VAL_NSE_TOL = 0.03
VAL_KGE_TOL = 0.05


def _trial_metric(trial, key: str, fallback: float) -> float:
    v = trial.user_attrs.get(key, fallback)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = fallback
    if not np.isfinite(v):
        return fallback
    return v


def select_candidate_trials(study, top_n=3, val_nse_tol=0.03, val_kge_tol=0.05):
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    if not completed:
        return [{
            "trial_number": None,
            "params": study.best_params,
            "train_nse": float("-inf"),
            "val_nse": float(study.best_value),
            "val_kge": float("-inf"),
        }]

    best_val_nse = max(_trial_metric(t, "val_nse", float(t.value)) for t in completed)
    best_val_kge = max(_trial_metric(t, "val_kge", float("-inf")) for t in completed)
    floor_nse = best_val_nse - val_nse_tol
    floor_kge = best_val_kge - val_kge_tol if np.isfinite(best_val_kge) else float("-inf")

    filtered = []
    for t in completed:
        val_nse = _trial_metric(t, "val_nse", float(t.value))
        val_kge = _trial_metric(t, "val_kge", float("-inf"))
        train_nse = _trial_metric(t, "train_nse", float("-inf"))
        if val_nse >= floor_nse and val_kge >= floor_kge:
            filtered.append({
                "trial_number": t.number,
                "params": t.params,
                "train_nse": train_nse,
                "val_nse": val_nse,
                "val_kge": val_kge,
            })

    if not filtered:
        ranked = sorted(completed, key=lambda t: float(t.value), reverse=True)[:top_n]
        return [{
            "trial_number": t.number,
            "params": t.params,
            "train_nse": _trial_metric(t, "train_nse", float("-inf")),
            "val_nse": _trial_metric(t, "val_nse", float(t.value)),
            "val_kge": _trial_metric(t, "val_kge", float("-inf")),
        } for t in ranked]

    filtered.sort(key=lambda r: (r["train_nse"], r["val_nse"]), reverse=True)
    return filtered[:top_n]


lr_candidates = select_candidate_trials(
    study_lr, top_n=TOP_N_CANDIDATES, val_nse_tol=VAL_NSE_TOL, val_kge_tol=VAL_KGE_TOL
)
ann_candidates = select_candidate_trials(
    study_ann, top_n=TOP_N_CANDIDATES, val_nse_tol=VAL_NSE_TOL, val_kge_tol=VAL_KGE_TOL
)
lstm_candidates = select_candidate_trials(
    study_lstm, top_n=TOP_N_CANDIDATES, val_nse_tol=VAL_NSE_TOL, val_kge_tol=VAL_KGE_TOL
)

bp_lr = lr_candidates[0]["params"]
bp_ann = ann_candidates[0]["params"]
bp_lstm = lstm_candidates[0]["params"]
best_params = {
    "log_ridge": bp_lr,
    "res_ann": bp_ann,
    "attn_lstm": bp_lstm,
}

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 3 │ Data Loading and Storm-Aware Split
# ─────────────────────────────────────────────────────────────────────────────
input_file = Path(args.input_file)
file_path = input_file if input_file.is_absolute() else (DATA_DIR / input_file)
if not file_path.exists():
    raise FileNotFoundError(f"Data not found at: {file_path}")
 
df = pd.read_parquet(file_path)
print(f"✅ Loaded data from: {file_path}")
print(f"✅ Loaded data: {len(df):,} rows")
 
# ── Resolve storm identifier column ──────────────────────────────────────────
STORM_COL = None
for candidate in ['global_storm_id', 'storm_id', 'event_id', 'storm', 'event']:
    if candidate in df.columns:
        STORM_COL = candidate
        break
 
if STORM_COL is None:
    if isinstance(df.index, pd.DatetimeIndex):
        gap_seconds = df.index.to_series().diff().dt.total_seconds().fillna(0)
        df['_storm_id'] = (gap_seconds > 6 * 3600).cumsum()
    else:
        CHUNK = 500
        df['_storm_id'] = np.arange(len(df)) // CHUNK
    STORM_COL = '_storm_id'
    print(f"⚠️  No storm ID found. Inferred {df[STORM_COL].nunique()} events.")
else:
    print(f"✅ Storm column '{STORM_COL}' — {df[STORM_COL].nunique()} events.")

# %%
# ── Drop incomplete rows, cast to float32 ────────────────────────────────────
# Keep optional metadata columns for downstream diagnostics/exports.
META_CANDIDATES = ["deployment_id", "time", "timestamp", "datetime", "global_storm_id", "storm_start", "storm_end"]
META_COLS = [c for c in META_CANDIDATES if c in df.columns]

# Use a set to avoid duplicate column names if STORM_COL is in META_COLS
all_cols = list(dict.fromkeys(FEATURES + [TARGET, STORM_COL] + META_COLS))
df_clean = df[all_cols].dropna(
    subset=FEATURES + [TARGET, STORM_COL]
).copy()
df_clean[FEATURES + [TARGET]] = df_clean[FEATURES + [TARGET]].astype('float32')

# ── Chronological non-leaky split ──────────────────────────────────────────
# We group overlapping storms across sensors into 'global_events' to prevent
# data leakage (same storm in both train and test).
if 'global_storm_id' in df_clean.columns and 'storm_start' in df_clean.columns:
    storm_meta = df_clean[['global_storm_id', 'storm_start', 'storm_end']].drop_duplicates()
    storm_meta = storm_meta.sort_values('storm_start')
    
    event_ids = []
    if not storm_meta.empty:
        curr_id = 0
        curr_end = storm_meta.iloc[0]['storm_end']
        for _, row in storm_meta.iterrows():
            if row['storm_start'] < curr_end:
                event_ids.append(curr_id)
                curr_end = max(curr_end, row['storm_end'])
            else:
                curr_id += 1
                event_ids.append(curr_id)
                curr_end = row['storm_end']
        storm_meta['global_event_id'] = event_ids
        
        # Map back to df_clean
        df_clean = df_clean.merge(storm_meta[['global_storm_id', 'global_event_id']], on='global_storm_id', how='left')
        SPLIT_COL = 'global_event_id'
    else:
        SPLIT_COL = STORM_COL
else:
    SPLIT_COL = STORM_COL

split_ids = df_clean[SPLIT_COL].unique()
n_events = len(split_ids)

# Robust rebalancing (from hpo_search.py)
split_props = np.array(TV_SPLIT, dtype=float)
split_props = split_props / split_props.sum()
split_counts = np.floor(split_props * n_events).astype(int)

for i, p in enumerate(split_props):
    if p > 0 and split_counts[i] == 0:
        split_counts[i] = 1

while split_counts.sum() > n_events:
    idx = int(np.argmax(split_counts))
    if split_counts[idx] > 1:
        split_counts[idx] -= 1
    else:
        break
while split_counts.sum() < n_events:
    idx = int(np.argmax(split_props))
    split_counts[idx] += 1

n_tr, n_va, n_te = split_counts.tolist()
train_events = split_ids[:n_tr]
val_events   = split_ids[n_tr : n_tr + n_va]
test_events  = split_ids[n_tr + n_va :]
 
train_df = df_clean[df_clean[SPLIT_COL].isin(train_events)].copy()
val_df   = df_clean[df_clean[SPLIT_COL].isin(val_events)].copy()
test_df  = df_clean[df_clean[SPLIT_COL].isin(test_events)].copy()

# Storm IDs per split (used for run logging and plotting selection)
train_storms = train_df[STORM_COL].dropna().unique().tolist()
val_storms   = val_df[STORM_COL].dropna().unique().tolist()
test_storms  = test_df[STORM_COL].dropna().unique().tolist()
 
print(f"\n📊 Chronological Event-Based Split (Non-Leaky):")
print(f"   Train : {len(train_df):>8,} rows  ({len(train_events):>4} events)")
print(f"   Val   : {len(val_df):>8,} rows  ({len(val_events):>4} events)")
print(f"   Test  : {len(test_df):>8,} rows  ({len(test_events):>4} events)")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 4 │ Train+Val Scaling and GPU Tensor Push
# ─────────────────────────────────────────────────────────────────────────────
# Scalers fitted on Train+Val combined — test is never seen.
# Scalers saved to disk immediately so inference scripts can load them
# without re-running this script.
 
train_val_df = pd.concat([train_df, val_df])
sid_tv       = train_val_df[STORM_COL].values
sid_te       = test_df[STORM_COL].values
 
scaler_X = StandardScaler()
scaler_y = StandardScaler()
 
X_tv       = scaler_X.fit_transform(train_val_df[FEATURES]).astype('float32')
y_tv       = scaler_y.fit_transform(train_val_df[[TARGET]]).astype('float32')
X_te_final = scaler_X.transform(test_df[FEATURES]).astype('float32')
y_te_raw   = test_df[TARGET].values.astype('float32')
 
# Save scalers immediately
joblib.dump(scaler_X, CHECKPOINT_DIR / "scaler_X.pkl")
joblib.dump(scaler_y, CHECKPOINT_DIR / "scaler_y.pkl")
print(f"✅ Scalers saved to {CHECKPOINT_DIR}")
 
# Push to GPU
X_tv_gpu       = torch.tensor(X_tv,       device=PRIMARY)
y_tv_gpu       = torch.tensor(y_tv,       device=PRIMARY)
X_te_final_gpu = torch.tensor(X_te_final, device=PRIMARY)
y_te_raw_gpu   = torch.tensor(y_te_raw,   device=PRIMARY)
 
Y_MEAN = torch.tensor(scaler_y.mean_,  device=PRIMARY, dtype=torch.float32)
Y_STD  = torch.tensor(scaler_y.scale_, device=PRIMARY, dtype=torch.float32)
 
def descale(p: torch.Tensor) -> torch.Tensor:
    """Invert standard-scaling on predicted depth."""
    return p * Y_STD + Y_MEAN
 
print(f"✅ Tensors on {PRIMARY}. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 5 │ Storm-Safe Window Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_storm_windows(X: np.ndarray, y: np.ndarray,
                        storm_ids: np.ndarray, window: int):
    """
    Returns (X_windows, y_targets) where each window is drawn from a
    single storm. Windows bridging storm boundaries are excluded.
    """
    Xw, yw = [], []
    for sid in np.unique(storm_ids):
        mask = storm_ids == sid
        Xs, ys = X[mask], y[mask]
        n = len(Xs)
        if n <= window:
            continue
        for i in range(n - window):
            Xw.append(Xs[i : i + window])
            yw.append(ys[i + window])
    if len(Xw) == 0:
        return (np.empty((0, window, X.shape[1]), dtype='float32'),
                np.empty((0, 1), dtype='float32'))
    return (np.array(Xw, dtype='float32'),
            np.array(yw, dtype='float32').reshape(-1, 1))

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 6 │ Model Architectures
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.fc   = nn.Linear(size, size)
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x):
        return x + self.drop(F.relu(self.fc(self.norm(x))))
 
 
class SotaANN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj   = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        return self.head(self.blocks(F.relu(self.proj(x))))
 
 
class SotaAttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        lstm_drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_drop)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Linear(hidden_size * 2, 1)
 
    def forward(self, x):
        out, _  = self.lstm(x)
        weights = F.softmax(self.attn(out), dim=1)
        context = torch.sum(out * weights, dim=1)
        return self.head(self.norm(context))
 
 
def wrap_model(model: nn.Module) -> nn.Module:
    if N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
    return model.to(PRIMARY)

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 7 │ Hydrological Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────
def nse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-9) -> float:
    var_term = torch.sum((y_true - y_true.mean()) ** 2)
    if torch.isnan(var_term) or torch.isinf(var_term) or var_term.item() <= eps:
        return float("nan")
    num = torch.sum((y_true - y_pred) ** 2)
    return (1 - num / var_term).item()
 
def kge(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    yt_std = float(np.std(y_true))
    yp_std = float(np.std(y_pred))
    yt_mean = float(np.mean(y_true))
    if (not np.isfinite(yt_std)) or (not np.isfinite(yp_std)) or (not np.isfinite(yt_mean)):
        return float("nan")
    if yt_std <= eps or abs(yt_mean) <= eps:
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(r):
        return float("nan")
    alpha = yp_std / yt_std
    beta = float(np.mean(y_pred)) / yt_mean
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
 
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
 
def pbias(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    den = float(np.sum(y_true))
    if (not np.isfinite(den)) or abs(den) <= eps:
        return float("nan")
    return float(100 * np.sum(y_true - y_pred) / den)
 
def eval_metrics(name: str, y_true_np: np.ndarray, y_pred_np: np.ndarray) -> dict:
    y_true_np = np.asarray(y_true_np, dtype=np.float32).reshape(-1)
    y_pred_np = np.asarray(y_pred_np, dtype=np.float32).reshape(-1)
    valid = np.isfinite(y_true_np) & np.isfinite(y_pred_np)
    y_true_np = y_true_np[valid]
    y_pred_np = y_pred_np[valid]
    if y_true_np.size == 0:
        return {'Model': name, 'NSE': np.nan, 'KGE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan}
    yt = torch.tensor(y_true_np, device=PRIMARY)
    yp = torch.tensor(y_pred_np, device=PRIMARY)
    return {
        'Model': name,
        'NSE':   round(nse(yt, yp), 4),
        'KGE':   round(kge(y_true_np, y_pred_np), 4),
        'RMSE':  round(rmse(y_true_np, y_pred_np), 4),
        'PBIAS': round(pbias(y_true_np, y_pred_np), 2),
    }

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 8 │ Final Training — Log-Ridge
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  [1/3] Training Log-Ridge …")

best_lr_fit = None
for cand in lr_candidates:
    p = cand["params"]
    alpha = float(p.get("alpha", 1e-3))
    log_shift = float(p.get("log_shift", 1e-3))
    target_transform = p.get("target_transform", "log")
    lr_weight_lambda = float(p.get("loss_lambda", 2.0))
    lr_use_weighted = bool(p.get("use_weighted_loss", True))
    sample_weight = 1.0 + lr_weight_lambda * np.clip(
        train_val_df[TARGET].values.astype("float32"), a_min=0.0, a_max=None
    )
    fit_kwargs = {"sample_weight": sample_weight} if lr_use_weighted else {}
    print(
        f"   🎯 Log-Ridge: {'weighted' if lr_use_weighted else 'unweighted'} fit "
        f"(lambda={lr_weight_lambda:.3f})"
    )

    if target_transform == "plain":
        model = Ridge(alpha=alpha).fit(
            train_val_df[FEATURES], train_val_df[TARGET].values, **fit_kwargs
        )
        tr_preds = model.predict(train_val_df[FEATURES])
        te_preds = model.predict(test_df[FEATURES])
    else:
        model = Ridge(alpha=alpha).fit(
            train_val_df[FEATURES],
            np.log(train_val_df[TARGET] + log_shift),
            **fit_kwargs
        )
        tr_preds = np.exp(model.predict(train_val_df[FEATURES])) - log_shift
        te_preds = np.exp(model.predict(test_df[FEATURES])) - log_shift

    tr_metrics = eval_metrics("Log-Ridge", train_val_df[TARGET].values.astype('float32'), tr_preds.astype('float32'))
    fit_score = tr_metrics["NSE"]
    if (best_lr_fit is None) or (fit_score > best_lr_fit["fit_score"]):
        best_lr_fit = {
            "model": model,
            "params": p,
            "train_preds": tr_preds.astype('float32'),
            "test_preds": te_preds.astype('float32'),
            "train_metrics": tr_metrics,
            "fit_score": fit_score,
            "trial_number": cand["trial_number"],
            "val_nse": cand["val_nse"],
            "val_kge": cand["val_kge"],
        }

lr_final = best_lr_fit["model"]
bp_lr = best_lr_fit["params"]
lr_train_preds = best_lr_fit["train_preds"]
lr_preds = best_lr_fit["test_preds"]
joblib.dump(lr_final, CHECKPOINT_DIR / "log_ridge_final.pkl")
print(f"   ✅ Log-Ridge saved (trial={best_lr_fit['trial_number']}, train NSE={best_lr_fit['train_metrics']['NSE']:.4f}).")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 9 │ Final Training — Residual ANN (with early stopping)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  [2/3] Training Residual ANN …")
 
# ── VRAM guard before allocating model ───────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()
require_vram(gb_needed=2.0, label="Res-ANN init")
 
EPOCHS_ANN = 100
PATIENCE   = 15

# Hold out the last 15% of train_val storms as a stopping signal only —
# these rows are NOT used for metric reporting (test_df handles that).
n_tv_storms = len(np.unique(sid_tv))
stop_storms = np.unique(sid_tv)[int(n_tv_storms * 0.85):]
stop_mask   = np.isin(sid_tv, stop_storms)

X_stop_gpu = torch.tensor(X_tv[stop_mask],  device=PRIMARY)
y_stop_raw = scaler_y.inverse_transform(y_tv[stop_mask]).flatten().astype('float32')
y_stop_gpu = torch.tensor(y_stop_raw, device=PRIMARY)

X_fit_gpu  = torch.tensor(X_tv[~stop_mask], device=PRIMARY)
y_fit_gpu  = torch.tensor(y_tv[~stop_mask], device=PRIMARY)

best_ann_fit = None
for cand in ann_candidates:
    p = cand["params"]
    ann_model = wrap_model(SotaANN(
        len(FEATURES), int(p["hidden_size"]),
        int(p["n_layers"]), float(p["dropout"])
    ))
    loss_fn = build_loss_fn(p, model_name="Res-ANN")
    wd = float(p.get("weight_decay", 0.0))
    opt_ann = optim.AdamW(ann_model.parameters(), lr=float(p["lr"]), weight_decay=wd)
    sched_ann = optim.lr_scheduler.CosineAnnealingLR(opt_ann, T_max=EPOCHS_ANN)
    batch_ann = 512

    best_stop_nse, wait = float('-inf'), 0
    ann_ckpt = CHECKPOINT_DIR / f"ann_best_trial_{cand['trial_number'] if cand['trial_number'] is not None else 'fallback'}.pt"

    ann_model.train()
    for epoch in range(EPOCHS_ANN):
        perm = torch.randperm(len(X_fit_gpu), device=PRIMARY)
        for i in range(0, len(X_fit_gpu), batch_ann):
            idx = perm[i : i + batch_ann]
            train_step(ann_model, opt_ann, scaler_amp, X_fit_gpu[idx], y_fit_gpu[idx], loss_fn)
        sched_ann.step()

        ann_model.eval()
        with torch.no_grad(), autocast(device_type='cuda'):
            stop_preds = descale(ann_model(X_stop_gpu)).flatten()
            stop_nse = nse(y_stop_gpu, stop_preds)

        if stop_nse > best_stop_nse + 1e-4:
            best_stop_nse = stop_nse
            wait = 0
            torch.save(ann_model.state_dict(), ann_ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        ann_model.train()

    ann_model.load_state_dict(torch.load(ann_ckpt))
    ann_model.eval()
    with torch.no_grad():
        ann_preds_test = descale(ann_model(X_te_final_gpu)).cpu().numpy().flatten().astype('float32')
        ann_preds_train = descale(ann_model(X_tv_gpu)).cpu().numpy().flatten().astype('float32')
    tr_metrics = eval_metrics("Res-ANN", train_val_df[TARGET].values.astype('float32'), ann_preds_train)
    fit_score = tr_metrics["NSE"]
    if (best_ann_fit is None) or (fit_score > best_ann_fit["fit_score"]):
        best_ann_fit = {
            "state_dict": ann_model.state_dict(),
            "params": p,
            "train_preds": ann_preds_train,
            "test_preds": ann_preds_test,
            "train_metrics": tr_metrics,
            "fit_score": fit_score,
            "trial_number": cand["trial_number"],
            "val_nse": cand["val_nse"],
            "val_kge": cand["val_kge"],
        }

    del ann_model, opt_ann
    torch.cuda.empty_cache()
    gc.collect()

bp_ann = best_ann_fit["params"]
ann_preds = best_ann_fit["test_preds"]
ann_train_preds = best_ann_fit["train_preds"]
torch.save({
    'model_state': best_ann_fit["state_dict"],
    'best_params': bp_ann,
    'val_nse':     best_ann_fit["val_nse"],
}, CHECKPOINT_DIR / "ann_final.pt")

# %%
# ── Free ANN tensors before LSTM ─────────────────────────────────────────────
del X_stop_gpu, y_stop_gpu, X_fit_gpu, y_fit_gpu
torch.cuda.empty_cache()
gc.collect()
print(f"   ✅ Res-ANN saved (trial={best_ann_fit['trial_number']}, train NSE={best_ann_fit['train_metrics']['NSE']:.4f}).")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 10 │ Final Training — Attention-LSTM (with early stopping)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  [3/3] Training Attention-LSTM …")
 
# ── VRAM guard before allocating model ───────────────────────────────────────
require_vram(gb_needed=2.0, label="Attn-LSTM init")
 
WINDOW_FINAL = bp_lstm['window_size']
EPOCHS_LSTM  = 100
PATIENCE_L   = 15

y_tv_sc = scaler_y.transform(train_val_df[[TARGET]]).astype('float32')
y_te_sc = scaler_y.transform(test_df[[TARGET]]).astype('float32')

best_lstm_fit = None
for cand in lstm_candidates:
    p = cand["params"]
    window = int(p["window_size"])
    Xtv_w, ytv_w = build_storm_windows(X_tv, y_tv_sc, sid_tv, window)
    Xte_w, yte_w = build_storm_windows(X_te_final, y_te_sc, sid_te, window)
    if len(Xtv_w) == 0 or len(Xte_w) == 0:
        continue

    Xtv_w_cpu = torch.tensor(Xtv_w, dtype=torch.float32)
    ytv_w_cpu = torch.tensor(ytv_w, dtype=torch.float32)
    Xte_w_cpu = torch.tensor(Xte_w, dtype=torch.float32)

    n_stop_l = max(1, int(len(Xtv_w_cpu) * 0.15))
    X_fit_l  = Xtv_w_cpu[:-n_stop_l]
    y_fit_l  = ytv_w_cpu[:-n_stop_l]
    X_stop_l = Xtv_w_cpu[-n_stop_l:]
    y_stop_l = ytv_w_cpu[-n_stop_l:]

    lstm_model = wrap_model(SotaAttentionLSTM(
        len(FEATURES), int(p["hidden_size"]),
        int(p["n_layers"]), float(p["dropout"])
    ))
    loss_fn = build_loss_fn(p, model_name="Attn-LSTM")
    wd = float(p.get("weight_decay", 0.0))
    opt_lstm = optim.AdamW(lstm_model.parameters(), lr=float(p["lr"]), weight_decay=wd)
    sched_lstm = optim.lr_scheduler.CosineAnnealingLR(opt_lstm, T_max=EPOCHS_LSTM)

    _probe_gpu = X_fit_l[:1].to(PRIMARY)
    probe_n = max(1, int(p.get("batch_size", 2048)))
    batch_lstm = 64
    del _probe_gpu
    torch.cuda.empty_cache()

    best_stop_nse_l, wait_l = float('-inf'), 0
    lstm_ckpt = CHECKPOINT_DIR / f"lstm_best_trial_{cand['trial_number'] if cand['trial_number'] is not None else 'fallback'}.pt"

    lstm_model.train()
    for epoch in range(EPOCHS_LSTM):
        perm = torch.randperm(len(X_fit_l))
        for i in range(0, len(X_fit_l), batch_lstm):
            idx = perm[i : i + batch_lstm]
            bx  = X_fit_l[idx].to(PRIMARY, non_blocking=True)
            by  = y_fit_l[idx].to(PRIMARY, non_blocking=True)
            train_step(lstm_model, opt_lstm, scaler_amp, bx, by, loss_fn, clip_grad=1.0)
        sched_lstm.step()

        lstm_model.eval()
        with torch.no_grad():
            all_p = []
            for j in range(0, len(X_stop_l), batch_lstm):
                all_p.append(lstm_model(X_stop_l[j : j + batch_lstm].to(PRIMARY)))
            preds_s = torch.cat(all_p)
            y_stop_d = descale(y_stop_l.to(PRIMARY)).flatten()
            p_stop_d = descale(preds_s).flatten()
            stop_nse_l = nse(y_stop_d, p_stop_d)

        if stop_nse_l > best_stop_nse_l + 1e-4:
            best_stop_nse_l = stop_nse_l
            wait_l = 0
            torch.save(lstm_model.state_dict(), lstm_ckpt)
        else:
            wait_l += 1
            if wait_l >= PATIENCE_L:
                break
        lstm_model.train()

    lstm_model.load_state_dict(torch.load(lstm_ckpt))
    lstm_model.eval()
    with torch.no_grad():
        lstm_preds_s = torch.cat(
            [lstm_model(Xte_w_cpu[i : i + batch_lstm].to(PRIMARY))
             for i in range(0, len(Xte_w_cpu), batch_lstm)]
        )
        lstm_preds_test = descale(lstm_preds_s).cpu().numpy().flatten().astype('float32')
        lstm_obs_test = descale(torch.tensor(yte_w, device=PRIMARY)).cpu().numpy().flatten().astype('float32')

        lstm_train_preds_s = torch.cat(
            [lstm_model(Xtv_w_cpu[i : i + batch_lstm].to(PRIMARY))
             for i in range(0, len(Xtv_w_cpu), batch_lstm)]
        )
        lstm_preds_train = descale(lstm_train_preds_s).cpu().numpy().flatten().astype('float32')
        lstm_obs_train = descale(torch.tensor(ytv_w, device=PRIMARY)).cpu().numpy().flatten().astype('float32')

    tr_metrics = eval_metrics("Attn-LSTM", lstm_obs_train, lstm_preds_train)
    fit_score = tr_metrics["NSE"]
    if (best_lstm_fit is None) or (fit_score > best_lstm_fit["fit_score"]):
        best_lstm_fit = {
            "state_dict": lstm_model.state_dict(),
            "params": p,
            "window": window,
            "train_preds": lstm_preds_train,
            "train_obs": lstm_obs_train,
            "test_preds": lstm_preds_test,
            "test_obs": lstm_obs_test,
            "train_metrics": tr_metrics,
            "fit_score": fit_score,
            "trial_number": cand["trial_number"],
            "val_nse": cand["val_nse"],
            "val_kge": cand["val_kge"],
        }

    del Xtv_w_cpu, ytv_w_cpu, Xte_w_cpu, X_fit_l, y_fit_l, X_stop_l, y_stop_l
    del lstm_model, opt_lstm
    gc.collect()
    torch.cuda.empty_cache()

if best_lstm_fit is None:
    raise RuntimeError("No valid LSTM candidate produced windowed train/test data.")

bp_lstm = best_lstm_fit["params"]
WINDOW_FINAL = best_lstm_fit["window"]
lstm_preds = best_lstm_fit["test_preds"]
lstm_obs = best_lstm_fit["test_obs"]
lstm_train_preds = best_lstm_fit["train_preds"]
lstm_train_obs = best_lstm_fit["train_obs"]
torch.save({
    'model_state': best_lstm_fit["state_dict"],
    'best_params': bp_lstm,
    'val_nse':     best_lstm_fit["val_nse"],
    'window_size': WINDOW_FINAL,
}, CHECKPOINT_DIR / "lstm_final.pt")

gc.collect()
torch.cuda.empty_cache()
print(f"   ✅ Attn-LSTM saved (trial={best_lstm_fit['trial_number']}, train NSE={best_lstm_fit['train_metrics']['NSE']:.4f}).")
print(f"\n✅ All models trained and checkpointed to {CHECKPOINT_DIR}")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 11 │ Test-Set Metrics  (first and only evaluation on test data)
# ─────────────────────────────────────────────────────────────────────────────
train_metrics = [
    eval_metrics("Log-Ridge", train_val_df[TARGET].values.astype('float32'), lr_train_preds),
    eval_metrics("Res-ANN",   train_val_df[TARGET].values.astype('float32'), ann_train_preds),
    eval_metrics("Attn-LSTM", lstm_train_obs, lstm_train_preds),
]
train_metrics_df = pd.DataFrame(train_metrics).set_index('Model')

metrics = [
    eval_metrics("Log-Ridge", y_te_raw,  lr_preds),
    eval_metrics("Res-ANN",   y_te_raw,  ann_preds),
    eval_metrics("Attn-LSTM", lstm_obs,  lstm_preds),
]
metrics_df = pd.DataFrame(metrics).set_index('Model')

gap_df = (train_metrics_df[["NSE", "KGE"]] - metrics_df[["NSE", "KGE"]]).rename(
    columns={"NSE": "Delta_NSE_train_minus_test", "KGE": "Delta_KGE_train_minus_test"}
)

print("\n📊 ── Final Train-Set Metrics ─────────────────────────")
print(f"{'':20} {'NSE':>8} {'KGE':>8} {'RMSE(in)':>10} {'PBIAS%':>8}")
print(f"{'─'*56}")
for name, row in train_metrics_df.iterrows():
    print(f"{name:20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
          f"{row['RMSE']:>10.4f} {row['PBIAS']:>8.2f}")
print(f"{'─'*56}")

print("\n📊 ── Final Test-Set Metrics ──────────────────────────")
print(f"{'':20} {'NSE':>8} {'KGE':>8} {'RMSE(in)':>10} {'PBIAS%':>8}")
print(f"{'─'*56}")
for name, row in metrics_df.iterrows():
    print(f"{name:20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
          f"{row['RMSE']:>10.4f} {row['PBIAS']:>8.2f}")
print(f"{'─'*56}")
print("  NSE/KGE: 1=perfect | PBIAS: 0%=no bias, +=under, -=over")

print("\n📉 ── Overfit Gap (Train - Test) ─────────────────────")
for name, row in gap_df.iterrows():
    print(f"{name:20} ΔNSE={row['Delta_NSE_train_minus_test']:+.4f}  ΔKGE={row['Delta_KGE_train_minus_test']:+.4f}")

# %%
# ── Persist metrics and run metadata to disk ──────────────────────────────────
run_log = {
    "timestamp":    datetime.now().isoformat(),
    "best_params":  {
        "log_ridge": bp_lr,
        "res_ann": bp_ann,
        "attn_lstm": bp_lstm,
    },
    "train_metrics": train_metrics_df.to_dict(),
    "test_metrics": metrics_df.to_dict(),
    "overfit_gaps": gap_df.to_dict(),
    "guardrails": {
        "top_n_candidates": TOP_N_CANDIDATES,
        "val_nse_tolerance": VAL_NSE_TOL,
        "val_kge_tolerance": VAL_KGE_TOL,
        "selected_trials": {
            "log_ridge": {
                "trial_number": best_lr_fit["trial_number"],
                "val_nse": best_lr_fit["val_nse"],
                "val_kge": best_lr_fit["val_kge"],
            },
            "res_ann": {
                "trial_number": best_ann_fit["trial_number"],
                "val_nse": best_ann_fit["val_nse"],
                "val_kge": best_ann_fit["val_kge"],
            },
            "attn_lstm": {
                "trial_number": best_lstm_fit["trial_number"],
                "val_nse": best_lstm_fit["val_nse"],
                "val_kge": best_lstm_fit["val_kge"],
            },
        },
    },
    "data": {
        "train_rows":   int(len(train_df)),
        "val_rows":     int(len(val_df)),
        "test_rows":    int(len(test_df)),
        "train_storms": int(len(train_storms)),
        "val_storms":   int(len(val_storms)),
        "test_storms":  int(len(test_storms)),
    },
}
 
log_path = RESULTS_DIR / "run_log.json"
with open(log_path, "w") as f:
    json.dump(run_log, f, indent=2)
print(f"\n✅ Run log saved → {log_path}")

# --- Row-level errors for Log-Ridge + Res-ANN (aligned to test_df rows) ---
deployment_col = "deployment_id" if "deployment_id" in test_df.columns else STORM_COL
time_col = None
for candidate in ["time", "timestamp", "datetime"]:
    if candidate in test_df.columns:
        time_col = candidate
        break

base_cols = [deployment_col, STORM_COL, TARGET]
if time_col is not None:
    base_cols.insert(1, time_col)
base_cols = list(dict.fromkeys(base_cols))  # preserve order, remove duplicates

examples = test_df[base_cols].copy()
if deployment_col != "deployment_id":
    examples["deployment_id"] = test_df[deployment_col].values
if time_col is not None and time_col != "time":
    examples["time"] = test_df[time_col].values

if "time" not in examples.columns:
    examples["time"] = np.arange(len(examples), dtype=np.int64)

examples["pred_log_ridge"] = lr_preds
examples["pred_res_ann"] = ann_preds

for model in ["log_ridge", "res_ann"]:
    examples[f"err_{model}"] = examples[TARGET] - examples[f"pred_{model}"]
    examples[f"abs_err_{model}"] = examples[f"err_{model}"].abs()

# "Got right" threshold (edit as needed)
HIT_TOL = 0.25  # inches
examples["hit_log_ridge"] = examples["abs_err_log_ridge"] <= HIT_TOL
examples["hit_res_ann"] = examples["abs_err_res_ann"] <= HIT_TOL

# Top examples
best_ann  = examples.nsmallest(30, "abs_err_res_ann")
worst_ann = examples.nlargest(30, "abs_err_res_ann")
best_lr   = examples.nsmallest(30, "abs_err_log_ridge")
worst_lr  = examples.nlargest(30, "abs_err_log_ridge")

# Storm-level failure cases
storm_fail_ann = (
    examples.groupby([STORM_COL, "deployment_id"], as_index=False)
    .agg(mae_ann=("abs_err_res_ann", "mean"),
         peak_obs=(TARGET, "max"),
         peak_pred_ann=("pred_res_ann", "max"),
         n_rows=(TARGET, "size"))
    .sort_values("mae_ann", ascending=False)
)

# Save for review
out_dir = RESULTS_DIR / "error_examples"
out_dir.mkdir(parents=True, exist_ok=True)
examples.to_parquet(out_dir / "test_predictions_errors.parquet", index=False)
best_ann.to_csv(out_dir / "best_ann_examples.csv", index=False)
worst_ann.to_csv(out_dir / "worst_ann_examples.csv", index=False)
best_lr.to_csv(out_dir / "best_logridge_examples.csv", index=False)
worst_lr.to_csv(out_dir / "worst_logridge_examples.csv", index=False)
storm_fail_ann.head(100).to_csv(out_dir / "storm_failures_ann.csv", index=False)


# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 12 │ Visualisation
# ─────────────────────────────────────────────────────────────────────────────
# Re-load ANN for inference only — was deleted after Block 9 to free VRAM.
ann_eval = wrap_model(SotaANN(
    len(FEATURES), bp_ann['hidden_size'],
    bp_ann['n_layers'], bp_ann['dropout']
))
ann_eval.load_state_dict(torch.load(CHECKPOINT_DIR / "ann_final.pt")['model_state'])
ann_eval.eval()
 
with torch.no_grad():
    ann_preds_plot = descale(ann_eval(X_te_final_gpu)).cpu().numpy().flatten()
 
COLORS = {
    'obs':  '#1a1a2e',
    'lr':   '#e67e22',
    'ann':  '#8e44ad',
    'lstm': '#16a085',
    'grid': '#cccccc',
}
 
fig = plt.figure(figsize=(18, 16), dpi=150)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)
 
def pick_display_storm(min_rows=60):
    for sid in test_storms:
        n = (test_df[STORM_COL] == sid).sum()
        if n >= min_rows:
            return sid
    return test_storms[0]
 
focus_sid  = pick_display_storm()
storm_mask = test_df[STORM_COL].values == focus_sid
storm_obs  = test_df.loc[test_df[STORM_COL] == focus_sid, TARGET].values
storm_lr   = lr_preds[storm_mask]
storm_ann  = ann_preds_plot[storm_mask]
t_axis     = np.arange(len(storm_obs))

# %%
# ── Panel A: ANN & Log-Ridge storm time-series ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(t_axis, 0, storm_obs, color=COLORS['obs'], alpha=0.12)
ax1.plot(t_axis, storm_obs, label='Observed (FloodNet)',
         color=COLORS['obs'], lw=2.5, alpha=0.95, zorder=3)
ax1.plot(t_axis, storm_lr,
         label=(f"Log-Ridge  NSE={metrics_df.loc['Log-Ridge','NSE']:.3f}  "
                f"KGE={metrics_df.loc['Log-Ridge','KGE']:.3f}"),
         color=COLORS['lr'], ls='--', lw=1.8, zorder=2)
ax1.plot(t_axis, storm_ann,
         label=(f"Res-ANN    NSE={metrics_df.loc['Res-ANN','NSE']:.3f}  "
                f"KGE={metrics_df.loc['Res-ANN','KGE']:.3f}"),
         color=COLORS['ann'], lw=2.0, zorder=2)
ax1.set_title(f"Storm Event Comparison — Storm ID: {focus_sid}",
              fontsize=13, fontweight='bold')
ax1.set_ylabel("Water Depth (in)", fontsize=11)
ax1.set_xlabel("Timestep (min)", fontsize=11)
ax1.legend(fontsize=9, framealpha=0.9)
ax1.grid(True, color=COLORS['grid'], alpha=0.5)

# %%
# ── Panel B: LSTM storm segment ───────────────────────────────────────────────
DISP = min(800, len(lstm_preds))
ax2  = fig.add_subplot(gs[1, :])
t2   = np.arange(DISP)
ax2.fill_between(t2, 0, lstm_obs[:DISP], color=COLORS['obs'], alpha=0.12)
ax2.plot(t2, lstm_obs[:DISP], label='Observed (FloodNet)',
         color=COLORS['obs'], lw=2.5, alpha=0.95, zorder=3)
ax2.plot(t2, lstm_preds[:DISP],
         label=(f"Attn-LSTM  NSE={metrics_df.loc['Attn-LSTM','NSE']:.3f}  "
                f"KGE={metrics_df.loc['Attn-LSTM','KGE']:.3f}"),
         color=COLORS['lstm'], lw=2.0, zorder=2)
ax2.set_title("Attention-LSTM — Test Set Segment",
              fontsize=13, fontweight='bold')
ax2.set_ylabel("Water Depth (in)", fontsize=11)
ax2.set_xlabel("Timestep (min)", fontsize=11)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, color=COLORS['grid'], alpha=0.5)

# %%
# ── Panel C: Scatter — Observed vs ANN Predicted ─────────────────────────────
ax3  = fig.add_subplot(gs[2, 0])
lim  = (min(y_te_raw.min(), ann_preds_plot.min()) * 0.95,
        max(y_te_raw.max(), ann_preds_plot.max()) * 1.05)
ax3.scatter(y_te_raw, ann_preds_plot, alpha=0.12, s=3,
            color=COLORS['ann'], rasterized=True)
ax3.plot(lim, lim, 'k--', lw=1.2, label='1:1 line')
ax3.set_xlim(lim); ax3.set_ylim(lim)
ax3.set_title("Res-ANN: Observed vs Predicted", fontsize=12, fontweight='bold')
ax3.set_xlabel("Observed (in)", fontsize=10)
ax3.set_ylabel("Predicted (in)", fontsize=10)
ax3.legend(fontsize=9); ax3.grid(True, color=COLORS['grid'], alpha=0.5)

# %%
# ── Panel D: Dumbbell metric comparison (NSE vs KGE per model) ───────────────
ax4 = fig.add_subplot(gs[2, 1])
models = metrics_df.index.tolist()
y_pos = np.arange(len(models))
nse_vals = metrics_df.loc[models, "NSE"].values
kge_vals = metrics_df.loc[models, "KGE"].values

# Reference bands/lines for quick interpretation
ax4.axvspan(0.65, 1.0, color="#d9f2d9", alpha=0.35, lw=0)
ax4.axvline(0.0, color="black", lw=0.9, ls="--", alpha=0.6)
ax4.axvline(0.5, color="green", lw=0.9, ls=":",  alpha=0.7)
ax4.axvline(0.65, color="green", lw=0.9, ls="--", alpha=0.55)

for i, (nse_v, kge_v) in enumerate(zip(nse_vals, kge_vals)):
    ax4.plot([kge_v, nse_v], [i, i], color="#7f8c8d", lw=2.0, alpha=0.85)

ax4.scatter(nse_vals, y_pos, s=75, color="#2980b9", label="NSE", zorder=3)
ax4.scatter(kge_vals, y_pos, s=75, color="#c0392b", marker="D", label="KGE", zorder=3)

for i, (nse_v, kge_v) in enumerate(zip(nse_vals, kge_vals)):
    ax4.text(nse_v + 0.02, i + 0.06, f"{nse_v:.3f}", fontsize=8, color="#1f4e79")
    ax4.text(kge_v + 0.02, i - 0.16, f"{kge_v:.3f}", fontsize=8, color="#7f1d1d")

ax4.set_yticks(y_pos)
ax4.set_yticklabels(models, fontsize=10)
ax4.set_xlim(-0.65, 1.02)
ax4.set_xlabel("Skill Score  (1 = perfect)", fontsize=10)
ax4.set_title("Model Skill: NSE vs KGE", fontsize=12, fontweight="bold")
ax4.grid(True, color=COLORS["grid"], alpha=0.5, axis="x")
ax4.legend(fontsize=8, loc="lower right", framealpha=0.9)
 
fig.suptitle(
    "NYC FloodNet — Flood Depth Prediction Model Shootout\n"
    "(Storm-aware CV  ·  Train / Val / Test  ·  Early Stopping  ·  2-GPU DataParallel)",
    fontsize=14, fontweight='bold', y=1.01
)
 
OUT_FIG = FIGURES_DIR / "flood_model_shootout.png"
plt.savefig(OUT_FIG, bbox_inches='tight', dpi=150)
plt.show()
print(f"✅ Figure saved → {OUT_FIG}")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 13 │ Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║           Training Complete — Output Summary         ║
╠══════════════════════════════════════════════════════╣
║  Checkpoints  → {str(CHECKPOINT_DIR):<36}║
║    ann_final.pt / lstm_final.pt / log_ridge_final.pkl║
║    scaler_X.pkl / scaler_y.pkl                      ║
║  Run log      → {str(log_path):<36}║
║  Figure       → {str(OUT_FIG):<36}║
╚══════════════════════════════════════════════════════╝
""")
 
 
# ── Entry point guard ─────────────────────────────────────────────────────────
# Keeps SLURM / module imports from triggering training on import.
if __name__ == "__main__":
    pass  # All blocks above run unconditionally in Jupyter.
          # Wrap in main() if converting to a pure .py script.
