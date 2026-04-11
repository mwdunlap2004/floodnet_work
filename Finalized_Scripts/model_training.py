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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import optuna
from optuna.samplers import TPESampler
 
warnings.filterwarnings('ignore')

# %%
# ── Multi-GPU Setup ──────────────────────────────────────────────────────────
N_GPUS  = min(torch.cuda.device_count(), 2)
PRIMARY = torch.device("cuda:0")
scaler_amp = GradScaler(device='cuda')
 
print(f"🚀 Using {N_GPUS} GPU(s):")
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
 
FEATURES = [
    'precip_1hr [inch]',
    'precip_max_intensity [inch/hour]',
    'temp_2m [degF]',
    'soil_moisture_05cm [m^3/m^3]',
    'elevation [feet]'
]
TARGET   = 'depth_inches'
TV_SPLIT = (0.70, 0.15, 0.15)
 
DB      = f"sqlite:///{PROJECT_ROOT}/Data_Files/floodnet_hpo.db"
db_path = PROJECT_ROOT / "Data_Files" / "floodnet_hpo.db"
if not db_path.exists():
    raise FileNotFoundError(
        f"Optuna database not found at {db_path}. "
        "Run 01_hpo_search.py first."
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

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 3 │ Data Loading and Storm-Aware Split
# ─────────────────────────────────────────────────────────────────────────────
file_path = DATA_DIR / "delineated_storms.parquet"
if not file_path.exists():
    raise FileNotFoundError(f"Data not found at: {file_path}")
 
df = pd.read_parquet(file_path)
print(f"✅ Loaded data: {len(df):,} rows")
 
# ── Resolve storm identifier column ──────────────────────────────────────────
STORM_COL = None
for candidate in ['storm_id', 'event_id', 'storm', 'event']:
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
df_clean = df[FEATURES + [TARGET, STORM_COL]].dropna().copy()
df_clean[FEATURES + [TARGET]] = df_clean[FEATURES + [TARGET]].astype('float32')
 
# ── Chronological storm-level split ──────────────────────────────────────────
storm_ids = df_clean[STORM_COL].unique()
n_storms  = len(storm_ids)
n_tr      = int(n_storms * TV_SPLIT[0])
n_va      = int(n_storms * TV_SPLIT[1])
 
train_storms = storm_ids[:n_tr]
val_storms   = storm_ids[n_tr : n_tr + n_va]
test_storms  = storm_ids[n_tr + n_va :]
 
train_df = df_clean[df_clean[STORM_COL].isin(train_storms)].copy()
val_df   = df_clean[df_clean[STORM_COL].isin(val_storms)].copy()
test_df  = df_clean[df_clean[STORM_COL].isin(test_storms)].copy()
 
print(f"\n📊 Storm-aware partition:")
print(f"   Train : {len(train_df):>8,} rows  ({len(train_storms):>4} storms)")
print(f"   Val   : {len(val_df):>8,} rows  ({len(val_storms):>4} storms)")
print(f"   Test  : {len(test_df):>8,} rows  ({len(test_storms):>4} storms)")

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
def nse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    num = torch.sum((y_true - y_pred) ** 2)
    den = torch.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return (1 - num / den).item()
 
def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r     = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std()  / (y_true.std()  + 1e-9)
    beta  = y_pred.mean() / (y_true.mean() + 1e-9)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
 
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
 
def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(100 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))
 
def eval_metrics(name: str, y_true_np: np.ndarray, y_pred_np: np.ndarray) -> dict:
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
 
lr_final = Ridge(alpha=bp_lr['alpha']).fit(
    train_val_df[FEATURES],
    np.log(train_val_df[TARGET] + bp_lr['log_shift'])
)
lr_preds = np.exp(lr_final.predict(test_df[FEATURES])) - bp_lr['log_shift']
 
joblib.dump(lr_final, CHECKPOINT_DIR / "log_ridge_final.pkl")
print("   ✅ Log-Ridge saved.")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 9 │ Final Training — Residual ANN (with early stopping)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  [2/3] Training Residual ANN …")
 
# ── VRAM guard before allocating model ───────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()
require_vram(gb_needed=2.0, label="Res-ANN init")
 
EPOCHS_ANN = 60      # generous ceiling — early stopping will cut this short
PATIENCE   = 8
loss_fn    = nn.HuberLoss()
 
ann_final = wrap_model(SotaANN(
    len(FEATURES), bp_ann['hidden_size'],
    bp_ann['n_layers'], bp_ann['dropout']
))
opt_ann   = optim.AdamW(ann_final.parameters(), lr=bp_ann['lr'], weight_decay=1e-4)
sched_ann = optim.lr_scheduler.CosineAnnealingLR(opt_ann, T_max=EPOCHS_ANN)
 
# ── Dynamic batch size — avoids hard-coded OOM ───────────────────────────────
BATCH_ANN = safe_batch_size(ann_final, X_tv_gpu, starting_batch=32768, min_batch=512)
 
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
 
best_stop_nse, wait = float('-inf'), 0
ANN_BEST_PATH = CHECKPOINT_DIR / "ann_best.pt"
 
ann_final.train()
for epoch in range(EPOCHS_ANN):
    perm = torch.randperm(len(X_fit_gpu), device=PRIMARY)
    for i in range(0, len(X_fit_gpu), BATCH_ANN):
        idx = perm[i : i + BATCH_ANN]
        # ── OOM-safe train step ───────────────────────────────────────────
        train_step(ann_final, opt_ann, scaler_amp,
                   X_fit_gpu[idx], y_fit_gpu[idx], loss_fn)
    sched_ann.step()
 
    ann_final.eval()
    with torch.no_grad(), autocast(device_type='cuda'):
        stop_preds = descale(ann_final(X_stop_gpu)).flatten()
        stop_nse   = nse(y_stop_gpu, stop_preds)
 
    if stop_nse > best_stop_nse + 1e-4:
        best_stop_nse = stop_nse
        wait = 0
        torch.save(ann_final.state_dict(), ANN_BEST_PATH)
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"   Early stop at epoch {epoch+1} (best stop-NSE {best_stop_nse:.4f})")
            break
    ann_final.train()
 
# Load the best checkpoint, not the last epoch
ann_final.load_state_dict(torch.load(ANN_BEST_PATH))
ann_final.eval()
 
with torch.no_grad():
    ann_preds = descale(ann_final(X_te_final_gpu)).cpu().numpy().flatten()
 
# Save final state
torch.save({
    'model_state': ann_final.state_dict(),
    'best_params': best_params["res_ann"],
    'val_nse':     study_ann.best_value,
}, CHECKPOINT_DIR / "ann_final.pt")

# %%
# ── Free ANN tensors before LSTM ─────────────────────────────────────────────
del X_stop_gpu, y_stop_gpu, X_fit_gpu, y_fit_gpu
del ann_final          # release GPU weights — LSTM needs the headroom
torch.cuda.empty_cache()
gc.collect()
print("   ✅ Res-ANN saved.")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 10 │ Final Training — Attention-LSTM (with early stopping)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  [3/3] Training Attention-LSTM …")
 
# ── VRAM guard before allocating model ───────────────────────────────────────
require_vram(gb_needed=2.0, label="Attn-LSTM init")
 
WINDOW_FINAL = bp_lstm['window_size']
EPOCHS_LSTM  = 45
PATIENCE_L   = 8
 
y_tv_sc = scaler_y.transform(train_val_df[[TARGET]]).astype('float32')
y_te_sc = scaler_y.transform(test_df[[TARGET]]).astype('float32')
 
Xtv_w, ytv_w = build_storm_windows(X_tv, y_tv_sc, sid_tv, WINDOW_FINAL)
Xte_w, yte_w = build_storm_windows(X_te_final, y_te_sc, sid_te, WINDOW_FINAL)
 
# CPU tensors — stream to GPU batch by batch
Xtv_w_cpu = torch.tensor(Xtv_w, dtype=torch.float32)
ytv_w_cpu = torch.tensor(ytv_w, dtype=torch.float32)
Xte_w_cpu = torch.tensor(Xte_w, dtype=torch.float32)
 
# Stopping split (last 15% of windowed train+val rows)
n_stop_l = int(len(Xtv_w_cpu) * 0.15)
X_fit_l  = Xtv_w_cpu[:-n_stop_l]
y_fit_l  = ytv_w_cpu[:-n_stop_l]
X_stop_l = Xtv_w_cpu[-n_stop_l:]
y_stop_l = ytv_w_cpu[-n_stop_l:]
 
lstm_final = wrap_model(SotaAttentionLSTM(
    len(FEATURES), bp_lstm['hidden_size'],
    bp_lstm['n_layers'], bp_lstm['dropout']
))
opt_lstm   = optim.AdamW(lstm_final.parameters(), lr=bp_lstm['lr'], weight_decay=1e-4)
sched_lstm = optim.lr_scheduler.CosineAnnealingLR(opt_lstm, T_max=EPOCHS_LSTM)
 
# ── Dynamic batch size ────────────────────────────────────────────────────────
# Use a CPU sample streamed to GPU for the probe (avoids pre-loading all windows)
_probe_gpu = X_fit_l[:1].to(PRIMARY)
BATCH_LSTM = safe_batch_size(lstm_final, _probe_gpu.expand(2048, -1, -1),
                             starting_batch=2048, min_batch=64)
del _probe_gpu
torch.cuda.empty_cache()
 
best_stop_nse_l, wait_l = float('-inf'), 0
LSTM_BEST_PATH = CHECKPOINT_DIR / "lstm_best.pt"
 
lstm_final.train()
for epoch in range(EPOCHS_LSTM):
    perm = torch.randperm(len(X_fit_l))
    for i in range(0, len(X_fit_l), BATCH_LSTM):
        idx = perm[i : i + BATCH_LSTM]
        bx  = X_fit_l[idx].to(PRIMARY, non_blocking=True)
        by  = y_fit_l[idx].to(PRIMARY, non_blocking=True)
        # ── OOM-safe train step with gradient clipping ────────────────────
        train_step(lstm_final, opt_lstm, scaler_amp, bx, by, loss_fn, clip_grad=1.0)
    sched_lstm.step()
 
    # Stopping evaluation
    lstm_final.eval()
    with torch.no_grad():
        all_p = []
        for j in range(0, len(X_stop_l), BATCH_LSTM):
            all_p.append(lstm_final(X_stop_l[j : j + BATCH_LSTM].to(PRIMARY)))
        preds_s    = torch.cat(all_p)
        y_stop_d   = descale(y_stop_l.to(PRIMARY)).flatten()
        p_stop_d   = descale(preds_s).flatten()
        stop_nse_l = nse(y_stop_d, p_stop_d)
 
    if stop_nse_l > best_stop_nse_l + 1e-4:
        best_stop_nse_l = stop_nse_l
        wait_l = 0
        torch.save(lstm_final.state_dict(), LSTM_BEST_PATH)
    else:
        wait_l += 1
        if wait_l >= PATIENCE_L:
            print(f"   Early stop at epoch {epoch+1} (best stop-NSE {best_stop_nse_l:.4f})")
            break
    lstm_final.train()
 
# Load the best checkpoint
lstm_final.load_state_dict(torch.load(LSTM_BEST_PATH))
lstm_final.eval()
 
with torch.no_grad():
    lstm_preds_s = torch.cat(
        [lstm_final(Xte_w_cpu[i : i + BATCH_LSTM].to(PRIMARY))
         for i in range(0, len(Xte_w_cpu), BATCH_LSTM)]
    )
    lstm_preds = descale(lstm_preds_s).cpu().numpy().flatten()
 
lstm_obs = descale(torch.tensor(yte_w, device=PRIMARY)).cpu().numpy().flatten()
 
# Save final state
torch.save({
    'model_state': lstm_final.state_dict(),
    'best_params': best_params["attn_lstm"],
    'val_nse':     study_lstm.best_value,
    'window_size': WINDOW_FINAL,
}, CHECKPOINT_DIR / "lstm_final.pt")
 
del X_fit_l, y_fit_l, X_stop_l, y_stop_l
gc.collect()
torch.cuda.empty_cache()
print("   ✅ Attn-LSTM saved.")
print(f"\n✅ All models trained and checkpointed to {CHECKPOINT_DIR}")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 11 │ Test-Set Metrics  (first and only evaluation on test data)
# ─────────────────────────────────────────────────────────────────────────────
metrics = [
    eval_metrics("Log-Ridge", y_te_raw,  lr_preds),
    eval_metrics("Res-ANN",   y_te_raw,  ann_preds),
    eval_metrics("Attn-LSTM", lstm_obs,  lstm_preds),
]
metrics_df = pd.DataFrame(metrics).set_index('Model')
 
print("\n📊 ── Final Test-Set Metrics ──────────────────────────")
print(f"{'':20} {'NSE':>8} {'KGE':>8} {'RMSE(in)':>10} {'PBIAS%':>8}")
print(f"{'─'*56}")
for name, row in metrics_df.iterrows():
    print(f"{name:20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
          f"{row['RMSE']:>10.4f} {row['PBIAS']:>8.2f}")
print(f"{'─'*56}")
print("  NSE/KGE: 1=perfect | PBIAS: 0%=no bias, +=under, -=over")

# %%
# ── Persist metrics and run metadata to disk ──────────────────────────────────
run_log = {
    "timestamp":    datetime.now().isoformat(),
    "best_params":  best_params,         # FIX: was undefined in original
    "test_metrics": metrics_df.to_dict(),
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
# ── Panel D: Grouped metric bars (NSE & KGE per model) ───────────────────────
ax4    = fig.add_subplot(gs[2, 1])
models = metrics_df.index.tolist()
x_pos  = np.arange(len(models))
bw     = 0.32
b1 = ax4.bar(x_pos - bw / 2, metrics_df['NSE'], bw,
             label='NSE', color='#2980b9', alpha=0.87)
b2 = ax4.bar(x_pos + bw / 2, metrics_df['KGE'], bw,
             label='KGE', color='#c0392b', alpha=0.87)
ax4.axhline(0,    color='black', lw=0.8, ls='--', alpha=0.5)
ax4.axhline(0.5,  color='green', lw=0.8, ls=':',  alpha=0.6, label='0.5 (satisfactory)')
ax4.axhline(0.65, color='green', lw=0.8, ls='--', alpha=0.4, label='0.65 (good)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, fontsize=10)
ax4.set_ylim(-0.15, 1.08)
ax4.set_ylabel("Score  (1 = perfect)", fontsize=10)
ax4.set_title("Model Performance: NSE & KGE", fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax4.grid(True, color=COLORS['grid'], alpha=0.5, axis='y')
 
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
             f"{h:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
 
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


