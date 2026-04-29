# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 1 │ Imports & Hardware Setup
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import gc
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path
import argparse
 
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
# ── Multi-GPU Setup ──────────────────────────────────────────────────────────
# DataParallel splits each mini-batch across available GPUs automatically.
# PRIMARY is where tensors live; DataParallel handles the rest.
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
# Configuration and Feature Selection
FEATURES = [
    'precip_1hr [inch]', 
    'precip_max_intensity [inch/hour]', 
    'temp_2m [degF]', 
    # 'soil_moisture_05cm [m^3/m^3]',  <-- REMOVE THIS
    'elevation [feet]'
]
TARGET = 'depth_inches'
TV_SPLIT = (0.70, 0.15, 0.15)  # Train / Val / Test proportions

# %%
# CLI flags (parse known args to avoid notebook-injected flags)
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--input-file",
    default="rain_influenced_gages.parquet",
    help="Parquet filename under Data_Files/ or absolute path.",
)
args, _unknown = parser.parse_known_args()

# %%
# 1. Identify the current directory (Handles both .py scripts and Jupyter)
try:
    # Use __file__ for standalone scripts
    current_location = Path(__file__).resolve().parent
except NameError:
    # Use Current Working Directory for Jupyter/Interactive sessions
    current_location = Path.cwd().resolve()

# 2. Navigate to the actual Project Root
# If current_location is 'Finalized_Scripts', move to the parent directory
if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"]:
    PROJECT_ROOT = current_location.parent
else:
    PROJECT_ROOT = current_location

# 3. Define the absolute path to the data
# This ensures the path is /floodnet_work/Data_Files/ instead of /Finalized_Scripts/Data_Files/
data_dir = PROJECT_ROOT / "Data_Files"
input_file = Path(args.input_file)
file_path = input_file if input_file.is_absolute() else (data_dir / input_file)

# 4. Safety Check and Data Loading
if not file_path.exists():
    # This will print the exact path being searched to help with troubleshooting
    raise FileNotFoundError(f"Target file not found at: {file_path}")

df = pd.read_parquet(file_path)

print(f"Successfully loaded data from: {file_path}")

# %%
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
        df_clean = df_clean.merge(storm_meta[['global_storm_id', 'global_event_id']], on='global_storm_id', how='left')
        SPLIT_COL = 'global_event_id'
    else:
        SPLIT_COL = STORM_COL
else:
    SPLIT_COL = STORM_COL

split_ids = df_clean[SPLIT_COL].unique()
n_events = len(split_ids)

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
 
print(f"\n📊 Chronological Event-Based Split (Non-Leaky):")
print(f"   Train : {len(train_df):>8,} rows  ({len(train_events):>4} events)")
print(f"   Val   : {len(val_df):>8,} rows  ({len(val_events):>4} events)")
print(f"   Test  : {len(test_df):>8,} rows  ({len(test_events):>4} events)")


# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 3 │ Scaling & GPU Tensor Push
# ─────────────────────────────────────────────────────────────────────────────
# Scalers fitted ONLY on train to avoid val/test statistics leaking into
# normalisation — a subtle but real form of data snooping.
 
scaler_X = StandardScaler()
scaler_y = StandardScaler()
 
X_tr  = scaler_X.fit_transform(train_df[FEATURES]).astype('float32')
X_val = scaler_X.transform(val_df[FEATURES]).astype('float32')
X_te  = scaler_X.transform(test_df[FEATURES]).astype('float32')
 
y_tr  = scaler_y.fit_transform(train_df[[TARGET]]).astype('float32')
y_val = scaler_y.transform(val_df[[TARGET]]).astype('float32')
 
# Raw (un-scaled) targets for metric evaluation
y_val_raw = val_df[TARGET].values.astype('float32')
y_te_raw  = test_df[TARGET].values.astype('float32')
 
# Storm ID arrays (CPU numpy — used by window builder)
sid_tr  = train_df[STORM_COL].values
sid_val = val_df[STORM_COL].values
sid_te  = test_df[STORM_COL].values
 
# Push tabular tensors to GPU (ANN & Log-Reg eval)
X_tr_gpu      = torch.tensor(X_tr,      device=PRIMARY)
y_tr_gpu      = torch.tensor(y_tr,      device=PRIMARY)
X_val_gpu     = torch.tensor(X_val,     device=PRIMARY)
X_te_gpu      = torch.tensor(X_te,      device=PRIMARY)
y_val_raw_gpu = torch.tensor(y_val_raw, device=PRIMARY)
y_te_raw_gpu  = torch.tensor(y_te_raw,  device=PRIMARY)
 
Y_MEAN = torch.tensor(scaler_y.mean_,  device=PRIMARY, dtype=torch.float32)
Y_STD  = torch.tensor(scaler_y.scale_, device=PRIMARY, dtype=torch.float32)
 
def descale(p: torch.Tensor) -> torch.Tensor:
    """Invert standard-scaling on predicted depth."""
    return p * Y_STD + Y_MEAN
 
print(f"✅ Tensors on {PRIMARY}. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 4 │ LSTM Storm-Safe Window Builder
# ─────────────────────────────────────────────────────────────────────────────
# Windows must NOT span storm boundaries. A window bridging the end of one
# storm and the start of the next would corrupt the temporal context fed to
# the LSTM with irrelevant inter-event data.
 
def build_storm_windows(X: np.ndarray, y: np.ndarray,
                        storm_ids: np.ndarray,
                        window: int):
    """
    Returns (X_windows, y_targets) where each row in X_windows is a
    (window, n_features) sequence drawn from a single storm, and the
    corresponding y_target is the depth at timestep t+1 after the window.
 
    Storms shorter than (window + 1) are skipped — they cannot contribute
    a full window without contaminating the boundary.
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
        return np.empty((0, window, X.shape[1]), dtype='float32'), np.empty((0, 1), dtype='float32')
    return (np.array(Xw, dtype='float32'),
            np.array(yw, dtype='float32').reshape(-1, 1))

# %%
def get_windows(split: str, window: int):
    """Return CPU tensors to save VRAM. Windows move to GPU batch-by-batch."""
    key = (split, window)
    if key not in _WINDOW_CACHE:
        if split == 'train':
            Xw, yw = build_storm_windows(X_tr, y_tr, sid_tr, window)
        elif split == 'val':
            Xw, yw = build_storm_windows(X_val, y_val, sid_val, window)
        else:
            y_te_sc = scaler_y.transform(test_df[[TARGET]]).astype('float32')
            Xw, yw  = build_storm_windows(X_te, y_te_sc, sid_te, window)
        
        # Store on CPU (default)
        _WINDOW_CACHE[key] = (
            torch.tensor(Xw), 
            torch.tensor(yw),
        )
    return _WINDOW_CACHE[key]
 

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 5 │ Model Architectures
# ─────────────────────────────────────────────────────────────────────────────
 
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → ReLU → Dropout + skip."""
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.fc   = nn.Linear(size, size)
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x):
        return x + self.drop(F.relu(self.fc(self.norm(x))))
 
 
class SotaANN(nn.Module):
    """
    Deep residual MLP for tabular flood prediction.
    Input: (B, n_features) → Output: (B, 1) scaled depth.
    Skip connections stabilise training depth up to 6+ layers.
    """
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
    """
    Bidirectional LSTM + soft attention for sequence-to-scalar flood prediction.
    Input: (B, T, n_features) → Output: (B, 1) scaled depth.
    Attention weights allow the model to focus on the most hydrologically
    informative timesteps within each window (e.g., peak intensity).
    Bidirectionality helps leverage the full intra-storm context.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        lstm_drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=lstm_drop)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Linear(hidden_size * 2, 1)
 
    def forward(self, x):
        out, _   = self.lstm(x)                          # (B, T, 2H)
        weights  = F.softmax(self.attn(out), dim=1)      # (B, T, 1)
        context  = torch.sum(out * weights, dim=1)        # (B, 2H)
        return self.head(self.norm(context))
 
 
def wrap_model(model: nn.Module) -> nn.Module:
    """Apply DataParallel across all available GPUs and move to PRIMARY."""
    if N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
    return model.to(PRIMARY)

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 6 │ Hydrological Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────
# Using the standard hydrological evaluation suite rather than RMSE alone.
 
def nse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Nash–Sutcliffe Efficiency. NSE = 1 is perfect; NSE < 0 means the mean
    observed is a better predictor than the model (unacceptable).
    """
    num = torch.sum((y_true - y_pred) ** 2)
    den = torch.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return (1 - num / den).item()
 
 
def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kling–Gupta Efficiency. Decomposes error into correlation (r),
    variability bias (α), and mean bias (β). KGE = 1 is perfect.
    Preferred over NSE for flood peak assessment.
    """
    r     = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        r = 0.0
    alpha = y_pred.std()  / (y_true.std()  + 1e-9)
    beta  = y_pred.mean() / (y_true.mean() + 1e-9)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
 
 
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
 
 
def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percent Bias. 0% = perfect. Positive = model underestimates volume
    (dangerous for flood risk); negative = overestimates.
    """
    return float(100 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))
 
 
def eval_metrics(name: str, y_true_np: np.ndarray,
                 y_pred_np: np.ndarray) -> dict:
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
# BLOCK 7 │ Optuna Objectives  (Val NSE is the optimisation target)
# ─────────────────────────────────────────────────────────────────────────────
# All objectives report VAL performance — never test. Test set is untouched
# until Block 10.
 
# ── 7a. Log-Space Ridge Regression (CPU) ────────────────────────────────────
# Log-transforming the target is hydrologically motivated: flood depths are
# log-normally distributed; the transform stabilises variance and prevents
# negative depth predictions.
 
def objective_log_reg(trial):
    alpha     = trial.suggest_float("alpha",     1e-6, 20.0, log=True)
    log_shift = trial.suggest_float("log_shift", 1e-4,  1.0, log=True)
    target_transform = trial.suggest_categorical("target_transform", ["log", "plain"])

    if target_transform == "log":
        y_tr_log = np.log(train_df[TARGET] + log_shift)
        model = Ridge(alpha=alpha).fit(train_df[FEATURES], y_tr_log)
        preds_val = np.exp(model.predict(val_df[FEATURES])) - log_shift
        preds_tr = np.exp(model.predict(train_df[FEATURES])) - log_shift
    else:
        model = Ridge(alpha=alpha).fit(train_df[FEATURES], train_df[TARGET].values)
        preds_val = model.predict(val_df[FEATURES])
        preds_tr = model.predict(train_df[FEATURES])

    y_val_np  = val_df[TARGET].values
    y_tr_np   = train_df[TARGET].values
    denom     = np.sum((y_val_np - y_val_np.mean()) ** 2) + 1e-9
    val_nse = float(1 - np.sum((y_val_np - preds_val) ** 2) / denom)
    trial.set_user_attr("train_nse", float(1 - np.sum((y_tr_np - preds_tr) ** 2) / (np.sum((y_tr_np - y_tr_np.mean()) ** 2) + 1e-9)))
    trial.set_user_attr("val_nse", val_nse)
    trial.set_user_attr("val_kge", float(kge(y_val_np, preds_val)))
    return val_nse
 

# %%
# ── 7b. Residual ANN (GPU, DataParallel) ────────────────────────────────────
# HuberLoss is more robust than MSE to the large depth spikes during extreme
# events, which would otherwise dominate the gradient signal.

def objective_ann(trial):
    h_size   = trial.suggest_int("hidden_size",  128, 1024, step=128)
    n_layers = trial.suggest_int("n_layers",       2,    6)
    lr       = trial.suggest_float("lr",       5e-5, 5e-3, log=True)
    dropout  = trial.suggest_float("dropout",     0.0, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    loss_name = trial.suggest_categorical("loss_fn", ["huber", "mse"])
    
    # CRITICAL FIX: Lowered from 262,144 to 32,768 for 11GB VRAM safety
    batch_sz = 32768   
 
    model   = wrap_model(SotaANN(len(FEATURES), h_size, n_layers, dropout))
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loss_fn = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()
 
    best_val, patience, wait = float('inf'), 8, 0
 
    for _ in range(40):
        model.train()
        perm = torch.randperm(len(X_tr_gpu), device=PRIMARY)
        for i in range(0, len(X_tr_gpu), batch_sz):
            idx = perm[i : i + batch_sz]
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=PRIMARY.type):
                loss = loss_fn(model(X_tr_gpu[idx]), y_tr_gpu[idx])
            if scaler_amp:
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                loss.backward()
                opt.step()
        sched.step()
 
        model.eval()
        with torch.no_grad(), autocast(device_type=PRIMARY.type):
            val_nse = nse(y_val_raw_gpu, descale(model(X_val_gpu)).flatten())
            tr_nse  = nse(torch.tensor(train_df[TARGET].values.astype('float32'), device=PRIMARY),
                          descale(model(X_tr_gpu)).flatten())
            val_np_preds = descale(model(X_val_gpu)).detach().cpu().numpy().flatten()
            val_kge = kge(y_val_raw, val_np_preds)
 
        val_loss = -val_nse
        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
                
    # Free memory before the next trial
    del model, opt
    torch.cuda.empty_cache()
    trial.set_user_attr("train_nse", float(tr_nse))
    trial.set_user_attr("val_nse", float(-best_val))
    trial.set_user_attr("val_kge", float(val_kge))
    return -best_val

# %%
def objective_lstm(trial):
    global _WINDOW_CACHE
    
    # ── Defensive: rebuild scaling tensors if cleanup wiped them ──────────
    _Y_MEAN = torch.tensor(scaler_y.mean_,  device=PRIMARY, dtype=torch.float32)
    _Y_STD  = torch.tensor(scaler_y.scale_, device=PRIMARY, dtype=torch.float32)

    def _descale(p: torch.Tensor) -> torch.Tensor:
        return p * _Y_STD + _Y_MEAN

    window   = trial.suggest_int("window_size",  30, 120, step=30)
    h_size   = trial.suggest_int("hidden_size",  64, 256, step=32)
    n_layers = trial.suggest_int("n_layers",      1,   3)
    lr       = trial.suggest_float("lr",      5e-5, 2e-3, log=True)
    dropout  = trial.suggest_float("dropout",    0.0, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    loss_name = trial.suggest_categorical("loss_fn", ["huber", "mse"])
    batch_sz = 1024

    Xtw_cpu, ytw_cpu = get_windows('train', window)
    Xvw_cpu, yvw_cpu = get_windows('val',   window)

    if len(Xtw_cpu) == 0:
        return float('-inf')

    model   = wrap_model(SotaAttentionLSTM(len(FEATURES), h_size, n_layers, dropout))
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()

    try:
        best_val_nse = float("-inf")
        best_train_nse = float("-inf")
        best_val_kge = float("-inf")
        for epoch in range(25):
            model.train()
            perm = torch.randperm(len(Xtw_cpu))
            for i in range(0, len(Xtw_cpu), batch_sz):
                idx = perm[i : i + batch_sz]
                bx = Xtw_cpu[idx].to(PRIMARY, non_blocking=True)
                by = ytw_cpu[idx].to(PRIMARY, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=PRIMARY.type):
                    loss = loss_fn(model(bx), by)
                if scaler_amp:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler_amp.step(opt)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval()
            with torch.no_grad():
                all_p = []
                for j in range(0, len(Xvw_cpu), batch_sz):
                    bxv = Xvw_cpu[j : j + batch_sz].to(PRIMARY)
                    all_p.append(model(bxv))
                preds_s = torch.cat(all_p)
                val_nse = nse(
                    _descale(yvw_cpu.to(PRIMARY)).flatten(),
                    _descale(preds_s).flatten()
                )
                val_pred_np = _descale(preds_s).cpu().numpy().flatten()
                val_true_np = _descale(yvw_cpu.to(PRIMARY)).cpu().numpy().flatten()
                val_kge = kge(val_true_np, val_pred_np)

                all_p_tr = []
                for j in range(0, len(Xtw_cpu), batch_sz):
                    bxt = Xtw_cpu[j : j + batch_sz].to(PRIMARY)
                    all_p_tr.append(model(bxt))
                tr_preds = torch.cat(all_p_tr)
                tr_nse = nse(
                    _descale(ytw_cpu.to(PRIMARY)).flatten(),
                    _descale(tr_preds).flatten()
                )
                if val_nse > best_val_nse:
                    best_val_nse = val_nse
                    best_train_nse = tr_nse
                    best_val_kge = val_kge

        trial.set_user_attr("train_nse", float(best_train_nse))
        trial.set_user_attr("val_nse", float(best_val_nse))
        trial.set_user_attr("val_kge", float(best_val_kge))
        return best_val_nse

    except torch.OutOfMemoryError:
        trial.set_user_attr("train_nse", -1.0)
        trial.set_user_attr("val_nse", -1.0)
        trial.set_user_attr("val_kge", -1.0)
        return -1.0
    finally:
        del model, opt
        torch.cuda.empty_cache()

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 8 │ Hyperparameter Search
# ─────────────────────────────────────────────────────────────────────────────
N_TRIALS_LR   = 10
N_TRIALS_ANN  = 10
N_TRIALS_LSTM = 10
 
# ── Define the database path FIRST ──────────────────────────────────────────
# ── Define a NEW database path for this dataset variant ──────────────────────
HPO_DB_NAME = "floodnet_hpo_newfilter.db"
DB = f"sqlite:///{PROJECT_ROOT}/Data_Files/{HPO_DB_NAME}"
print(f"Using Optuna DB: {DB}")

print("🔎 [1/3] Log-Ridge baseline …")
study_lr = optuna.create_study(
    study_name="log_ridge",
    direction="maximize", 
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_lr.optimize(objective_log_reg, n_trials=N_TRIALS_LR)
 
print("🔎 [2/3] Residual ANN …")
study_ann = optuna.create_study(
    study_name="res_ann",
    direction="maximize", 
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_ann.optimize(objective_ann, n_trials=N_TRIALS_ANN)

# %%
# 1. Force the allocator to be more efficient with fragments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Identify and kill every large tensor in the global namespace
for name in list(globals().keys()):
    if not name.startswith('_'):
        obj = globals()[name]
        if torch.is_tensor(obj) or (isinstance(obj, list) and len(obj) > 0 and torch.is_tensor(obj[0])):
            print(f"🧹 Deleting: {name}")
            del globals()[name]

# 3. Standard garbage collection and cache clear
gc.collect()
torch.cuda.empty_cache()

print(f"✅ VRAM Reset. Current Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
# Re-initialize the CPU cache dictionary
_WINDOW_CACHE = {}

def get_windows(split: str, window: int):
    """
    Builds or retrieves windowed data from System RAM.
    This prevents the 5.72 GB 'Megatensor' from crashing the 11GB VRAM.
    """
    key = (split, window)
    if key not in _WINDOW_CACHE:
        if split == 'train':
            Xw, yw = build_storm_windows(X_tr, y_tr, sid_tr, window)
        elif split == 'val':
            Xw, yw = build_storm_windows(X_val, y_val, sid_val, window)
        else:
            # For test set evaluation
            y_te_sc = scaler_y.transform(test_df[[TARGET]]).astype('float32')
            Xw, yw  = build_storm_windows(X_te, y_te_sc, sid_te, window)
        
        # Store as CPU tensors (default)
        _WINDOW_CACHE[key] = (
            torch.tensor(Xw, dtype=torch.float32), 
            torch.tensor(yw, dtype=torch.float32),
        )
        print(f"📦 Cached {split} windows for size {window} on CPU.")
        
    return _WINDOW_CACHE[key]

# %%
# ── Ensure scaling tensors exist before LSTM search ──────────────────────
Y_MEAN = torch.tensor(scaler_y.mean_,  device=PRIMARY, dtype=torch.float32)
Y_STD  = torch.tensor(scaler_y.scale_, device=PRIMARY, dtype=torch.float32)
print(f"✅ Y_MEAN={Y_MEAN.item():.4f}, Y_STD={Y_STD.item():.4f} — ready for LSTM search")

print("🔎 [3/3] Attention-LSTM …")
study_lstm = optuna.create_study(
    study_name="attn_lstm",
    direction="maximize", 
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_lstm.optimize(objective_lstm, n_trials=N_TRIALS_LSTM)
 
print(f"\n{'─'*45}")
print(f"{'Model':<20} {'Val NSE':>10}")
print(f"{'─'*45}")
print(f"{'Log-Ridge':20} {study_lr.best_value:>10.4f}")
print(f"{'Res-ANN':20} {study_ann.best_value:>10.4f}")
print(f"{'Attn-LSTM':20} {study_lstm.best_value:>10.4f}")
print(f"{'─'*45}")
