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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path
import argparse
 
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# Configuration and Feature Selection
FEATURES = [
    'precip_1hr [inch]', 
    'precip_max_intensity [inch/hour]', 
    'precip_incremental [inch]',
    'total_precip_in',
    'temp_2m [degF]', 
    'relative_humidity [percent]', 
    'hours_since_storm_start',
    'storm_duration_hr',
    'peak_intensity_inh',
    'intensity_hits_ge_threshold'
]
TARGET = 'depth_inches'
TV_SPLIT = (0.70, 0.15, 0.15)  # Train / Val / Test proportions

# ── HPO Objective Settings ────────────────────────────────────────────────────
# Primary optimisation target is KGE — it balances correlation, variability
# bias, and mean bias independently, making it harder to game than NSE alone.
# Trials where |PBIAS| exceeds this threshold are penalised to prevent the
# optimizer from accepting models with acceptable KGE but dangerous volume bias.
PBIAS_PRUNE_THRESHOLD = 15.0  # percent — standard hydrological acceptability limit

# %%
# CLI flags (parse known args to avoid notebook-injected flags)
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--input-file",
    default="apparently-darling-gecko.parquet",
    help="Parquet filename under Data_Files/ or absolute path.",
)
args, _unknown = parser.parse_known_args()

# %%
# 1. Identify the current directory (Handles both .py scripts and Jupyter)
try:
    current_location = Path(__file__).resolve().parent
except NameError:
    current_location = Path.cwd().resolve()

# 2. Navigate to the actual Project Root
if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"]:
    PROJECT_ROOT = current_location.parent
else:
    PROJECT_ROOT = current_location

# 3. Define the absolute path to the data
data_dir = PROJECT_ROOT / "Data_Files"
input_file = Path(args.input_file)
file_path = input_file if input_file.is_absolute() else (data_dir / input_file)

# 4. Safety Check and Data Loading
if not file_path.exists():
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
META_CANDIDATES = ["deployment_id", "time", "timestamp", "datetime", "global_storm_id", "storm_start", "storm_end"]
META_COLS = [c for c in META_CANDIDATES if c in df.columns]

all_cols = list(dict.fromkeys(FEATURES + [TARGET, STORM_COL] + META_COLS))
df_clean = df[all_cols].dropna(
    subset=FEATURES + [TARGET, STORM_COL]
).copy()
df_clean[FEATURES + [TARGET]] = df_clean[FEATURES + [TARGET]].astype('float32')

# ── Stratified Event-Based Split (Reactivity-Aware) ─────────────────────────
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

# 1. Aggregate to the Storm Level to calculate Reactivity
storm_metrics = df_clean.groupby(SPLIT_COL).agg(
    max_depth=(TARGET, 'max'),
    total_precip=('total_precip_in', 'max')
).reset_index()

# 2. Calculate Reactivity Index
storm_metrics['reactivity'] = storm_metrics['max_depth'] / (storm_metrics['total_precip'] + 1e-6)

# 3. Bin into 3 Classes using quantiles
storm_metrics['reactivity_class'] = pd.qcut(
    storm_metrics['reactivity'], 
    q=3, 
    labels=['Low', 'Medium', 'High'], 
    duplicates='drop'
)

print("\n🌪️ Storm Reactivity Distribution:")
print(storm_metrics['reactivity_class'].value_counts())

# 4. Perform the Stratified Split
train_pct, val_pct, test_pct = TV_SPLIT

train_storms, temp_storms = train_test_split(
    storm_metrics, 
    train_size=train_pct, 
    stratify=storm_metrics['reactivity_class'],
    random_state=SEED
)

relative_val_pct = val_pct / (val_pct + test_pct) 
val_storms, test_storms = train_test_split(
    temp_storms, 
    train_size=relative_val_pct, 
    stratify=temp_storms['reactivity_class'],
    random_state=SEED
)

train_events = train_storms[SPLIT_COL].values
val_events   = val_storms[SPLIT_COL].values
test_events  = test_storms[SPLIT_COL].values

train_df = df_clean[df_clean[SPLIT_COL].isin(train_events)].copy()
val_df   = df_clean[df_clean[SPLIT_COL].isin(val_events)].copy()
test_df  = df_clean[df_clean[SPLIT_COL].isin(test_events)].copy()
 
print(f"\n📊 Stratified Event-Based Split (Non-Leaky):")
print(f"   Train : {len(train_df):>8,} rows  ({len(train_events):>4} events)")
print(f"   Val   : {len(val_df):>8,} rows  ({len(val_events):>4} events)")
print(f"   Test  : {len(test_df):>8,} rows  ({len(test_events):>4} events)")


# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 3 │ Scaling & GPU Tensor Push
# ─────────────────────────────────────────────────────────────────────────────
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

# Precompute the 90th-percentile depth threshold on the validation set.
# Used by peak_nse() to isolate model performance during high-depth events,
# which are the operationally critical timesteps for flood warning.
VAL_PEAK_THRESHOLD = float(np.percentile(y_val_raw, 90))
print(f"📈 Val 90th-percentile depth threshold: {VAL_PEAK_THRESHOLD:.4f} inches")
 
# Storm ID arrays (CPU numpy — used by window builder)
sid_tr  = train_df[STORM_COL].values
sid_val = val_df[STORM_COL].values
sid_te  = test_df[STORM_COL].values
 
# Push tabular tensors to GPU
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

class AsymmetricWeightedDepthLoss(nn.Module):
    """
    Depth-weighted AND Asymmetric regression loss.
    1. Depth-weighted: High-depth events are penalized more than shallow events.
    2. Asymmetric: Under-predictions (dangerous) are heavily penalized 
       compared to over-predictions (false alarms).
    """
    def __init__(self, base: str = "huber", lambda_weight: float = 2.0, underpredict_penalty: float = 4.0):
        super().__init__()
        self.base = str(base).lower()
        self.lambda_weight = float(lambda_weight)
        self.underpredict_penalty = float(underpredict_penalty) 
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true_scaled: torch.Tensor) -> torch.Tensor:
        if self.base == "mse":
            base_loss = (y_pred - y_true_scaled) ** 2
        else:
            base_loss = self.huber(y_pred, y_true_scaled)
            
        y_true_depth = descale(y_true_scaled)
        depth_weight = 1.0 + self.lambda_weight * torch.clamp(y_true_depth, min=0.0)
        
        asymmetry_weight = torch.where(
            y_pred < y_true_scaled, 
            self.underpredict_penalty, 
            1.0
        )
        
        return (base_loss * depth_weight * asymmetry_weight).mean()
 
print(f"✅ Tensors on {PRIMARY}. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 4 │ LSTM Storm-Safe Window Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_storm_windows(X: np.ndarray, y: np.ndarray,
                        storm_ids: np.ndarray,
                        window: int):
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
        out, _   = self.lstm(x)
        weights  = F.softmax(self.attn(out), dim=1)
        context  = torch.sum(out * weights, dim=1)
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
 
def nse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Nash–Sutcliffe Efficiency. NSE = 1 is perfect; NSE < 0 means the mean
    observed is a better predictor than the model.
    """
    num = torch.sum((y_true - y_pred) ** 2)
    den = torch.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return (1 - num / den).item()
 
 
def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kling–Gupta Efficiency. Decomposes error into correlation (r),
    variability bias (α), and mean bias (β). KGE = 1 is perfect.
    Primary optimisation target: harder to game than NSE because it
    penalises correlation, variability, and mean bias independently.
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


def peak_nse(y_true: np.ndarray, y_pred: np.ndarray,
             threshold: float) -> float:
    """
    NSE computed only on timesteps where observed depth exceeds `threshold`
    (default: validation 90th percentile). Isolates model skill on the
    high-depth events that matter most for flood warning. Logged as a
    user_attr on every trial — not the optimisation target, but a critical
    diagnostic to detect models that are statistically acceptable overall
    but fail during the events that actually count.
    """
    mask = y_true >= threshold
    if mask.sum() < 2:
        return float('nan')
    yt = torch.tensor(y_true[mask], device=PRIMARY)
    yp = torch.tensor(y_pred[mask], device=PRIMARY)
    return nse(yt, yp)


def check_pbias_penalty(pb: float) -> float:
    """
    Return a KGE penalty if |PBIAS| exceeds the acceptability threshold.
    Rather than hard-pruning (which wastes the trial's compute), we apply a
    smooth penalty that drives the optimizer away from biased regions while
    still recording the trial for diagnostics.
    Penalty = 0 when |PBIAS| <= threshold; scales linearly beyond it.
    """
    excess = max(0.0, abs(pb) - PBIAS_PRUNE_THRESHOLD)
    return excess * 0.02   # 0.02 KGE penalty per 1% excess bias
 
 
def eval_metrics(name: str, y_true_np: np.ndarray,
                 y_pred_np: np.ndarray) -> dict:
    yt = torch.tensor(y_true_np, device=PRIMARY)
    yp = torch.tensor(y_pred_np, device=PRIMARY)
    return {
        'Model':    name,
        'KGE':      round(kge(y_true_np, y_pred_np), 4),
        'NSE':      round(nse(yt, yp), 4),
        'RMSE':     round(rmse(y_true_np, y_pred_np), 4),
        'PBIAS':    round(pbias(y_true_np, y_pred_np), 2),
        'Peak_NSE': round(peak_nse(y_true_np, y_pred_np, VAL_PEAK_THRESHOLD), 4),
    }

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 7 │ Optuna Objectives  (Val KGE is the primary optimisation target)
# ─────────────────────────────────────────────────────────────────────────────
# All objectives:
#   • Optimise toward KGE (not NSE) — balances r, α, β independently.
#   • Apply a smooth PBIAS penalty when |PBIAS| > PBIAS_PRUNE_THRESHOLD.
#   • Log NSE, RMSE, PBIAS, peak_NSE, and KGE as user_attrs for post-hoc
#     analysis without losing any trial information.
#   • Test set is untouched until Block 10.
 
# ── 7a. Log-Space Ridge Regression (CPU) ────────────────────────────────────
 
def objective_log_reg(trial):
    alpha            = trial.suggest_float("alpha",            1e-6, 20.0, log=True)
    log_shift        = trial.suggest_float("log_shift",        1e-4,  1.0, log=True)
    target_transform = trial.suggest_categorical("target_transform", ["log", "plain"])
    use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])
    loss_lambda      = trial.suggest_float("loss_lambda",      0.1,  10.0, log=True)

    y_tr_np = train_df[TARGET].values.astype(np.float32)
    sample_weight = 1.0 + loss_lambda * np.clip(y_tr_np, a_min=0.0, a_max=None)
    fit_kwargs = {"sample_weight": sample_weight} if use_weighted_loss else {}

    if target_transform == "log":
        y_tr_log = np.log(train_df[TARGET] + log_shift)
        model = Ridge(alpha=alpha).fit(train_df[FEATURES], y_tr_log, **fit_kwargs)
        preds_val = np.exp(model.predict(val_df[FEATURES])) - log_shift
        preds_tr  = np.exp(model.predict(train_df[FEATURES])) - log_shift
    else:
        model = Ridge(alpha=alpha).fit(train_df[FEATURES], train_df[TARGET].values, **fit_kwargs)
        preds_val = model.predict(val_df[FEATURES])
        preds_tr  = model.predict(train_df[FEATURES])

    y_val_np = val_df[TARGET].values

    val_kge   = kge(y_val_np, preds_val)
    val_pb    = pbias(y_val_np, preds_val)
    val_nse_v = float(1 - np.sum((y_val_np - preds_val) ** 2) /
                      (np.sum((y_val_np - y_val_np.mean()) ** 2) + 1e-9))
    val_rmse  = rmse(y_val_np, preds_val)
    val_pkn   = peak_nse(y_val_np, preds_val, VAL_PEAK_THRESHOLD)
    tr_nse    = float(1 - np.sum((y_tr_np - preds_tr) ** 2) /
                      (np.sum((y_tr_np - y_tr_np.mean()) ** 2) + 1e-9))

    trial.set_user_attr("train_nse",  tr_nse)
    trial.set_user_attr("val_nse",    val_nse_v)
    trial.set_user_attr("val_kge",    val_kge)
    trial.set_user_attr("val_rmse",   val_rmse)
    trial.set_user_attr("val_pbias",  val_pb)
    trial.set_user_attr("val_peak_nse", float(val_pkn) if not np.isnan(val_pkn) else -9.0)

    penalty = check_pbias_penalty(val_pb)
    return val_kge - penalty
 

# %%
# ── 7b. Residual ANN (GPU, DataParallel) ────────────────────────────────────

def objective_ann(trial):
    h_size   = trial.suggest_int("hidden_size",  128, 1024, step=128)
    n_layers = trial.suggest_int("n_layers",       2,    6)
    lr       = trial.suggest_float("lr",       5e-5, 5e-3, log=True)
    dropout  = trial.suggest_float("dropout",     0.0, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    loss_name = trial.suggest_categorical("loss_fn", ["huber", "mse"])
    use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])
    loss_lambda = trial.suggest_float("loss_lambda", 0.1, 10.0, log=True)
    underpredict_penalty = trial.suggest_float("underpredict_penalty", 1.0, 10.0)
    batch_sz = trial.suggest_categorical("batch_size", [2048, 4096, 8192, 16384, 32768])
 
    model   = wrap_model(SotaANN(len(FEATURES), h_size, n_layers, dropout))
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loss_fn = (AsymmetricWeightedDepthLoss(base=loss_name, lambda_weight=loss_lambda, underpredict_penalty=underpredict_penalty)
               if use_weighted_loss else
               (nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()))
 
    best_val_kge  = float('-inf')
    best_val_nse  = float('-inf')
    best_val_rmse = float('inf')
    best_val_pb   = float('nan')
    best_val_pkn  = float('nan')
    best_tr_nse   = float('-inf')
    patience, wait = 8, 0
 
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
            val_preds_gpu = descale(model(X_val_gpu)).flatten()
            val_np_preds  = val_preds_gpu.cpu().numpy()
            tr_preds_gpu  = descale(model(X_tr_gpu)).flatten()

            epoch_kge  = kge(y_val_raw, val_np_preds)
            epoch_nse  = nse(y_val_raw_gpu, val_preds_gpu)
            epoch_rmse = rmse(y_val_raw, val_np_preds)
            epoch_pb   = pbias(y_val_raw, val_np_preds)
            epoch_pkn  = peak_nse(y_val_raw, val_np_preds, VAL_PEAK_THRESHOLD)
            epoch_tr_nse = nse(
                torch.tensor(train_df[TARGET].values.astype('float32'), device=PRIMARY),
                tr_preds_gpu
            )

        # Track best by penalised KGE
        penalised_kge = epoch_kge - check_pbias_penalty(epoch_pb)
        if penalised_kge > best_val_kge - 1e-4:
            best_val_kge  = penalised_kge
            best_val_nse  = epoch_nse
            best_val_rmse = epoch_rmse
            best_val_pb   = epoch_pb
            best_val_pkn  = epoch_pkn
            best_tr_nse   = epoch_tr_nse
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
                
    del model, opt
    torch.cuda.empty_cache()

    trial.set_user_attr("train_nse",    float(best_tr_nse))
    trial.set_user_attr("val_nse",      float(best_val_nse))
    trial.set_user_attr("val_kge",      float(best_val_kge))
    trial.set_user_attr("val_rmse",     float(best_val_rmse))
    trial.set_user_attr("val_pbias",    float(best_val_pb))
    trial.set_user_attr("val_peak_nse", float(best_val_pkn) if not np.isnan(best_val_pkn) else -9.0)

    return best_val_kge  # penalised KGE is the optimisation target

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
    use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])
    loss_lambda = trial.suggest_float("loss_lambda", 0.1, 10.0, log=True)
    underpredict_penalty = trial.suggest_float("underpredict_penalty", 1.0, 10.0)
    batch_sz = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])

    Xtw_cpu, ytw_cpu = get_windows('train', window)
    Xvw_cpu, yvw_cpu = get_windows('val',   window)

    if len(Xtw_cpu) == 0:
        return float('-inf')

    model   = wrap_model(SotaAttentionLSTM(len(FEATURES), h_size, n_layers, dropout))
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = (AsymmetricWeightedDepthLoss(base=loss_name, lambda_weight=loss_lambda, underpredict_penalty=underpredict_penalty)
               if use_weighted_loss else
               (nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()))

    try:
        best_val_kge  = float("-inf")
        best_val_nse  = float("-inf")
        best_val_rmse = float("inf")
        best_val_pb   = float("nan")
        best_val_pkn  = float("nan")
        best_tr_nse   = float("-inf")

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

                val_true_np = _descale(yvw_cpu.to(PRIMARY)).cpu().numpy().flatten()
                val_pred_np = _descale(preds_s).cpu().numpy().flatten()

                epoch_kge  = kge(val_true_np, val_pred_np)
                epoch_nse  = nse(
                    torch.tensor(val_true_np, device=PRIMARY),
                    torch.tensor(val_pred_np, device=PRIMARY)
                )
                epoch_rmse = rmse(val_true_np, val_pred_np)
                epoch_pb   = pbias(val_true_np, val_pred_np)
                epoch_pkn  = peak_nse(val_true_np, val_pred_np, VAL_PEAK_THRESHOLD)

                all_p_tr = []
                for j in range(0, len(Xtw_cpu), batch_sz):
                    bxt = Xtw_cpu[j : j + batch_sz].to(PRIMARY)
                    all_p_tr.append(model(bxt))
                tr_preds  = torch.cat(all_p_tr)
                tr_true_np = _descale(ytw_cpu.to(PRIMARY)).cpu().numpy().flatten()
                tr_pred_np = _descale(tr_preds).cpu().numpy().flatten()
                epoch_tr_nse = nse(
                    torch.tensor(tr_true_np, device=PRIMARY),
                    torch.tensor(tr_pred_np, device=PRIMARY)
                )

                penalised_kge = epoch_kge - check_pbias_penalty(epoch_pb)
                if penalised_kge > best_val_kge:
                    best_val_kge  = penalised_kge
                    best_val_nse  = epoch_nse
                    best_val_rmse = epoch_rmse
                    best_val_pb   = epoch_pb
                    best_val_pkn  = epoch_pkn
                    best_tr_nse   = epoch_tr_nse

        trial.set_user_attr("train_nse",    float(best_tr_nse))
        trial.set_user_attr("val_nse",      float(best_val_nse))
        trial.set_user_attr("val_kge",      float(best_val_kge))
        trial.set_user_attr("val_rmse",     float(best_val_rmse))
        trial.set_user_attr("val_pbias",    float(best_val_pb))
        trial.set_user_attr("val_peak_nse", float(best_val_pkn) if not np.isnan(best_val_pkn) else -9.0)

        return best_val_kge  # penalised KGE is the optimisation target

    except torch.OutOfMemoryError:
        for attr in ("train_nse", "val_nse", "val_kge", "val_rmse", "val_pbias", "val_peak_nse"):
            trial.set_user_attr(attr, -1.0)
        return -1.0
    finally:
        del model, opt
        torch.cuda.empty_cache()

# %%
# %%─────────────────────────────────────────────────────────────────────────
# BLOCK 8 │ Hyperparameter Search
# ─────────────────────────────────────────────────────────────────────────────
N_TRIALS_LR   = 1000
N_TRIALS_ANN  = 1000
N_TRIALS_LSTM = 1000
 
HPO_DB_NAME = "floodnet_hpo_newfilter.db"
DB = f"sqlite:///{PROJECT_ROOT}/Data_Files/{HPO_DB_NAME}"
print(f"Using Optuna DB: {DB}")

print("🔎 [1/3] Log-Ridge baseline …")
study_lr = optuna.create_study(
    study_name="log_ridge",
    direction="maximize",   # maximise penalised KGE
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_lr.optimize(objective_log_reg, n_trials=N_TRIALS_LR)
 
print("🔎 [2/3] Residual ANN …")
study_ann = optuna.create_study(
    study_name="res_ann",
    direction="maximize",   # maximise penalised KGE
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_ann.optimize(objective_ann, n_trials=N_TRIALS_ANN)

# %%
# ── VRAM Cleanup before LSTM search ──────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for name in list(globals().keys()):
    if not name.startswith('_'):
        obj = globals()[name]
        if torch.is_tensor(obj) or (isinstance(obj, list) and len(obj) > 0 and torch.is_tensor(obj[0])):
            print(f"🧹 Deleting: {name}")
            del globals()[name]

gc.collect()
torch.cuda.empty_cache()
print(f"✅ VRAM Reset. Current Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
# Re-initialize the CPU cache dictionary
_WINDOW_CACHE = {}

def get_windows(split: str, window: int):
    """
    Builds or retrieves windowed data from System RAM.
    Prevents the large window tensor from exhausting VRAM.
    """
    key = (split, window)
    if key not in _WINDOW_CACHE:
        if split == 'train':
            Xw, yw = build_storm_windows(X_tr, y_tr, sid_tr, window)
        elif split == 'val':
            Xw, yw = build_storm_windows(X_val, y_val, sid_val, window)
        else:
            y_te_sc = scaler_y.transform(test_df[[TARGET]]).astype('float32')
            Xw, yw  = build_storm_windows(X_te, y_te_sc, sid_te, window)
        
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
    direction="maximize",   # maximise penalised KGE
    sampler=TPESampler(seed=SEED),
    storage=DB,
    load_if_exists=True
)
study_lstm.optimize(objective_lstm, n_trials=N_TRIALS_LSTM)
 
# ── Summary Table ─────────────────────────────────────────────────────────────
# best_value is now penalised KGE; raw KGE and other diagnostics are in
# user_attrs of the best trial for each study.
def _best_attrs(study):
    bt = study.best_trial
    return {
        'pen_kge':   round(study.best_value, 4),
        'kge':       round(bt.user_attrs.get('val_kge',      float('nan')), 4),
        'nse':       round(bt.user_attrs.get('val_nse',      float('nan')), 4),
        'rmse':      round(bt.user_attrs.get('val_rmse',     float('nan')), 4),
        'pbias':     round(bt.user_attrs.get('val_pbias',    float('nan')), 2),
        'peak_nse':  round(bt.user_attrs.get('val_peak_nse', float('nan')), 4),
    }

lr_a   = _best_attrs(study_lr)
ann_a  = _best_attrs(study_ann)
lstm_a = _best_attrs(study_lstm)

print(f"\n{'─'*75}")
print(f"{'Model':<15} {'PenKGE':>8} {'KGE':>8} {'NSE':>8} {'RMSE':>8} {'PBIAS%':>8} {'PeakNSE':>9}")
print(f"{'─'*75}")
for label, a in [('Log-Ridge', lr_a), ('Res-ANN', ann_a), ('Attn-LSTM', lstm_a)]:
    print(f"{label:<15} {a['pen_kge']:>8.4f} {a['kge']:>8.4f} {a['nse']:>8.4f} "
          f"{a['rmse']:>8.4f} {a['pbias']:>8.2f} {a['peak_nse']:>9.4f}")
print(f"{'─'*75}")
print("Note: PenKGE = KGE − PBIAS penalty (optimisation target). "
      "KGE is the raw unpenalised score.")