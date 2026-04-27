# %%
# ─────────────────────────────────────────────────────────────────────────────
# 03_local_training.py
# FloodNet — Local Model Training Script (Per-Gage)
# Loads global optimal hyperparameters, then loops through each deployment_id.
# For each gage, it scales data locally, trains bespoke Log-Ridge, Res-ANN, 
# and Attn-LSTM models, evaluates on a held-out Test set, and aggregates the 
# local accuracy metrics.
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
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import optuna

warnings.filterwarnings('ignore')

# ── Multi-GPU Setup & VRAM Tools ─────────────────────────────────────────────
N_GPUS  = min(torch.cuda.device_count(), 2)
PRIMARY = torch.device("cuda:0")
scaler_amp = GradScaler(device='cuda')

print(f"🚀 Using {N_GPUS} GPU(s) for Local Modeling:")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def vram_free_gb(device: int = 0):
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return free / 1e9, total / 1e9

def safe_batch_size(model, sample_input, starting_batch=32768, min_batch=512):
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
            return batch
        except torch.cuda.OutOfMemoryError:
            batch //= 2
    return min_batch

def train_step(model, opt, amp_scaler, bx, by, loss_fn, clip_grad=None):
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
        return None

def build_storm_windows(X, y, storm_ids, window):
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
    return np.array(Xw, dtype='float32'), np.array(yw, dtype='float32').reshape(-1, 1)

# ── Architectures ────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.fc   = nn.Linear(size, size)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return x + self.drop(F.relu(self.fc(self.norm(x))))

class SotaANN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.head(self.blocks(F.relu(self.proj(x))))

class SotaAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.15):
        super().__init__()
        lstm_drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=lstm_drop)
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

# ── Metrics ──────────────────────────────────────────────────────────────────
def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return 1 - num / den

def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std() / (y_true.std() + 1e-9)
    beta = y_pred.mean() / (y_true.mean() + 1e-9)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# ── Configuration & Paths ────────────────────────────────────────────────────
try:
    current_location = Path(__file__).resolve().parent
except NameError:
    current_location = Path.cwd().resolve()

PROJECT_ROOT = current_location.parent if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"] else current_location

DATA_DIR       = PROJECT_ROOT / "Data_Files"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "local_models"
RESULTS_DIR    = PROJECT_ROOT / "results" / "local_models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['precip_1hr [inch]', 'precip_max_intensity [inch/hour]', 'temp_2m [degF]', 'soil_moisture_05cm [m^3/m^3]', 'elevation [feet]']
TARGET   = 'depth_inches'
TV_SPLIT = (0.70, 0.15, 0.15)

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", default="rain_influenced_gages.parquet")
args, _ = parser.parse_known_args()

file_path = Path(args.input_file) if Path(args.input_file).is_absolute() else (DATA_DIR / args.input_file)
df = pd.read_parquet(file_path)

# ── Identify Core Columns ────────────────────────────────────────────────────
STORM_COL = next((c for c in ['storm_id', 'event_id', 'storm', 'event'] if c in df.columns), '_storm_id')
if STORM_COL == '_storm_id':
    df['_storm_id'] = np.arange(len(df)) // 500

DEPLOY_COL = next((c for c in ['deployment_id', 'gage_id', 'sensor_id'] if c in df.columns), None)
if not DEPLOY_COL:
    raise ValueError("Could not identify a deployment/gage ID column to split local models.")

# Load Global HPO Best Params (Used as architecture blueprints for local models)
HPO_DB_NAME = "floodnet_hpo_newfilter.db"
DB = f"sqlite:///{PROJECT_ROOT}/Data_Files/{HPO_DB_NAME}"
study_lr = optuna.load_study(study_name="log_ridge", storage=DB)
study_ann = optuna.load_study(study_name="res_ann", storage=DB)
study_lstm = optuna.load_study(study_name="attn_lstm", storage=DB)

bp_lr = study_lr.best_params
bp_ann = study_ann.best_params
bp_lstm = study_lstm.best_params

# ── Global Storm Split (Ensures local models train/test on the same events) ──
df_clean = df.dropna(subset=FEATURES + [TARGET, STORM_COL, DEPLOY_COL]).copy()
df_clean[FEATURES + [TARGET]] = df_clean[FEATURES + [TARGET]].astype('float32')

global_storms = df_clean[STORM_COL].unique()
n_tr = int(len(global_storms) * TV_SPLIT[0])
n_va = int(len(global_storms) * TV_SPLIT[1])

train_storms = global_storms[:n_tr]
val_storms   = global_storms[n_tr : n_tr + n_va]
test_storms  = global_storms[n_tr + n_va :]

print(f"✅ Loaded {len(df_clean):,} rows across {df_clean[DEPLOY_COL].nunique()} unique gages.")

# ── Local Training Loop ──────────────────────────────────────────────────────
local_results = []
gages = df_clean[DEPLOY_COL].unique()

for idx, gage in enumerate(gages):
    print(f"\n[{idx+1}/{len(gages)}] 📍 Training Local Models for Gage: {gage}")
    gage_df = df_clean[df_clean[DEPLOY_COL] == gage].copy()
    
    tr_df = gage_df[gage_df[STORM_COL].isin(train_storms)]
    va_df = gage_df[gage_df[STORM_COL].isin(val_storms)]
    te_df = gage_df[gage_df[STORM_COL].isin(test_storms)]
    
    if len(tr_df) < 500 or len(te_df) < 100:
        print(f"   ⚠️ Not enough data (Train: {len(tr_df)}, Test: {len(te_df)}). Skipping.")
        continue

    tv_df = pd.concat([tr_df, va_df])
    sid_tv = tv_df[STORM_COL].values
    sid_te = te_df[STORM_COL].values

    # Local Scaling
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_tv = scaler_X.fit_transform(tv_df[FEATURES]).astype('float32')
    y_tv = scaler_y.fit_transform(tv_df[[TARGET]]).astype('float32')
    X_te = scaler_X.transform(te_df[FEATURES]).astype('float32')
    y_te = te_df[TARGET].values.astype('float32')

    X_tv_gpu = torch.tensor(X_tv, device=PRIMARY)
    X_te_gpu = torch.tensor(X_te, device=PRIMARY)
    
    Y_MEAN = torch.tensor(scaler_y.mean_, device=PRIMARY, dtype=torch.float32)
    Y_STD  = torch.tensor(scaler_y.scale_, device=PRIMARY, dtype=torch.float32)
    def descale(p): return p * Y_STD + Y_MEAN

    gage_metrics = {"deployment_id": gage, "train_rows": len(tr_df), "test_rows": len(te_df)}

    # 1. Local Log-Ridge
    alpha = float(bp_lr.get("alpha", 1e-3))
    shift = float(bp_lr.get("log_shift", 1e-3))
    if bp_lr.get("target_transform", "log") == "plain":
        model_lr = Ridge(alpha=alpha).fit(tv_df[FEATURES], tv_df[TARGET])
        preds_lr = model_lr.predict(te_df[FEATURES])
    else:
        model_lr = Ridge(alpha=alpha).fit(tv_df[FEATURES], np.log(tv_df[TARGET] + shift))
        preds_lr = np.exp(model_lr.predict(te_df[FEATURES])) - shift
    
    gage_metrics["LR_NSE"] = nse(y_te, preds_lr)
    gage_metrics["LR_KGE"] = kge(y_te, preds_lr)

    # 2. Local Res-ANN
    ann = wrap_model(SotaANN(len(FEATURES), int(bp_ann["hidden_size"]), int(bp_ann["n_layers"]), float(bp_ann["dropout"])))
    opt_ann = optim.AdamW(ann.parameters(), lr=float(bp_ann["lr"]), weight_decay=float(bp_ann.get("weight_decay", 0.0)))
    batch_ann = safe_batch_size(ann, X_tv_gpu, starting_batch=8192, min_batch=256)
    
    ann.train()
    for epoch in range(40): # Reduced epochs for local training speed
        perm = torch.randperm(len(X_tv_gpu), device=PRIMARY)
        for i in range(0, len(X_tv_gpu), batch_ann):
            idx = perm[i : i + batch_ann]
            train_step(ann, opt_ann, scaler_amp, X_tv_gpu[idx], torch.tensor(y_tv[idx.cpu()], device=PRIMARY), nn.MSELoss())
            
    ann.eval()
    with torch.no_grad():
        preds_ann = descale(ann(X_te_gpu)).cpu().numpy().flatten()
    gage_metrics["ANN_NSE"] = nse(y_te, preds_ann)
    gage_metrics["ANN_KGE"] = kge(y_te, preds_ann)
    del ann, opt_ann, X_tv_gpu, X_te_gpu; torch.cuda.empty_cache(); gc.collect()

    # 3. Local Attn-LSTM
    window = int(bp_lstm["window_size"])
    Xtv_w, ytv_w = build_storm_windows(X_tv, y_tv, sid_tv, window)
    Xte_w, yte_w = build_storm_windows(X_te, scaler_y.transform(te_df[[TARGET]]), sid_te, window)

    if len(Xtv_w) > 100 and len(Xte_w) > 50:
        lstm = wrap_model(SotaAttentionLSTM(len(FEATURES), int(bp_lstm["hidden_size"]), int(bp_lstm["n_layers"]), float(bp_lstm["dropout"])))
        opt_lstm = optim.AdamW(lstm.parameters(), lr=float(bp_lstm["lr"]), weight_decay=float(bp_lstm.get("weight_decay", 0.0)))
        
        Xtv_t = torch.tensor(Xtv_w, dtype=torch.float32)
        ytv_t = torch.tensor(ytv_w, dtype=torch.float32)
        batch_lstm = safe_batch_size(lstm, Xtv_t[:1].to(PRIMARY).expand(1024, -1, -1), starting_batch=1024, min_batch=64)

        lstm.train()
        for epoch in range(30):
            perm = torch.randperm(len(Xtv_t))
            for i in range(0, len(Xtv_t), batch_lstm):
                idx = perm[i : i + batch_lstm]
                train_step(lstm, opt_lstm, scaler_amp, Xtv_t[idx].to(PRIMARY), ytv_t[idx].to(PRIMARY), nn.MSELoss(), clip_grad=1.0)
        
        lstm.eval()
        with torch.no_grad():
            preds_lstm_s = torch.cat([lstm(torch.tensor(Xte_w[i : i + batch_lstm], device=PRIMARY)) for i in range(0, len(Xte_w), batch_lstm)])
            preds_lstm = descale(preds_lstm_s).cpu().numpy().flatten()
            obs_lstm = descale(torch.tensor(yte_w, device=PRIMARY)).cpu().numpy().flatten()
        
        gage_metrics["LSTM_NSE"] = nse(obs_lstm, preds_lstm)
        gage_metrics["LSTM_KGE"] = kge(obs_lstm, preds_lstm)
        del lstm, opt_lstm, Xtv_t, ytv_t; torch.cuda.empty_cache(); gc.collect()
    else:
        print("   ⚠️ Not enough windowed data for LSTM.")
        gage_metrics["LSTM_NSE"] = np.nan
        gage_metrics["LSTM_KGE"] = np.nan

    print(f"   ↳ Results: ANN NSE={gage_metrics.get('ANN_NSE', np.nan):.3f} | LSTM NSE={gage_metrics.get('LSTM_NSE', np.nan):.3f}")
    local_results.append(gage_metrics)

# ── Summary & Export ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(local_results)
out_csv = RESULTS_DIR / "local_models_summary.csv"
results_df.to_csv(out_csv, index=False)

print("\n" + "═"*60)
print(f"✅ Local Training Complete. Processed {len(results_df)} gages.")
print("═"*60)
print("Median Local Test Performance:")
print(f"  Log-Ridge : NSE = {results_df['LR_NSE'].median():.3f} | KGE = {results_df['LR_KGE'].median():.3f}")
print(f"  Res-ANN   : NSE = {results_df['ANN_NSE'].median():.3f} | KGE = {results_df['ANN_KGE'].median():.3f}")
print(f"  Attn-LSTM : NSE = {results_df['LSTM_NSE'].median():.3f} | KGE = {results_df['LSTM_KGE'].median():.3f}")
print(f"\n📊 Full local metrics saved to: {out_csv}")