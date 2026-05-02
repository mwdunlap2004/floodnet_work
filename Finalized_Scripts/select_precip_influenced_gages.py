#!/usr/bin/env python3
"""
Select top precipitation-influenced, non-tidal FloodNet gages.

METHODOLOGY V3: EVENT-BASED CONTINGENCY
- Replaces global correlation with Conditional Probability (Hit Rate & Reliability).
- Applies Rolling Median despiking to enforce physical hydrograph shapes.
- Correlates precipitation and depth ONLY during active wet/flood events.
"""

from __future__ import annotations

import argparse
import os
import traceback
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def resolve_project_root() -> Path:
    """Find the project root directory to locate data files consistently."""
    try:
        current_location = Path(__file__).resolve().parent
    except NameError:
        current_location = Path.cwd().resolve()

    if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"]:
        return current_location.parent
    return current_location


def resolve_path(project_root: Path, raw: str) -> Path:
    """Resolve a relative or absolute path against the project root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return project_root / p


def qid(name: str) -> str:
    """Quote an SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def build_uniform_series(
    site_df: pd.DataFrame,
    time_col: str,
    precip_col: str,
    flow_col: str,
    interval_minutes: int,
    max_valid_depth: float,
) -> pd.DataFrame:
    """Build regular interval series and strictly enforce physical data rules."""
    freq = f"{interval_minutes}min"

    ts = site_df[[time_col, precip_col, flow_col]].dropna(subset=[time_col]).copy()
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce", utc=True)
    ts = ts.dropna(subset=[time_col])
    if ts.empty:
        return pd.DataFrame()

    ts = ts.sort_values(time_col)
    ts = ts.groupby(time_col, as_index=True)[[precip_col, flow_col]].mean()
    ts = ts.resample(freq).mean()

    ts[precip_col] = pd.to_numeric(ts[precip_col], errors="coerce").fillna(0.0)
    ts[flow_col] = pd.to_numeric(ts[flow_col], errors="coerce")
    
    # 1. Cap impossible depths
    ts.loc[ts[flow_col] > max_valid_depth, flow_col] = np.nan
    
    # 2. Interpolate small gaps (up to 1 hour)
    ts[flow_col] = ts[flow_col].interpolate(limit=4)
    
    # 3. ROLLING MEDIAN DESPIKING
    # Window of 3 requires a peak to be surrounded by elevated water to survive.
    # Single 15-minute spikes from trucks/debris are completely erased.
    ts[flow_col] = ts[flow_col].rolling(window=3, center=True, min_periods=1).median()
    
    return ts


def evaluate_contingency_and_correlation(
    ts: pd.DataFrame, 
    precip_col: str, 
    flow_col: str,
    rain_threshold_inch: float = 0.25,
    flood_threshold_inch: float = 0.5
) -> dict:
    """Grade the sensor based on conditional probabilities and wet-only correlation."""
    
    # Create rolling 3-hour precipitation sum to capture storm windows
    ts['rain_3hr'] = ts[precip_col].rolling(12, min_periods=1).sum()
    
    # Define binary states
    ts['is_raining'] = ts['rain_3hr'] >= rain_threshold_inch
    ts['is_flooding'] = ts[flow_col] >= flood_threshold_inch
    
    # Contingency Table
    # True Positive: Raining and Flooding
    tp = (ts['is_raining'] & ts['is_flooding']).sum()
    # False Positive: Flooding on a sunny day (Glitch / Bad drainage)
    fp = (~ts['is_raining'] & ts['is_flooding']).sum()
    # False Negative: Raining but NOT flooding (Dry sensor)
    fn = (ts['is_raining'] & ~ts['is_flooding']).sum()
    
    # Hit Rate: Out of all rainstorms, how many caused floods?
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # False Alarm Ratio: Out of all floods, how many were fake/sunny?
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    reliability = 1.0 - far # Higher is better
    
    # Wet-Only Correlation (Ignore the 90% of the year it is sunny and dry)
    wet_mask = ts['is_raining'] | ts['is_flooding']
    wet_ts = ts[wet_mask]
    
    if len(wet_ts) > 10:
        # Catch flatlines BEFORE calculating correlation to prevent ConstantInputWarning
        if wet_ts['rain_3hr'].max() == wet_ts['rain_3hr'].min() or wet_ts[flow_col].max() == wet_ts[flow_col].min():
            wet_corr = 0.0
        else:
            wet_corr = wet_ts['rain_3hr'].corr(wet_ts[flow_col], method='spearman')
            # If correlation is negative, cap at 0
            wet_corr = max(0.0, wet_corr) if not np.isnan(wet_corr) else 0.0
    else:
        wet_corr = 0.0

    return {
        "hit_rate": hit_rate,
        "reliability": reliability,
        "wet_correlation": wet_corr,
        "total_floods": tp + fp,
        "total_storms": tp + fn
    }


def process_site(site_id: str, args: argparse.Namespace) -> dict:
    """Worker function to analyze a single site."""
    try:
        project_root = resolve_project_root()
        input_path = resolve_path(project_root, args.input)
        in_sql = str(input_path).replace("'", "''")

        con = duckdb.connect()
        fetch_query = f"""
            SELECT
                {qid(args.time_col)} AS {qid(args.time_col)},
                {qid(args.precip_col)} AS {qid(args.precip_col)},
                {qid(args.flow_col)} AS {qid(args.flow_col)}
            FROM read_parquet('{in_sql}')
            WHERE {qid(args.site_col)} = ?
        """
        grp = con.execute(fetch_query, [site_id]).df()
        con.close()

        ts = build_uniform_series(
            grp,
            time_col=args.time_col,
            precip_col=args.precip_col,
            flow_col=args.flow_col,
            interval_minutes=args.interval_minutes,
            max_valid_depth=args.max_valid_depth,
        )

        n_samples = int(len(ts)) if not ts.empty else 0
        if n_samples < 100:
            return {"site_id": site_id, "usable": False, "reason": "too_few_samples", "composite_score": 0.0}

        metrics = evaluate_contingency_and_correlation(
            ts, 
            precip_col=args.precip_col, 
            flow_col=args.flow_col,
            rain_threshold_inch=args.rain_threshold,
            flood_threshold_inch=args.flood_threshold
        )

        # Ensure the sensor experienced at least ONE storm
        if metrics["total_storms"] == 0:
            return {"site_id": site_id, "usable": False, "reason": "no_storms_recorded", "composite_score": 0.0}

        # Weight the metrics: 
        # 40% Hit Rate (Does it react?)
        # 40% Reliability (Does it lie on sunny days?)
        # 20% Correlation (Does depth scale with rain intensity?)
        composite = (
            (0.40 * metrics["hit_rate"]) + 
            (0.40 * metrics["reliability"]) + 
            (0.20 * metrics["wet_correlation"])
        )

        usable = composite >= args.min_composite_score

        return {
            "site_id": site_id,
            "usable": usable,
            "reason": "ok" if usable else "failed_metrics",
            "hit_rate": metrics["hit_rate"],
            "reliability": metrics["reliability"],
            "wet_correlation": metrics["wet_correlation"],
            "composite_score": composite,
            "n_samples": n_samples,
        }
    except Exception:
        print(f"ERROR: Failed to process site {site_id}.")
        return {"site_id": site_id, "usable": False, "reason": "error", "composite_score": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Data_Files/floodnet_full_dataset_merged_with_weather.parquet")
    parser.add_argument("--output", default="Data_Files/rain_influenced_sites_raw.parquet")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-composite-score", type=float, default=0.3)
    parser.add_argument("--rain-threshold", type=float, default=0.25, help="3-hr rain total to trigger a 'storm' condition.")
    parser.add_argument("--flood-threshold", type=float, default=0.5, help="Depth in inches to trigger a 'flood' condition.")
    parser.add_argument("--max-valid-depth", type=float, default=120.0)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--report-csv", default="Data_Files/selected_precip_gages_report.csv")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--site-col", default="deployment_id")
    parser.add_argument("--precip-col", default="precip_1hr [inch]")
    parser.add_argument("--flow-col", default="depth_inches")
    parser.add_argument("--duckdb-threads", type=int, default=4)
    args = parser.parse_args()

    project_root = resolve_project_root()
    input_path = resolve_path(project_root, args.input)
    output_path = resolve_path(project_root, args.output)
    report_path = resolve_path(project_root, args.report_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading source data: {input_path}")
    con = duckdb.connect()
    in_sql = str(input_path).replace("'", "''")
    site_ids = [row[0] for row in con.execute(f"SELECT DISTINCT {qid(args.site_col)} FROM read_parquet('{in_sql}') WHERE {qid(args.site_col)} IS NOT NULL").fetchall()]
    con.close()

    print(f"Evaluating {len(site_ids)} candidate sites using Contingency Metrics...")
    worker_func = partial(process_site, args=args)
    
    with Pool(processes=min(args.duckdb_threads, max(1, os.cpu_count() // 2))) as pool:
        rows = list(pool.map(worker_func, site_ids))

    results_df = pd.DataFrame(rows)
    usable_df = results_df[results_df["usable"]].copy()
    
    if usable_df.empty:
        print("WARNING: No sites passed the contingency evaluation.")
        return

    ranked = usable_df.sort_values(["composite_score", "site_id"], ascending=[False, True]).reset_index(drop=True)
    selected = ranked.head(args.top_n).copy()
    selected_ids = selected["site_id"].tolist()

    report = results_df.copy()
    report["selected"] = report["site_id"].isin(selected_ids)
    report.to_csv(report_path, index=False)

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE selected_sites(site_id VARCHAR)")
    con.executemany("INSERT INTO selected_sites VALUES (?)", [(str(s),) for s in selected_ids])
    
    out_sql = str(output_path).replace("'", "''")
    con.execute(f"COPY (SELECT p.* FROM read_parquet('{in_sql}') p INNER JOIN selected_sites s ON p.{qid(args.site_col)} = s.site_id) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION 'ZSTD')")
    con.close()

    print("\nSelection complete")
    print(f"Selected IDs: {selected_ids}")

if __name__ == "__main__":
    main()