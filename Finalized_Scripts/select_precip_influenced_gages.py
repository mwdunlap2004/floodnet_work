#!/usr/bin/env python3
"""Select top precipitation-influenced, non-tidal FloodNet gages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_project_root() -> Path:
    try:
        current_location = Path(__file__).resolve().parent
    except NameError:
        current_location = Path.cwd().resolve()

    if current_location.name in ["Finalized_Scripts", "Test_Scripts", "scripts"]:
        return current_location.parent
    return current_location


def resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return project_root / p


def robust_scale_01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    lo = float(np.nanpercentile(valid, 5))
    hi = float(np.nanpercentile(valid, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=series.index)

    scaled = ((values - lo) / (hi - lo)).clip(0, 1)
    return scaled


def calc_wet_metrics(group: pd.DataFrame, rise_threshold_in: float) -> pd.Series:
    n_wet = len(group)
    corr = np.nan
    slope = np.nan

    if n_wet >= 2:
        x = group["total_precip_in"].to_numpy(dtype=float)
        y = group["net_depth_rise_in"].to_numpy(dtype=float)
        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0])

    wet_positive_frac = float((group["net_depth_rise_in"] > rise_threshold_in).mean()) if n_wet else np.nan

    return pd.Series(
        {
            "n_wet": int(n_wet),
            "corr_precip_depth": corr,
            "slope_in_per_in": slope,
            "wet_positive_frac": wet_positive_frac,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter delineated storms to top precipitation-influenced, non-tidal gages."
    )
    parser.add_argument("--input", default="Data_Files/delineated_storms.parquet")
    parser.add_argument("--output", default="Data_Files/rain_influenced_gages.parquet")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-wet-storms", type=int, default=15)
    parser.add_argument("--wet-threshold-in", type=float, default=0.05)
    parser.add_argument("--rise-threshold-in", type=float, default=0.02)
    parser.add_argument("--dry-precip-threshold-in", type=float, default=0.001)
    parser.add_argument("--report-csv", default="Data_Files/selected_precip_gages_report.csv")
    parser.add_argument(
        "--min-dry-samples",
        type=int,
        default=100,
        help="Minimum near-dry samples needed before applying tidal exclusion.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    project_root = resolve_project_root()
    input_path = resolve_path(project_root, args.input)
    output_path = resolve_path(project_root, args.output)
    report_path = resolve_path(project_root, args.report_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    print(f"Loading source data: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    required_cols = {
        "deployment_id",
        "total_precip_in",
        "net_depth_rise_in",
        "depth_inches",
        "precip_1hr [inch]",
    }
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")

    storm_col = None
    for candidate in ["global_storm_id", "storm_id"]:
        if candidate in df.columns:
            storm_col = candidate
            break
    if storm_col is None:
        raise KeyError("Missing storm identifier columns: expected one of ['global_storm_id', 'storm_id']")

    # Storm-level aggregation for precipitation-response behavior.
    storm_level = (
        df[["deployment_id", storm_col, "total_precip_in", "net_depth_rise_in"]]
        .dropna(subset=["deployment_id", storm_col, "total_precip_in", "net_depth_rise_in"])
        .drop_duplicates(["deployment_id", storm_col])
    )

    wet_storms = storm_level[storm_level["total_precip_in"] >= args.wet_threshold_in].copy()

    wet_rows = []
    for deployment_id, grp in wet_storms.groupby("deployment_id", sort=False):
        row = calc_wet_metrics(grp, args.rise_threshold_in).to_dict()
        row["deployment_id"] = deployment_id
        wet_rows.append(row)
    wet_metrics = pd.DataFrame(wet_rows)
    if wet_metrics.empty:
        raise RuntimeError("No wet storms found after filtering. Lower --wet-threshold-in.")

    dry_subset = df[df["precip_1hr [inch]"] <= args.dry_precip_threshold_in][["deployment_id", "depth_inches"]].copy()
    dry_metrics = (
        dry_subset.groupby("deployment_id", as_index=False)
        .agg(dry_depth_std=("depth_inches", "std"), dry_n=("depth_inches", "size"))
    )

    metrics = wet_metrics.merge(dry_metrics, on="deployment_id", how="left")

    metrics["flag_min_wet"] = metrics["n_wet"] >= args.min_wet_storms
    metrics["flag_corr"] = metrics["corr_precip_depth"] >= 0.25
    metrics["flag_wet_positive"] = metrics["wet_positive_frac"] >= 0.15
    metrics["flag_dry_samples"] = metrics["dry_n"].fillna(0) >= args.min_dry_samples

    tidal_pool = metrics.loc[metrics["flag_dry_samples"] & metrics["dry_depth_std"].notna(), "dry_depth_std"]
    dry_std_cutoff = float(tidal_pool.quantile(0.85)) if not tidal_pool.empty else np.nan

    metrics["flag_tidal_excluded"] = False
    if np.isfinite(dry_std_cutoff):
        metrics.loc[
            metrics["flag_dry_samples"] & (metrics["dry_depth_std"] > dry_std_cutoff),
            "flag_tidal_excluded",
        ] = True

    metrics["corr_z"] = robust_scale_01(metrics["corr_precip_depth"])
    metrics["slope_z"] = robust_scale_01(metrics["slope_in_per_in"])
    metrics["wet_positive_z"] = robust_scale_01(metrics["wet_positive_frac"])
    metrics["dry_std_penalty"] = robust_scale_01(metrics["dry_depth_std"])

    penalty_fill = float(metrics["dry_std_penalty"].median(skipna=True))
    if not np.isfinite(penalty_fill):
        penalty_fill = 0.0

    metrics["score"] = (
        0.45 * metrics["corr_z"].fillna(0)
        + 0.35 * metrics["slope_z"].fillna(0)
        + 0.20 * metrics["wet_positive_z"].fillna(0)
        - 0.25 * metrics["dry_std_penalty"].fillna(penalty_fill)
    )

    metrics["eligible"] = (
        metrics["flag_min_wet"]
        & metrics["flag_corr"]
        & metrics["flag_wet_positive"]
        & (~metrics["flag_tidal_excluded"])
    )

    ranked = metrics.sort_values(["score", "deployment_id"], ascending=[False, True]).reset_index(drop=True)
    eligible_ranked = ranked[ranked["eligible"]].copy()

    if len(eligible_ranked) < args.top_n:
        raise RuntimeError(
            f"Only {len(eligible_ranked)} eligible gages found, but --top-n={args.top_n}. "
            "Relax thresholds or reduce --top-n."
        )

    selected = eligible_ranked.head(args.top_n).copy()
    selected = selected.sort_values(["score", "deployment_id"], ascending=[False, True]).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)

    selected_ids = selected["deployment_id"].tolist()

    report = ranked.copy()
    report["selected"] = report["deployment_id"].isin(selected_ids)
    report = report.merge(
        selected[["deployment_id", "selected_rank"]],
        on="deployment_id",
        how="left",
    )
    report["storm_id_column_used"] = storm_col
    report["dry_std_cutoff_85pct"] = dry_std_cutoff

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_df = df[df["deployment_id"].isin(selected_ids)].copy()

    filtered_df.to_parquet(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("\nSelection complete")
    print(f"Storm ID column used: {storm_col}")
    print(f"Dry std cutoff (85th pct among dry-sample-eligible gages): {dry_std_cutoff}")
    print(f"Selected deployment IDs ({len(selected_ids)}):")
    for i, did in enumerate(selected_ids, start=1):
        print(f"  {i}. {did}")
    print(f"\nWrote filtered parquet: {output_path}")
    print(f"Wrote selection report: {report_path}")


if __name__ == "__main__":
    main()
