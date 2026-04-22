#!/usr/bin/env python3
"""Select top precipitation-influenced, non-tidal FloodNet gages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


def calculate_richards_baker_index(flow_series: pd.Series) -> float:
    """Compute R-B flashiness index: sum(|Q_i - Q_{i-1}|) / sum(Q_i)."""
    s = pd.to_numeric(flow_series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan

    denom = float(s.sum())
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan

    numer = float(s.diff().abs().sum())
    return numer / denom


def verify_spectral_non_tidal_status(
    flow_series: pd.Series,
    interval_minutes: int = 15,
    min_samples: int = 2880,
    peak_to_noise_ratio: float = 8.0,
    band_half_width_cpd: float = 0.03,
) -> bool:
    """
    Return True if likely tidal (dominant M2/K1/O1 spectral peaks), else False.

    Frequencies are in cycles per day (cpd):
      M2 ~ 1.93, K1 ~ 1.00, O1 ~ 0.93
    """
    s = pd.to_numeric(flow_series, errors="coerce").dropna()
    if len(s) < min_samples:
        return False

    x = s.to_numpy(dtype=float)
    x = x - np.nanmean(x)
    if np.allclose(np.nanstd(x), 0.0):
        return False

    dt_days = interval_minutes / (60.0 * 24.0)
    freqs = np.fft.rfftfreq(len(x), d=dt_days)
    psd = np.abs(np.fft.rfft(x)) ** 2

    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]
    if len(psd) == 0:
        return False

    noise_floor = float(np.nanmedian(psd))
    if not np.isfinite(noise_floor) or np.isclose(noise_floor, 0.0):
        return False

    tidal_bands = [1.93, 1.00, 0.93]
    for target in tidal_bands:
        band = (freqs >= target - band_half_width_cpd) & (freqs <= target + band_half_width_cpd)
        if not np.any(band):
            continue
        band_peak = float(np.nanmax(psd[band]))
        if np.isfinite(band_peak) and (band_peak / noise_floor) >= peak_to_noise_ratio:
            return True

    return False


def compute_lag_and_cross_correlation(
    precip_series: pd.Series,
    flow_series: pd.Series,
    max_lag_hours: int = 72,
    interval_minutes: int = 15,
    min_overlap_points: int = 32,
) -> tuple[float, float, int]:
    """
    Compute lag and peak cross-correlation with pandas shift-corr loop.

    Returns:
      optimal_lag_hours, peak_correlation_raw, max_lag_periods
    """
    max_lag_periods = int(max_lag_hours * 60 / interval_minutes)

    p = pd.to_numeric(precip_series, errors="coerce")
    q = pd.to_numeric(flow_series, errors="coerce")

    vals = []
    for lag in range(0, max_lag_periods + 1):
        shifted = p.shift(lag)
        joined = pd.concat([q, shifted], axis=1).dropna()
        if len(joined) < min_overlap_points:
            vals.append(np.nan)
            continue
        # Skip undefined Pearson cases (zero variance in either vector).
        if np.isclose(joined.iloc[:, 0].std(), 0.0) or np.isclose(joined.iloc[:, 1].std(), 0.0):
            vals.append(np.nan)
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        vals.append(corr)

    corr_arr = np.array(vals, dtype=float)
    if np.all(np.isnan(corr_arr)):
        return np.nan, np.nan, max_lag_periods

    idx = int(np.nanargmax(np.abs(corr_arr)))
    optimal_lag_hours = idx * interval_minutes / 60.0
    peak_correlation = float(corr_arr[idx])
    return optimal_lag_hours, peak_correlation, max_lag_periods


def build_uniform_series(
    site_df: pd.DataFrame,
    time_col: str,
    precip_col: str,
    flow_col: str,
    interval_minutes: int,
) -> pd.DataFrame:
    """Build regular interval precip/flow series for one site."""
    freq = f"{interval_minutes}min"

    ts = (
        site_df[[time_col, precip_col, flow_col]]
        .dropna(subset=[time_col])
        .copy()
    )
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce", utc=True)
    ts = ts.dropna(subset=[time_col])
    if ts.empty:
        return pd.DataFrame(columns=[precip_col, flow_col])

    ts = ts.sort_values(time_col)
    ts = ts.groupby(time_col, as_index=True)[[precip_col, flow_col]].mean()
    ts = ts.resample(freq).mean()

    # Precipitation gaps are treated as 0 for cross-correlation continuity.
    ts[precip_col] = pd.to_numeric(ts[precip_col], errors="coerce").fillna(0.0)

    # Keep flow as observed; interpolate short gaps only.
    ts[flow_col] = pd.to_numeric(ts[flow_col], errors="coerce")
    ts[flow_col] = ts[flow_col].interpolate(limit=4)

    return ts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter weather-joined data to top precipitation-influenced, non-tidal gages."
    )
    parser.add_argument("--input", default="Data_Files/floodnet_full_dataset_merged_with_weather.parquet")
    parser.add_argument("--output", default="Data_Files/rain_influenced_sites_raw.parquet")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-lag-hours", type=int, default=72)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--min-overlap-points", type=int, default=32)
    parser.add_argument("--report-csv", default="Data_Files/selected_precip_gages_report.csv")
    parser.add_argument(
        "--time-col",
        default="time",
        help="Timestamp column to use for lag/correlation analysis.",
    )
    parser.add_argument(
        "--site-col",
        default="deployment_id",
        help="Site identifier column.",
    )
    parser.add_argument(
        "--precip-col",
        default="precip_1hr [inch]",
        help="Precipitation column used for lag/correlation analysis.",
    )
    parser.add_argument(
        "--flow-col",
        default="depth_inches",
        help="Flow proxy column used for flashiness/correlation analysis.",
    )
    parser.add_argument(
        "--skip-tidal-filter",
        action="store_true",
        help="Disable FFT-based tidal screening.",
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

    required_cols = {args.site_col, args.time_col, args.precip_col, args.flow_col}
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")

    rows = []
    for site_id, grp in df.groupby(args.site_col, sort=False):
        ts = build_uniform_series(
            grp,
            time_col=args.time_col,
            precip_col=args.precip_col,
            flow_col=args.flow_col,
            interval_minutes=args.interval_minutes,
        )

        n_samples = int(len(ts))
        if n_samples < max(args.min_overlap_points, 8):
            rows.append(
                {
                    "site_id": site_id,
                    "rb_index": np.nan,
                    "optimal_lag_time": np.nan,
                    "peak_correlation": np.nan,
                    "is_tidal": False,
                    "usable": False,
                    "n_samples": n_samples,
                    "reason": "too_few_samples",
                }
            )
            continue

        flow_series = ts[args.flow_col]
        precip_series = ts[args.precip_col]

        is_tidal = False
        if not args.skip_tidal_filter:
            is_tidal = verify_spectral_non_tidal_status(
                flow_series,
                interval_minutes=args.interval_minutes,
            )

        rb_index = calculate_richards_baker_index(flow_series)
        lag_hours, peak_corr, max_lag_periods = compute_lag_and_cross_correlation(
            precip_series=precip_series,
            flow_series=flow_series,
            max_lag_hours=args.max_lag_hours,
            interval_minutes=args.interval_minutes,
            min_overlap_points=args.min_overlap_points,
        )

        usable = np.isfinite(rb_index) and np.isfinite(lag_hours) and np.isfinite(peak_corr) and (not is_tidal)

        rows.append(
            {
                "site_id": site_id,
                "rb_index": rb_index,
                "optimal_lag_time": lag_hours,
                "peak_correlation": peak_corr,
                "is_tidal": bool(is_tidal),
                "usable": bool(usable),
                "n_samples": n_samples,
                "max_lag_periods": max_lag_periods,
                "reason": "ok" if usable else ("tidal" if is_tidal else "invalid_metrics"),
            }
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError("No site metrics were computed.")

    usable_df = results_df[results_df["usable"]].copy()
    if usable_df.empty:
        raise RuntimeError("No non-tidal usable sites after metric computation.")

    scaler = MinMaxScaler()
    usable_df[["normalized_rb", "normalized_corr"]] = scaler.fit_transform(
        usable_df[["rb_index", "peak_correlation"]]
    )

    inv_lag = args.max_lag_hours - usable_df["optimal_lag_time"]
    usable_df["normalized_inv_lag"] = MinMaxScaler().fit_transform(inv_lag.to_numpy().reshape(-1, 1))

    usable_df["composite_score"] = (
        0.35 * usable_df["normalized_rb"]
        + 0.35 * usable_df["normalized_corr"]
        + 0.30 * usable_df["normalized_inv_lag"]
    )

    ranked = usable_df.sort_values(["composite_score", "site_id"], ascending=[False, True]).reset_index(drop=True)
    selected = ranked.nlargest(min(args.top_n, len(ranked)), "composite_score").copy()
    selected = selected.sort_values(["composite_score", "site_id"], ascending=[False, True]).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)

    selected_ids = selected["site_id"].tolist()

    report = results_df.merge(
        ranked[
            [
                "site_id",
                "normalized_rb",
                "normalized_corr",
                "normalized_inv_lag",
                "composite_score",
            ]
        ],
        on="site_id",
        how="left",
    )
    report["selected"] = report["site_id"].isin(selected_ids)
    report = report.merge(selected[["site_id", "selected_rank"]], on="site_id", how="left")
    report["max_lag_hours"] = args.max_lag_hours
    report["interval_minutes"] = args.interval_minutes
    report["method"] = "rb_lag_corr_weighted_v2"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_df = df[df[args.site_col].isin(selected_ids)].copy()
    filtered_df.to_parquet(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("\nSelection complete")
    print(f"Sites evaluated      : {len(results_df)}")
    print(f"Usable non-tidal     : {len(ranked)}")
    print(f"Selected top-N       : {len(selected_ids)}")
    print(f"Selected IDs         : {selected_ids}")
    print(f"Wrote filtered parquet: {output_path}")
    print(f"Wrote selection report: {report_path}")


if __name__ == "__main__":
    main()
