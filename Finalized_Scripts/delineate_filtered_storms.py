#!/usr/bin/env python3
"""Delineate storms on pre-filtered rain-influenced gage records."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delineate buffered/merged storms for filtered rain-influenced sites."
    )
    parser.add_argument("--input", default="Data_Files/rain_influenced_sites_raw.parquet")
    parser.add_argument("--output", default="Data_Files/rain_influenced_gages.parquet")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="180GB")
    parser.add_argument("--temp-directory", default="./tmp_duckdb")

    # Storm delineation parameters
    parser.add_argument("--mit-seconds", type=int, default=21600)
    parser.add_argument("--lead-time-hours", type=int, default=2)
    parser.add_argument("--lag-time-hours", type=int, default=6)
    parser.add_argument("--min-wet-threshold", type=float, default=0.01)
    parser.add_argument("--intensity-threshold-inh", type=float, default=2.0)
    parser.add_argument("--min-intensity-hits", type=int, default=2)
    parser.add_argument(
        "--weather-time-col",
        default='"DATE"',
        help="Quoted weather timestamp column name for age diagnostics.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()
    input_path = resolve_path(project_root, args.input)
    output_path = resolve_path(project_root, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads TO {int(args.threads)};")
    con.execute(f"SET memory_limit = '{args.memory_limit}';")
    con.execute(f"SET temp_directory = '{args.temp_directory}';")

    weather_col = args.weather_time_col
    if weather_col.strip().lower() == "none":
        weather_col = None

    print("🚀 Starting Storm Delineation (Merged Windows + Phase Labels)...")
    print(f"   Input : {input_path}")
    print(f"   Output: {output_path}")

    if weather_col is None:
        weather_age_expr = "NULL::DOUBLE AS minutes_since_weather_update"
    else:
        weather_age_expr = f"""
            CASE
              WHEN {weather_col} IS NULL THEN NULL::DOUBLE
              ELSE (epoch(d.time) - epoch(d.{weather_col})) / 60.0
            END AS minutes_since_weather_update
        """

    query = f"""
    COPY (
        WITH wet_records AS (
            SELECT
                deployment_id,
                time,
                LAG(time) OVER (PARTITION BY deployment_id ORDER BY time) AS prev_wet_time
            FROM '{str(input_path)}'
            WHERE "precip_incremental [inch]" >= {args.min_wet_threshold}
        ),

        wet_storm_ids AS (
            SELECT
                deployment_id,
                time,
                SUM(
                    CASE
                        WHEN prev_wet_time IS NULL THEN 1
                        WHEN (epoch(time) - epoch(prev_wet_time)) > {args.mit_seconds} THEN 1
                        ELSE 0
                    END
                ) OVER (
                    PARTITION BY deployment_id
                    ORDER BY time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS storm_id_raw
            FROM wet_records
        ),

        raw_windows AS (
            SELECT
                deployment_id,
                storm_id_raw,
                MIN(time) AS rain_start,
                MAX(time) AS rain_end,
                MIN(time) - INTERVAL '{args.lead_time_hours} hours' AS storm_start,
                MAX(time) + INTERVAL '{args.lag_time_hours} hours' AS storm_end
            FROM wet_storm_ids
            GROUP BY deployment_id, storm_id_raw
        ),

        ordered AS (
            SELECT
                *,
                LAG(storm_end) OVER (PARTITION BY deployment_id ORDER BY storm_start) AS prev_end
            FROM raw_windows
        ),

        flags AS (
            SELECT
                *,
                CASE
                    WHEN prev_end IS NULL THEN 1
                    WHEN storm_start > prev_end THEN 1
                    ELSE 0
                END AS new_group_flag
            FROM ordered
        ),

        merged_ids AS (
            SELECT
                *,
                SUM(new_group_flag) OVER (PARTITION BY deployment_id ORDER BY storm_start) AS storm_id
            FROM flags
        ),

        final_windows AS (
            SELECT
                deployment_id,
                storm_id,
                MIN(storm_start) AS storm_start,
                MAX(storm_end) AS storm_end,
                MIN(rain_start) AS rain_start,
                MAX(rain_end) AS rain_end
            FROM merged_ids
            GROUP BY deployment_id, storm_id
        ),

        all_records_in_storms AS (
            SELECT
                d.*,
                sw.storm_id,
                d.deployment_id || '_' || sw.storm_id AS global_storm_id,
                sw.storm_start,
                sw.storm_end,
                sw.rain_start,
                sw.rain_end,
                CASE
                    WHEN d.time < sw.rain_start THEN 'lead'
                    WHEN d.time <= sw.rain_end THEN 'storm'
                    ELSE 'lag'
                END AS phase,
                (epoch(d.time) - epoch(sw.rain_start)) / 3600.0 AS hours_since_rain_start,
                (epoch(d.time) - epoch(sw.storm_start)) / 3600.0 AS hours_since_storm_start,
                {weather_age_expr}
            FROM '{str(input_path)}' d
            INNER JOIN final_windows sw
                ON d.deployment_id = sw.deployment_id
                AND d.time BETWEEN sw.storm_start AND sw.storm_end
        ),

        storm_metrics AS (
            SELECT
                global_storm_id,
                deployment_id,
                storm_id,
                SUM("precip_incremental [inch]") AS total_precip_in,
                MAX("precip_max_intensity [inch/hour]") AS peak_intensity_inh,
                COUNT(DISTINCT CASE
                    WHEN "precip_max_intensity [inch/hour]" >= {args.intensity_threshold_inh}
                    THEN weather_time
                    ELSE NULL
                END) AS intensity_hits_ge_threshold,
                MAX(depth_inches) - MIN(depth_inches) AS net_depth_rise_in,
                COUNT(*) AS storm_record_count,
                (epoch(MAX(storm_end)) - epoch(MIN(storm_start))) / 3600.0 AS storm_duration_hr
            FROM all_records_in_storms
            GROUP BY global_storm_id, deployment_id, storm_id
        ),

        significant_storms AS (
    SELECT *,
        CASE 
            WHEN peak_intensity_inh >= 0.5 THEN 'Extreme'
            WHEN total_precip_in >= 1.0 THEN 'Heavy'
            ELSE 'Moderate'
        END as storm_severity
    FROM storm_metrics
    WHERE (total_precip_in >= 0.5)
       OR (peak_intensity_inh >= 0.25)
       OR (net_depth_rise_in >= 1.5)
)

        SELECT
            a.*,
            s.total_precip_in,
            s.peak_intensity_inh,
            s.intensity_hits_ge_threshold,
            s.net_depth_rise_in,
            s.storm_record_count,
            s.storm_duration_hr
        FROM all_records_in_storms a
        INNER JOIN significant_storms s
            ON a.global_storm_id = s.global_storm_id
        ORDER BY a.deployment_id, a.time

    ) TO '{str(output_path)}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """

    con.execute(query)
    print(f"✅ Delineation complete → {output_path}")

    # ── Pandas Post-Processing: Impute Missing Data & Convert Rates ──────────
    print("\n🩹 Applying interpolation and converting hour rates to 5-min volumes...")
    import pandas as pd
    
    df = pd.read_parquet(output_path)
    
    weather_cols = ['precip_1hr [inch]', 'precip_max_intensity [inch/hour]', 'temp_2m [degF]']
    df[weather_cols] = df[weather_cols].interpolate(method='linear', limit=12)
    
    soil_col = 'soil_moisture_05cm [m^3/m^3]'
    if soil_col in df.columns:
        df[soil_col] = df[soil_col].ffill(limit=288)

    # Convert hour rates to 5-minute interval volumes for the ML models
    for col in ['precip_1hr [inch]', 'precip_max_intensity [inch/hour]']:
        if col in df.columns:
            df[col] = df[col] / 12.0
        
    df.to_parquet(output_path, compression='zstd')
    print(f"✅ Data processed and saved back to {output_path}")
    print("   Ready for hpo_search.py and training.py!")


if __name__ == "__main__":
    main()