# -*- coding: utf-8 -*-
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------
# 1. Fetch Deployment Metadata
# ---------------------------------------------------------
def get_deployments():
    '''Retrieve a table of all sensor deployment locations.'''
    print("Fetching sensor deployment metadata...")
    response = requests.get("https://api.floodnet.nyc/api/rest/deployments/flood").json()
    if 'error' in response:
        raise RuntimeError(str(response))

    df = pd.DataFrame(response['deployments'])
    df = df.dropna(subset=['location'])

    # Convert date strings to datetime objects
    df['date_deployed'] = pd.to_datetime(df.date_deployed, format='ISO8601').dt.tz_localize(tz='America/New_York')
    df['date_down'] = pd.to_datetime(df.date_down, format='ISO8601').dt.tz_localize(tz='America/New_York')

    # Extract latitude, longitude pair for mapping
    df['coordinates'] = [np.array(x['coordinates'][::-1]) for x in df.location]
    df[['borough', 'intersection']] = df['name'].str.split(" - ", n=1, expand=True)
    df['latitude']  = df['coordinates'].apply(lambda x: x[0])
    df['longitude'] = df['coordinates'].apply(lambda x: x[1])
    
    return df

# ---------------------------------------------------------
# 2. Chunked API Query Logic (WITH RETRIES & MIXED DATETIME)
# ---------------------------------------------------------
def query_depth_data_chunked(deployment_id, start_time, end_time, chunk_days=2, sleep_sec=0):
    '''
    Queries the FloodNet API by breaking a large date range into smaller chunks.
    Includes a robust retry mechanism for dropped connections.
    '''
    current_start = pd.to_datetime(start_time, utc=True)
    final_end = pd.to_datetime(end_time, utc=True)
    all_chunks = []
    
    # Set up a robust session to handle connection drops/rate limits automatically
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    while current_start < final_end:
        current_end = current_start + timedelta(days=chunk_days)
        if current_end > final_end:
            current_end = final_end

        api_start = current_start.isoformat()
        api_end = current_end.isoformat()

        print(f"    Fetching {api_start[:10]} to {api_end[:10]}...")

        try:
            response = session.get(
                f"https://api.floodnet.nyc/api/rest/deployments/flood/{deployment_id}/depth", 
                params={'start_time': api_start, 'end_time': api_end},
                timeout=30 
            )
            data = response.json()
            
            if 'error' in data:
                print(f"      API Error: {data['error']}")
            elif 'depth_data' in data and len(data['depth_data']) > 0:
                df_chunk = pd.DataFrame(data['depth_data'], columns=['deployment_id', 'time', 'depth_proc_mm'])
                all_chunks.append(df_chunk)
            else:
                pass 
                
        except requests.exceptions.RequestException as e:
            print(f"      CRITICAL NETWORK ERROR skipping chunk: {e}")
            
        time.sleep(sleep_sec)
        current_start = current_end

    if not all_chunks:
        return pd.DataFrame()

    df_depth = pd.concat(all_chunks, ignore_index=True)
    df_depth = df_depth.dropna(subset=['depth_proc_mm'])
    
    # Using format='mixed' to handle the API's changing timestamp formats safely
    df_depth['time'] = pd.to_datetime(df_depth['time'], format='mixed', utc=True)
    df_depth['depth_inches'] = df_depth['depth_proc_mm'] / 25.4
    
    return df_depth

# ---------------------------------------------------------
# 3. Process, Save, and Checkpoint Logic
# ---------------------------------------------------------
def process_and_save_sensors(df_sensors, start_time, end_time, output_folder=None):
    '''
    Loops through sensors, skipping any that already have a saved Parquet file.
    '''
    if output_folder is None:
        output_folder = PROJECT_ROOT / "floodnet_parquet_data"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
        
    # Checkpoint: Identify sensors already downloaded
    completed_ids = [p.stem for p in output_folder.glob("*.parquet")]
    df_to_do = df_sensors[~df_sensors['deployment_id'].isin(completed_ids)]
    
    print(f"\nCheckpoint Status: {len(completed_ids)} sensors found. {len(df_to_do)} remaining to pull.")
    
    if len(df_to_do) == 0:
        print("All sensors are up to date!")
        return

    for index, (row_idx, row_data) in enumerate(df_to_do.iterrows()):
        deployment_id = row_data['deployment_id']
        print(f"\n[{index + 1}/{len(df_to_do)}] Processing {deployment_id}...")
        
        df_sensor = query_depth_data_chunked(
            deployment_id=deployment_id, 
            start_time=start_time, 
            end_time=end_time
        )
        
        if not df_sensor.empty:
            df_meta = pd.DataFrame([row_data])
            df_final = pd.merge(df_sensor, df_meta, on='deployment_id', how='left')
            
            file_path = output_folder / f"{deployment_id}.parquet"
            df_final.to_parquet(file_path, index=False)
            print(f"  -> Saved {len(df_final)} rows.")
            
        # Explicit Memory Clearing
        del df_sensor
        if 'df_final' in locals(): del df_final
        gc.collect() 

# ---------------------------------------------------------
# 4. Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get Metadata
    df_deployments = get_deployments()
    
    # Parameters
    START_TIME = "2021-01-01T00:00:00"
    END_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Filter for all "good" sensors (No coastal-only filter)
    df_target_sensors = df_deployments[df_deployments['sensor_status'] == 'good']

    print(f"\nScanning for all active sensors from {START_TIME} to {END_TIME}...")
    
    # Run Extraction
    process_and_save_sensors(
        df_sensors=df_target_sensors,
        start_time=START_TIME,
        end_time=END_TIME,
        output_folder=PROJECT_ROOT / "floodnet_parquet_data"
    )
    
    print("\nProcess finished successfully.")
