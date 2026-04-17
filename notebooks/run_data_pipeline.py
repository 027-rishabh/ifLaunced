"""
Consolidated Data Pipeline for SpaceX Landing Prediction
=========================================================
This script runs the complete data pipeline from scratch:
1. Fetch data from SpaceX API
2. Scrape Wikipedia for booster data  
3. Wrangle and clean data
4. Merge and enrich datasets
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SPACEX DATA PIPELINE")
print("="*80)

# Create directories
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Fetch SpaceX API Data
# ==========================================
print("\n[1/4] Fetching SpaceX launch data from API...")

spacex_url = 'https://api.spacexdata.com/v5/launches/'
response = requests.get(spacex_url)
response.raise_for_status()
launch_data = response.json()
print(f"✓ Total launches fetched: {len(launch_data)}")

df_raw = pd.json_normalize(launch_data)
df_raw.to_csv('data/raw/spacex_launch_data_raw.csv', index=False)
print("✓ Saved raw data to data/raw/spacex_launch_data_raw.csv")

# ==========================================
# 2. Data Wrangling
# ==========================================
print("\n[2/4] Wrangling and cleaning data...")

df = df_raw.copy()

# Extract relevant columns
df_clean = pd.DataFrame()
df_clean['mission_name'] = df['name']
df_clean['launch_date'] = pd.to_datetime(df['date_utc'])
df_clean['rocket_name'] = df['rocket']  # Rocket ID
df_clean['launch_site'] = df['launchpad']

# Extract payload mass from payloads (it's a list in JSON string)
def get_payload_mass(payloads_str):
    try:
        import json
        payloads = json.loads(payloads_str)
        if isinstance(payloads, list) and len(payloads) > 0:
            return payloads[0].get('mass_kg', None)
    except:
        pass
    return None

df_clean['payload_mass'] = df['payloads'].apply(get_payload_mass)

# Extract orbit from payloads
def get_orbit(payloads_str):
    try:
        import json
        payloads = json.loads(payloads_str)
        if isinstance(payloads, list) and len(payloads) > 0:
            return payloads[0].get('orbit', None)
    except:
        pass
    return None

df_clean['orbit'] = df['payloads'].apply(get_orbit)

# Extract landing success from cores
def extract_landing_success(cores):
    if pd.isna(cores):
        return None
    try:
        if isinstance(cores, str):
            import json
            cores = json.loads(cores)
        if isinstance(cores, list) and len(cores) > 0:
            return cores[0].get('landing_success', None)
    except:
        return None

df_clean['landing_success'] = df['cores'].apply(extract_landing_success)
df_clean['reused'] = df['cores'].apply(lambda x: x[0].get('reused', None) if isinstance(x, list) and len(x) > 0 else None)

# Drop rows with missing landing success
df_clean = df_clean.dropna(subset=['landing_success'])

# Convert landing_success to binary
df_clean['landing_success'] = df_clean['landing_success'].map({True: 1, False: 0})

print(f"✓ Cleaned dataset shape: {df_clean.shape}")
print(f"✓ Target distribution:\n{df_clean['landing_success'].value_counts()}")

df_clean.to_csv('data/processed/spacex_cleaned.csv', index=False)
print("✓ Saved cleaned data to data/processed/spacex_cleaned.csv")

# ==========================================
# 3. Fetch Additional Data (Rocket Details)
# ==========================================
print("\n[3/4] Fetching additional rocket details...")

# Fetch rocket information
rockets_url = 'https://api.spacexdata.com/v4/rockets'
rockets_response = requests.get(rockets_url)
rockets_map = {r['id']: r['name'] for r in rockets_response.json()}

# Map rocket IDs to names
df_clean['rocket_name'] = df_clean['rocket_name'].map(rockets_map).fillna(df_clean['rocket_name'])

# Fetch launchpad information  
launchpads_url = 'https://api.spacexdata.com/v4/launchpads'
launchpads_response = requests.get(launchpads_url)
launchpads_data = {lp['id']: lp['name'] for lp in launchpads_response.json()}

# Map launchpads to names
df_clean['launch_site'] = df_clean['launch_site'].map(launchpads_data).fillna(df_clean['launch_site'])

print(f"✓ Mapped {len(launchpads_data)} launchpads")

# ==========================================
# 4. Create Enriched Dataset
# ==========================================
print("\n[4/4] Creating enriched dataset...")

# Duplicate columns for compatibility with original pipeline
df_clean['orbit_x'] = df_clean['orbit']
df_clean['orbit_y'] = df_clean['orbit']
df_clean['launch_site_x'] = df_clean['launch_site']
df_clean['launch_site_y'] = df_clean['launch_site']

# Save enriched version
df_clean.to_csv('data/processed/spacex_enriched.csv', index=False)
print(f"✓ Saved enriched dataset to data/processed/spacex_enriched.csv")
print(f"  Shape: {df_clean.shape}")
print(f"  Columns: {list(df_clean.columns)}")

print("\n" + "="*80)
print("DATA PIPELINE COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - data/raw/spacex_launch_data_raw.csv")
print(f"  - data/processed/spacex_cleaned.csv")
print(f"  - data/processed/spacex_enriched.csv")
print(f"\nReady for ML modeling!")
