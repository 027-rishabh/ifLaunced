# Data Wrangling - SpaceX Launch Data

#In this notebook, we:
#- Load raw data from the SpaceX API
#- Flatten nested JSON fields (rocket, payloads, launchpad)
#- Merge with booster information (from Wikipedia)
#- Clean and preprocess the dataset for analysis

import pandas as pd
import requests

# Load raw API data
df_raw = pd.read_csv('../data/raw/spacex_launch_data_raw.csv')

# Preview
df_raw.head(2)

df_raw[['rocket', 'payloads', 'launchpad', 'cores']].head(2)

# Get rocket details via rocket API
rocket_ids = df_raw['rocket'].unique()

rocket_data = {}
for rocket_id in rocket_ids:
    res = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}')
    rocket_data[rocket_id] = res.json()

# Create mapping
rocket_names = {k: v['name'] for k, v in rocket_data.items()}

# Map to main DataFrame
df_raw['rocket_name'] = df_raw['rocket'].map(rocket_names)

# Get payload info
payloads = {}
for pid_list in df_raw['payloads']:
    if isinstance(pid_list, list):  # sometimes it's a string
        for pid in pid_list:
            if pid not in payloads:
                res = requests.get(f'https://api.spacexdata.com/v4/payloads/{pid}')
                payloads[pid] = res.json()

# Create payload mapping
payload_mass_map = {k: v.get('mass_kg') for k, v in payloads.items()}
payload_type_map = {k: v.get('orbit') for k, v in payloads.items()}

# Get first payload (only one per flight in most cases)
df_raw['payload_mass'] = df_raw['payloads'].apply(lambda x: payload_mass_map.get(x[0]) if isinstance(x, list) else None)
df_raw['orbit'] = df_raw['payloads'].apply(lambda x: payload_type_map.get(x[0]) if isinstance(x, list) else None)

# Get launchpad info
launchpads = {}
for lid in df_raw['launchpad'].unique():
    res = requests.get(f'https://api.spacexdata.com/v4/launchpads/{lid}')
    launchpads[lid] = res.json()

launch_site_map = {k: v['name'] for k, v in launchpads.items()}
df_raw['launch_site'] = df_raw['launchpad'].map(launch_site_map)

df_raw['landing_success'] = df_raw['cores'].apply(lambda x: x[0]['landing_success'] if isinstance(x, list) and x[0] else None)
df_raw['reused'] = df_raw['cores'].apply(lambda x: x[0]['reused'] if isinstance(x, list) and x[0] else None)

df_final = df_raw[[
    'name', 'date_utc', 'rocket_name', 'payload_mass', 'orbit',
    'launch_site', 'landing_success', 'reused'
]]

df_final.rename(columns={
    'name': 'mission_name',
    'date_utc': 'launch_date'
}, inplace=True)

# Convert date
df_final['launch_date'] = pd.to_datetime(df_final['launch_date'])

df_final.info()
df_final.isna().sum()

df_final.to_csv('../data/processed/spacex_cleaned.csv', index=False)
print("Saved cleaned dataset to: ../data/processed/spacex_cleaned.csv")

