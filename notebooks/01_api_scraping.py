# SpaceX Data Collection - API Access

#In this notebook, we fetch historical launch data from the SpaceX REST API and store it for further analysis.

import requests
import pandas as pd
import json

# SpaceX API URL
spacex_url = 'https://api.spacexdata.com/v5/launches/'

# Fetch data
response = requests.get(spacex_url)
response.raise_for_status()

# Load JSON response
launch_data = response.json()
print(f"Total launches fetched: {len(launch_data)}")

# Convert JSON to DataFrame
df = pd.json_normalize(launch_data)

# Display columns
df.columns.tolist()

# Save raw data
df.to_csv('../data/raw/spacex_launch_data_raw.csv', index=False)
print("Data saved to: ../data/raw/spacex_launch_data_raw.csv")


