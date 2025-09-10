import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import re

# -----------------------------
# 1. Load Data
# -----------------------------
api_df = pd.read_csv("../data/processed/spacex_cleaned.csv")
wiki_df = pd.read_csv("../data/raw/wiki_booster_data.csv")

# -----------------------------
# 2. Parse Dates
# -----------------------------
# API: UTC ISO format
api_df['launch_date'] = pd.to_datetime(api_df['launch_date'], utc=True, errors='coerce')
api_df['launch_date'] = api_df['launch_date'].dt.tz_localize(None)

# Wiki: Parse flexible text like "4 June 2010 18:45"
def clean_wiki_date(d):
    if pd.isna(d):
        return None
    d = re.sub(r"\[.*?\]", "", str(d))
    d = re.sub(r"(\d{4})(\d{2}):", r"\1 \2:", d)  # fix year+hour stuck
    return pd.to_datetime(d.strip(), errors='coerce', dayfirst=True)

wiki_df['date'] = wiki_df['date'].apply(clean_wiki_date)
wiki_df = wiki_df.dropna(subset=['date'])

# -----------------------------
# 3. Print Sample & Range Check
# -----------------------------
print("\nSample parsed dates (wiki):")
print(wiki_df['date'].dropna().head())

print("\nSample parsed dates (api):")
print(api_df['launch_date'].dropna().head())

print("\nAPI date range:", api_df['launch_date'].min(), "to", api_df['launch_date'].max())
print("Wiki date range:", wiki_df['date'].min(), "to", wiki_df['date'].max())

# -----------------------------
# 4. Sort and Merge (±30 days)
# -----------------------------
api_df = api_df.sort_values('launch_date')
wiki_df = wiki_df.sort_values('date')

merged = pd.merge_asof(
    api_df,
    wiki_df,
    left_on='launch_date',
    right_on='date',
    tolerance=pd.Timedelta(days=30),  # relaxed window
    direction='nearest'
)

# Drop rows without booster match (optional)
merged = merged.dropna(subset=['booster_version'])

# Save merged result
merged.to_csv("../data/processed/spacex_enriched.csv", index=False)
print(f"\nMerged rows: {merged.shape[0]}")
print("Saved to ../data/processed/spacex_enriched.csv")

# -----------------------------
# 5. EDA if Merge Worked
# -----------------------------
if not merged.empty:

    # Handle orbit column naming
    if 'orbit_x' in merged.columns:
        merged['orbit'] = merged['orbit_x']
    elif 'orbit_y' in merged.columns:
        merged['orbit'] = merged['orbit_y']

    merged['year'] = pd.to_datetime(merged['launch_date']).dt.year

    # Payload Mass vs Landing Success
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=merged, x='payload_mass', y='landing_success', hue='rocket_name')
    plt.title("Payload Mass vs. Landing Success")
    plt.xlabel("Payload Mass (kg)")
    plt.ylabel("Landing Success")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Orbit vs. Landing Success
    plt.figure(figsize=(12, 5))
    sns.countplot(data=merged, x='orbit', hue='landing_success')
    plt.title("Orbit vs. Landing Success")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Yearly Trend
    trend = merged.groupby('year')['landing_success'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=trend, x='year', y='landing_success', marker='o')
    plt.title("Landing Success Rate Over Years")
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Booster Version vs. Success
    plt.figure(figsize=(12, 6))
    sns.barplot(data=merged, x='booster_version', y='landing_success')
    plt.title("Booster Version vs. Success Rate")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo merged rows found — adjust date parsing or verify date ranges.")

