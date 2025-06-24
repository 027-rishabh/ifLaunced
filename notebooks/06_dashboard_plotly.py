import pandas as pd
import plotly.express as px
import plotly.io as pio

# Use system browser for Plotly output
pio.renderers.default = "browser"

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("../data/processed/spacex_enriched.csv")
print(" Loaded rows:", df.shape[0])
print(" Columns:", df.columns.tolist())

# -----------------------------
# 2. Clean Orbit and Dates
# -----------------------------
df['orbit'] = df['orbit_y'].combine_first(df['orbit_x'])
df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
df = df.dropna(subset=['launch_date'])
df['year'] = df['launch_date'].dt.year

# -----------------------------
# 3. Clean landing_success
# -----------------------------
# Map to binary (1, 0), and drop invalids
df['landing_success'] = df['landing_success'].map({
    True: 1, False: 0,
    'True': 1, 'False': 0,
    1: 1, 0: 0
})
df = df.dropna(subset=['landing_success'])

# -----------------------------
# 4. Debug Output
# -----------------------------
print("\n Cleaned rows:", df.shape[0])
print(" Unique orbits:", df['orbit'].dropna().unique())
print(" Unique boosters:", df['booster_version'].dropna().nunique())

# -----------------------------
# 5. Orbit-wise Success Rate
# -----------------------------
orbit_success = df.groupby('orbit')['landing_success'].mean().reset_index()

fig1 = px.bar(
    orbit_success,
    x='orbit',
    y='landing_success',
    title="Landing Success Rate by Orbit",
    labels={'landing_success': 'Success Rate'},
    color='landing_success',
    color_continuous_scale='Blues'
)
fig1.show()

# -----------------------------
# 6. Booster Version Usage
# -----------------------------
booster_counts = df['booster_version'].value_counts().reset_index()
booster_counts.columns = ['booster_version', 'launches']

fig2 = px.bar(
    booster_counts,
    x='booster_version',
    y='launches',
    title="Launches per Booster Version",
)
fig2.update_layout(xaxis_tickangle=-45)
fig2.show()

# -----------------------------
# 7. Landing Success Over Years
# -----------------------------
yearly_success = df.groupby('year')['landing_success'].mean().reset_index()

fig3 = px.line(
    yearly_success,
    x='year',
    y='landing_success',
    title='Landing Success Rate Over Time',
    markers=True
)
fig3.update_yaxes(range=[0, 1.05])
fig3.show()

# -----------------------------
# 8. Launch Site Distribution
# -----------------------------
launch_site = df['launch_site_x'].combine_first(df['launch_site_y'])
site_counts = launch_site.value_counts().reset_index()
site_counts.columns = ['launch_site', 'launches']

fig4 = px.pie(
    site_counts,
    values='launches',
    names='launch_site',
    title="Launch Site Distribution"
)
fig4.show()

