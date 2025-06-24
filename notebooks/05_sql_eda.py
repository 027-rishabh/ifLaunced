import pandas as pd
import sqlite3

# Load merged CSV
df = pd.read_csv("../data/processed/spacex_enriched.csv")

# -----------------------------
# 1. Fix orbit column
# -----------------------------
df['orbit'] = df['orbit_y'].combine_first(df['orbit_x'])

# -----------------------------
# 2. Ensure landing_success is numeric
# -----------------------------
# Convert NaN/None/"True"/"False" to 1/0
df['landing_success'] = df['landing_success'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

# Drop rows where landing_success is missing
df = df.dropna(subset=['landing_success'])

# -----------------------------
# 3. Fix launch_date for SQL
# -----------------------------
df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
df = df.dropna(subset=['launch_date'])  # ensure valid datetime

# -----------------------------
# 4. Create SQLite DB
# -----------------------------
conn = sqlite3.connect(":memory:")
df.to_sql("spacex", conn, index=False, if_exists="replace")

pd.read_sql("SELECT COUNT(*) AS total_launches FROM spacex", conn)

pd.read_sql("""
SELECT orbit,
       COUNT(*) AS total_launches,
       SUM(landing_success) AS successful,
       ROUND(100.0 * SUM(landing_success) / COUNT(*), 2) AS success_rate
FROM spacex
GROUP BY orbit
ORDER BY success_rate DESC
""", conn)

pd.read_sql("""
SELECT booster_version, COUNT(*) AS launches
FROM spacex
GROUP BY booster_version
ORDER BY launches DESC
""", conn)

pd.read_sql("""
SELECT strftime('%Y', launch_date) AS year,
       COUNT(*) AS total,
       SUM(landing_success) AS successful,
       ROUND(1.0 * SUM(landing_success) / COUNT(*), 2) AS success_rate
FROM spacex
GROUP BY year
ORDER BY year
""", conn)

conn.close()

