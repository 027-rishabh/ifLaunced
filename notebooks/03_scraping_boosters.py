# Scrape Falcon 9 Booster Info from Wikipedia

#In this notebook, we scrape the Falcon 9 booster launch table from Wikipedia to extract extra metadata like:
#- Booster version and block
#- Reused/refurbished status
#- Landing type and outcome

import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"

response = requests.get(URL)
soup = BeautifulSoup(response.content, "lxml")

tables = soup.find_all("table", class_="wikitable")

print(f"Total tables found: {len(tables)}")

booster_data = []

for table in tables[:10]:  # Looping through the first 10 years (2010–2020)
    rows = table.find_all("tr")[1:]  # Skip header row
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 8:  # Make sure row has enough columns
            booster_data.append([
                cols[0].text.strip(),  # Date
                cols[1].text.strip(),  # Booster
                cols[2].text.strip(),  # Launch Site
                cols[3].text.strip(),  # Payload
                cols[4].text.strip(),  # Orbit
                cols[5].text.strip(),  # Customer
                cols[6].text.strip(),  # Launch Outcome
                cols[7].text.strip(),  # Landing Type
                cols[8].text.strip() if len(cols) > 8 else None  # Landing Outcome
            ])

columns = [
    'date', 'booster_version', 'launch_site',
    'payload', 'orbit', 'customer',
    'launch_outcome', 'landing_type', 'landing_outcome'
]

booster_df = pd.DataFrame(booster_data, columns=columns)

booster_df.head()

booster_df.to_csv('../data/raw/wiki_booster_data.csv', index=False)
print("Scraped booster data saved to ../data/raw/wiki_booster_data.csv")

