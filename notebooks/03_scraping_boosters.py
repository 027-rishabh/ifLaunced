import requests
from bs4 import BeautifulSoup
import pandas as pd

# ---- 1. Define the two URLs ----
urls = [
    "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches_(2010%E2%80%932019)",
    "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches_(2020%E2%80%932022)"
]

# ---- 2. Initialize list to hold rows ----
all_launches = []

# ---- 3. Loop through both URLs ----
for url in urls:
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "lxml")
    tables = soup.find_all("table", class_="wikitable")

    print(f"Scraping {url} → Found {len(tables)} tables")

    for table in tables:
        for row in table.find_all("tr")[1:]:  # skip header
            cols = row.find_all("td")
            if len(cols) >= 8:
                launch = {
                    "date": cols[0].text.strip(),
                    "booster_version": cols[1].text.strip(),
                    "launch_site": cols[2].text.strip(),
                    "payload": cols[3].text.strip(),
                    "orbit": cols[4].text.strip(),
                    "customer": cols[5].text.strip(),
                    "launch_outcome": cols[6].text.strip(),
                    "landing_type": cols[7].text.strip(),
                    "landing_outcome": cols[8].text.strip() if len(cols) > 8 else None,
                }
                all_launches.append(launch)

# ---- 4. Convert to DataFrame ----
df = pd.DataFrame(all_launches)

# ---- 5. Save to CSV ----
df.to_csv("../data/raw/wiki_booster_data.csv", index=False)
print(f"Scraped total {len(df)} rows and saved to wiki_booster_data.csv")

