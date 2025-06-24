# 🚀 SpaceX Launch Analysis: Booster Landing Success Prediction

A complete data science portfolio project that explores SpaceX rocket launches and predicts booster landing success using machine learning.

> ✨ Built with real-world datasets from the [SpaceX API](https://github.com/r-spacex/SpaceX-API) and [Wikipedia launch logs](https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches).

---

## 📌 Project Overview

This project analyzes SpaceX rocket launches to answer:
- Which booster versions perform best?
- How do payloads and orbits impact landing success?
- Can we predict whether a booster will land successfully?

🔍 This involved web scraping, data wrangling, SQL-style exploration, EDA, dashboarding with Plotly, and training a classification model.

---

## 🧠 Skills Demonstrated

| Category           | Techniques Used                                   |
|--------------------|---------------------------------------------------|
| Data Collection    | REST API, Web scraping (BeautifulSoup)           |
| Data Wrangling     | Pandas, regex, feature engineering                |
| EDA                | Seaborn, Matplotlib, SQL-style queries            |
| Visualization      | Plotly Express, interactive browser dashboards    |
| Modeling           | Scikit-learn, Logistic Regression, Pipelines     |
| Deployment Prep    | Clean repo, CLI-ready code, Markdown formatting  |

---

## 🗂️ Project Structure

```bash
spacex-launch-analysis/
├── data/
│   ├── raw/               ← API + Wikipedia scraped data
│   └── processed/         ← Cleaned and merged data
├── notebooks/
│   ├── 01_api_scraping.py         # SpaceX API fetch
│   ├── 02_data_wrangling.py       # Clean and prepare API data
│   ├── 03_scraping_boosters.py    # Scrape Wikipedia boosters
│   ├── 04_eda_merge_visuals.py    # Merge + EDA
│   ├── 05_sql_eda.py              # SQL-style analysis
│   ├── 06_dashboard_plotly.py     # Interactive dashboards
│   └── 07_modeling.py             # ML prediction of landings
├── requirements.txt
└── README.md

