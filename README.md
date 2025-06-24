# SpaceX Launch Analysis & Booster Landing Prediction

A complete data science project that explores and models Falcon 9 rocket launches to predict **booster landing success**.

This repo showcases a full pipeline: :satellite: API + web scraping :arrow_right: data cleaning :arrow_right: EDA + SQL :arrow_right: dashboards :arrow_right: machine learning :arrow_right: final predictions.

---

## :clipboard: Problem Statement

Can we:
- Analyze booster performance by orbit, year, and payload?
- Predict the likelihood of successful booster landings?

---

## :hammer_and_wrench: Tools & Skills Demonstrated

| Area               | Tools & Techniques                           |
|--------------------|-----------------------------------------------|
| Data Collection    | `requests`, `BeautifulSoup`, SpaceX REST API |
| Data Wrangling     | `pandas`, `regex`, datetime parsing          |
| Data Merging       | `merge_asof`, fuzzy date alignment           |
| EDA & Dashboards   | `seaborn`, `matplotlib`, `plotly.express`    |
| SQL Analysis       | `sqlite3`, `read_sql()`                      |
| Machine Learning   | `scikit-learn`, pipelines, logistic regression |
| Deployment Ready   | CLI scripts, virtualenv, `.gitignore`        |

---

## :file_folder: Project Structure

```bash
spacex-launch-analysis/
├── data/
│   ├── raw/               ← API + scraped Wikipedia data
│   └── processed/         ← Cleaned + merged final CSVs
├── notebooks/
│   ├── 01_api_scraping.py
│   ├── 02_data_wrangling.py
│   ├── 03_scraping_boosters.py
│   ├── 04_eda_merge_visuals.py
│   ├── 05_sql_eda.py
│   ├── 06_dashboard_plotly.py
│   └── 07_modeling.py
├── requirements.txt
└── README.md

