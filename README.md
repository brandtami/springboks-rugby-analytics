# Predicting Rugby Match Outcomes: Springboks (1992–2022)

## Overview

This project investigates the prediction of international rugby match outcomes for the South African national team (Springboks) over the period 1992–2022. The objective is to develop a reproducible data pipeline and a baseline statistical model to estimate the probability of winning a match.

The analysis follows a structured data science workflow, including data ingestion, cleaning, feature engineering, exploratory data analysis (EDA), and predictive modelling.

---

## Data

The dataset is sourced from Kaggle:

**International Rugby Union Results (1871–2022)**

The raw data is not included in this repository. To reproduce the analysis:

1. Download the dataset from Kaggle  
2. Place the CSV file in `data/bronze/`

---

## Project Structure

springboks-rugby-analytics/
│
├── data/
│ ├── bronze/
│ ├── silver/
│ └── gold/
│
├── notebooks/
│ ├── 00_data_ingestion.ipynb
│ ├── 01_data_cleaning.ipynb
│ ├── 02_features_gold.ipynb
│ ├── 03_analysis_eda.ipynb
│ ├── 04_model_logistic_regression.ipynb
│ └── 05_model_xgboost_comparison.ipynb
│
├── scripts/
│ ├── 00_data_ingestion.py
│ ├── 01_data_cleaning.py
│ ├── 02_features_gold.py
│ ├── 03_analysis_eda.py
│ ├── 04_model_logistic_regression.py
│ └── 05_model_xgboost_comparison.py
│
├── figures/
├── reports/
│
├── run_pipeline.py
├── README.md
└── requirements.txt

---

## Pipeline

Bronze:
- Raw ingestion
- Parquet conversion

Silver:
- Filtering Springboks matches
- Cleaning
- Target creation

Gold:
- Rolling features
- Opponent features
- Elo rating difference
- Chronological split (no leakage)

---

## Reproducibility

Install dependencies:

pip install -r requirements.txt

Run full pipeline:

python run_pipeline.py

Pipeline steps:
1. Data ingestion  
2. Data cleaning  
3. Feature engineering  
4. Exploratory analysis  
5. Logistic regression  
6. XGBoost comparison  

---

## Notes

- Data excluded via `.gitignore`  
- Results stored in `figures/` and `reports/`  
- Elo is implemented South Africa-centred