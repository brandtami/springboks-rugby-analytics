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
3. Execute the notebooks in the order described below

---

## Project Structure

```
springboks-rugby-analytics/
│
├── data/
│   ├── bronze/        # raw data (not tracked)
│   ├── silver/        # cleaned data (not tracked)
│   └── gold/          # modelling dataset (not tracked)
│
├── notebooks/
│   ├── 00_data_ingestion.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_features_gold.ipynb
│   ├── 03_analysis_eda.ipynb
│   └── 04_model_logistic_regression.ipynb
│
├── figures/           # generated plots (EDA & model evaluation)
├── reports/           # model outputs (metrics, coefficients)
│
├── README.md
└── requirements.txt
```

---

## Pipeline

The project follows a Bronze–Silver–Gold architecture:

### Bronze

* Raw dataset ingestion
* Conversion to Parquet format
* Basic type standardisation

### Silver

* Filtering to Springboks matches (1992–2022)
* Data cleaning and consistency checks
* Creation of target variables (`win`, `draw`)

### Gold

* Feature engineering:

  * Rolling form (last matches)
  * Score margin trends
  * Home vs. away indicator
  * Opponent-specific statistics
* Strict chronological ordering to prevent data leakage
* Time-based train/test split

---

## Exploratory Data Analysis (EDA)

The EDA investigates key patterns in the data:

* Annual win rate trends
* Score margin distribution
* Rolling performance indicators
* Home advantage
* Opponent-specific win rates

All figures are stored in the `figures/` directory.

---

## Modelling

A logistic regression model is used as a baseline classifier.

### Features

* Rolling performance metrics
* Match context (home/away)
* Opponent-related variables

### Evaluation

The model is evaluated using:

* Accuracy
* ROC–AUC
* Calibration curve
* Brier score

Results are stored in the `reports/` directory.

---

## Reproducibility

To reproduce the full analysis:

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

1. `00_data_ingestion`
2. `01_data_cleaning`
3. `02_features_gold`
4. `03_analysis_eda`
5. `04_model_logistic_regression`

All intermediate datasets are generated automatically.

---

## Notes

* Data files are excluded from the repository via `.gitignore`
* Figures and reports are included to ensure transparency of results
* The pipeline is designed to be fully reproducible from raw data
