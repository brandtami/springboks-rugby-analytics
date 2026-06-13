# Predicting Rugby World Cup Outcomes: A Machine Learning Approach
### Focus: South Africa (Springboks) — RWC 2027 Championship Probability

*Reproducible ML pipeline for international rugby match prediction and tournament simulation.*

**Author:** Tamara Brand  
**Programme:** CAS Applied Data Science — University of Bern  
**Data period:** 1987–2024  
**📖 [Full Paper PDF →](paper/Final_Project_CAS_ADS_Tamara_Brand.pdf)**

---

## Overview

This project develops a reproducible machine learning pipeline to predict international rugby match outcomes and simulate the Rugby World Cup 2027 tournament. The analysis covers all 24 participating nations across 10 RWC tournaments (1987–2023) and over 1,700 international test matches.

The pipeline follows a structured Medallion Architecture (Bronze → Silver → Gold) and produces probabilistic championship forecasts via 10,000 Monte Carlo simulations using an ensemble of Gradient Boosting and Elo-based predictions.

**Case Study Focus:** South Africa's pursuit of a historic **three-peat championship** (2019, 2023, 2027)—unprecedented in rugby union history. Our analysis estimates the Springboks' RWC 2027 championship probability at **23.2%**.

---

## Quick Start

```bash
git clone https://github.com/brandtami/springboks-rugby-analytics.git
cd springboks-rugby-analytics

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# register jupyter kernel
python3 -m ipykernel install --user \
    --name=springboks-venv \
    --display-name="Python (springboks-venv)"

# run full pipeline (~45 min, Selenium required for data ingestion)
python3 run_pipeline.py
```

**Partial runs:**
```bash
python3 run_pipeline.py --from 02   # restart from feature engineering
python3 run_pipeline.py --only 05   # run simulation only
python3 run_pipeline.py --from 02 --to 04  # run NB02, 03, 04
```
---

## Data Sources

| # | Source | Content | Method |
|---|--------|---------|--------|
| A | [Kaggle — lylebegbie](https://www.kaggle.com/datasets/lylebegbie/international-rugby-union-results-from-18712022) | Tier 1 matches 1995–2024 | `kagglehub` (automatic) |
| B | [Flashscore](https://www.flashscore.com/rugby-union/world/) | All RWC matches 1987–2023 | Selenium browser scrape |
| C | [World Rugby Rankings](https://www.world.rugby/rankings) | Monthly ratings 2003–2024 | Selenium browser scrape |
| D | [World Rugby RWC History](https://www.world.rugby/tournaments/fixtures-results) | Stage reached per team | Static CSV (`data/sources/`) |

> **Note:** Raw data is not tracked in this repository.  
> Parts A, B, C are collected automatically when running `00_data_ingestion.ipynb`.  
> Static reference files are in `data/sources/` and are version-controlled.

---

## Project Structure

```
springboks-rugby-analytics/
│
├── data/
│   ├── sources/           # static reference files (version-controlled)
│   │   ├── rwc_stage_history.csv
│   │   ├── wr_rankings_oct2024_snapshot.csv
│   │   ├── rwc2027_pools.csv
│   │   └── rwc2027_ko_bracket.csv
│   ├── bronze/            # raw scraped/downloaded data (not tracked)
│   ├── silver/            # merged, cleaned data (not tracked)
│   └── gold/              # feature-engineered, split data (not tracked)
│
├── notebooks/
│   ├── 00_data_ingestion.ipynb       # Bronze layer — 4 data sources
│   ├── 01_data_cleaning.ipynb        # Silver layer — merge + clean
│   ├── 02_feature_engineering.ipynb  # Gold layer — features + Elo
│   ├── 03_exploratory_analysis.ipynb # EDA — 8 figures
│   ├── 04_model_training.ipynb       # GBT + LR + walk-forward CV
│   └── 05_model_application.ipynb    # Monte Carlo RWC 2027 simulation
│
├── models/                # trained models (not tracked)
│   └── best_model.pkl
│
├── figures/               # generated PDF figures (not tracked)
├── reports/               # CSV result files (not tracked)
│
├── run_pipeline.py        # pipeline runner
├── requirements.txt
└── README.md
```

---

## Pipeline

The project follows a **Medallion Architecture**:

```
Bronze  →  Silver  →  Gold  →  EDA  →  Training  →  Application
  ↓           ↓         ↓        ↓          ↓              ↓
4 sources   merged   features  8 figs    GBT+LR        Monte Carlo
 scrape     clean    Elo        stats    walk-fwd      RWC 2027
```

### NB00 — Data Ingestion
- **Part A:** Kaggle Tier 1 match results (1995–2024) via `kagglehub`
- **Part B:** Flashscore RWC matches (1987–2023) via Selenium — 428 matches, 26 nations
- **Part C:** World Rugby Rankings (2003–2024, monthly) via Selenium
- **Part D:** RWC stage history from `data/sources/rwc_stage_history.csv`

### NB01 — Data Cleaning
- Direction-agnostic deduplication (Kaggle + Flashscore overlap)
- Expands to team-perspective (2 rows per match)
- `merge_asof(direction='backward')` for WR Rankings and RWC pedigree — **no leakage**

### NB02 — Feature Engineering
- Rolling form (3/5/10 matches) with `shift(1)` — **no leakage**
- Head-to-head win rate via `groupby` — O(n log n)
- Elo rating computed at match level (one update per match, not per row)
- Tournament tier encoding (Gásquez & Royuela, 2016)
- Chronological split: **train ≤ 2019 / test ≥ 2020**

### NB03 — Exploratory Analysis
- Win rate trends, score margins, home advantage
- Elo distribution, RWC pedigree, opponent win rates

### NB04 — Model Training
- Models: Logistic Regression, Gradient Boosting (±Elo)
- Elo-only baseline (Hvattum & Arntzen, 2010)
- Walk-forward cross-validation (4 annual folds: 2016–2019)
- Bootstrap ROC curves with 95% CI

### NB05 — RWC 2027 Simulation
- 10,000 Monte Carlo simulations
- Official RWC 2027 pools and KO bracket (rugbyworldcup.com)
- GBT + Elo blend (50/50 for Tier 1; 20/80 for sparse teams)
- Championship probability per nation

---

## Key Results

The analysis produces **probabilistic tournament forecasts** for all 24 RWC 2027 teams via 10,000 Monte Carlo simulations. South Africa (the case study focus) achieves a 23.2% estimated championship probability, positioning them as title contenders alongside New Zealand (16.1%) and Ireland (15.4%).

Model performance on the held-out test set (2020–2024) demonstrates the value of domain-specific features: the Gradient Boosting classifier with Elo ratings achieves 73.4% accuracy and ROC-AUC of 0.796, with mean walk-forward cross-validation AUC of 0.860 across four annual folds (2016–2019).

**For detailed results tables and feature importance rankings, see the full thesis report.**

---

## Reproducibility

All data collection steps are fully automated. The pipeline is deterministic given the same input data.

```bash
# full pipeline from scratch
python3 run_pipeline.py

# resume after error
python3 run_pipeline.py --from 04
```

Static reference files in `data/sources/` are version-controlled and fully documented — see `data/sources/README.md`.

---

## References

Begbie, L. (2022). *International Rugby Union Results* [Dataset]. Kaggle.  
https://www.kaggle.com/datasets/lylebegbie/international-rugby-union-results-from-18712022) | Tier 1 matches 1995–2024

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning, 109*, 1997–2028.

Gásquez, R., & Royuela, V. (2016). The Determinants of International Football Success. *Social Science Quarterly, 97*(2), 125–141.

Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match result prediction in association football. *International Journal of Forecasting, 26*(3), 460–470.

World Rugby. (2024). *Men's Rankings*. https://www.world.rugby/rankings

World Rugby. (2024). *RWC History*. https://www.rugbyworldcup.com/2027/en/past-tournaments/1995 

World Rugby. (2025). *RWC 2027*. https://www.rugbyworldcup.com/2027/en/
