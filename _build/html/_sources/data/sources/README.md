# Data Sources — Springboks Rugby Analytics
## CAS Applied Data Science | University of Bern | Tamara Brand

---

## Directory Structure

```
data/
├── sources/                  ← static reference files (this directory)
│   ├── rwc_stage_history.csv          # RWC results 1987–2023
│   ├── wr_rankings_oct2024_snapshot.csv  # WR Rankings Oct 2024
│   ├── rwc2027_pools.csv              # RWC 2027 pool assignments
│   └── rwc2027_ko_bracket.csv         # RWC 2027 KO bracket
├── bronze/                   ← raw scraped/downloaded data
│   ├── bronze_kaggle.parquet          # Tier 1 matches 1995–2024
│   ├── bronze_rwc_matches.parquet     # All RWC matches 1987–2023
│   ├── bronze_rankings.parquet        # WR Rankings 2003–2024 (monthly)
│   └── bronze_rwc_history.parquet     # RWC pedigree features
├── silver/                   ← merged, cleaned data
│   └── silver_results.parquet
└── gold/                     ← feature-engineered, split data
    ├── gold_train.parquet             # train ≤ 2019-12-31
    └── gold_test.parquet              # test  ≥ 2020-01-01
```

---

## Source A — Kaggle: Tier 1 Match Results

| Field         | Value |
|---------------|-------|
| **Source**    | Kaggle — lylebegbie/international-rugby-union-results-from-18712022 |
| **DOI**       | https://doi.org/10.34740/KAGGLE/DSN/2510185 |
| **Citation**  | Begbie, L. (2022). International Rugby Union Results. Kaggle. |
| **Coverage**  | 1871–2024, Tier 1 nations only |
| **Used for**  | Professional era (1995–2024), match outcomes + scores |
| **Method**    | Automatic download via `kagglehub` in NB00 Part A |

---

## Source B — Flashscore: RWC Match Results

| Field         | Value |
|---------------|-------|
| **Source**    | Flashscore — Rugby Union / World / World Cup |
| **URL**       | https://www.flashscore.com/rugby-union/world/ |
| **Citation**  | Flashscore (2024). Rugby World Cup Results. |
| **Coverage**  | 1987–2023, all 10 RWC tournaments, all Tier 2 nations |
| **Used for**  | Adding Tier 2 nations (Georgia, Romania, Fiji etc.) to training data |
| **Method**    | Automated browser scraping via Selenium (Edge) in NB00 Part B |

---

## Source C — World Rugby Rankings

| Field         | Value |
|---------------|-------|
| **Source**    | World Rugby — Men's Rankings |
| **URL**       | https://www.world.rugby/rankings/mru |
| **Citation**  | World Rugby (2024). Men's Rankings. |
| **Coverage**  | 2003–2024, monthly snapshots (first Monday per month) |
| **Used for**  | `wr_rank`, `wr_rating` features; Elo anchor for simulation |
| **Method**    | Automated browser scraping via Selenium (Edge) in NB00 Part C |

---

## Source D — RWC Stage History

| Field         | Value |
|---------------|-------|
| **Source**    | World Rugby — RWC History |
| **URL**       | https://www.world.rugby/tournaments/rugbyworldcup/history |
| **Citation**  | World Rugby (2024). RWC History. |
| **Coverage**  | 1987–2023, all participating nations |
| **Used for**  | `rwc_appearances`, `rwc_best_stage`, `rwc_cumul_score` features |
| **File**      | `sources/rwc_stage_history.csv` |
| **Method**    | Manually compiled from official records; stored as CSV for reproducibility |
| **Stage encoding** | pool=1, quarter-final=2, semi-final=3, final=4, winner=5 |

---

## RWC 2027 Tournament Structure

| Field         | Value |
|---------------|-------|
| **Source**    | World Rugby — RWC 2027 Official Website |
| **URL**       | https://www.rugbyworldcup.com/2027/en/ |
| **Pools file**   | `sources/rwc2027_pools.csv` |
| **Bracket file** | `sources/rwc2027_ko_bracket.csv` |
| **Collected**    | June 2025 |

---

## Citation Template (APA)

```
Begbie, L. (2022). International Rugby Union Results [Dataset].
  Kaggle. https://doi.org/10.34740/KAGGLE/DSN/2510185

Flashscore. (2024). Rugby World Cup Results. Flashscore.
  https://www.flashscore.com/rugby-union/world/

World Rugby. (2024). Men's World Rankings. World Rugby.
  https://www.world.rugby/rankings/mru

World Rugby. (2024). Rugby World Cup History. World Rugby.
  https://www.world.rugby/tournaments/rugbyworldcup/history

World Rugby. (2025). Rugby World Cup 2027. World Rugby.
  https://www.rugbyworldcup.com/2027/en/
```
