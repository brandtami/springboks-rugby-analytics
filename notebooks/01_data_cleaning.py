#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np


# ## Silver cleaning

# In[ ]:


BRONZE_PATH = Path("../data/bronze/bronze_results.parquet")
SILVER_DIR = Path("../data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

TEAM = "South Africa"
MIN_YEAR = 1992
MAX_YEAR = 2022

df = pd.read_parquet(BRONZE_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

df = df.rename(columns={"competition": "tournament"})


# ## South Africa perspective

# In[ ]:


df_sa = df[
    df["date"].dt.year.between(MIN_YEAR, MAX_YEAR)
    & (
        (df["home_team"] == TEAM)
        | (df["away_team"] == TEAM)
    )
].copy()

df_sa["year"] = df_sa["date"].dt.year

sa_home = df_sa["home_team"] == TEAM

df_sa["team"] = TEAM
df_sa["opponent"] = np.where(sa_home, df_sa["away_team"], df_sa["home_team"])
df_sa["springboks_score"] = np.where(sa_home, df_sa["home_score"], df_sa["away_score"])
df_sa["opponent_score"] = np.where(sa_home, df_sa["away_score"], df_sa["home_score"])

df_sa["neutral"] = df_sa["neutral"].fillna(False).astype(bool)
df_sa["home"] = np.where(df_sa["neutral"], 0, sa_home.astype(int))

df_sa["score_margin"] = df_sa["springboks_score"] - df_sa["opponent_score"]
df_sa["win"] = (df_sa["score_margin"] > 0).astype(int)
df_sa["draw"] = (df_sa["score_margin"] == 0).astype(int)

df_sa["tournament"] = df_sa["tournament"].fillna("Unknown")


# ## Save Silver

# In[ ]:


columns = [
    "date", "year", "team", "opponent",
    "springboks_score", "opponent_score",
    "score_margin", "win", "draw",
    "home", "neutral", "tournament",
    "stadium", "city", "country"
]

# ensure chronological order (critical for later lagged feature engineering)
df_silver = (
    df_sa[columns]
    .sort_values("date")
    .reset_index(drop=True)
)

# enforce compact and explicit data types (reproducibility)
df_silver["home"] = df_silver["home"].astype("int8")
df_silver["win"] = df_silver["win"].astype("int8")
df_silver["draw"] = df_silver["draw"].astype("int8")

silver_path = SILVER_DIR / "springboks_matches.parquet"
df_silver.to_parquet(silver_path, index=False)

print(f"[SILVER] Saved: {silver_path}")
print(f"[SILVER] Rows: {len(df_silver)}")
print(f"[SILVER] Period: {df_silver['year'].min()}–{df_silver['year'].max()}")
print(f"[SILVER] Win rate: {df_silver['win'].mean():.3f}")

df_silver.head()

