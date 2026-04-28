#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np


# ## Load Silver

# In[ ]:


SILVER_PATH = Path("../data/silver/springboks_matches.parquet")
GOLD_DIR = Path("../data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SILVER_PATH).sort_values("date").reset_index(drop=True)


# ## Temporal feature construction and leakage prevention
# All rolling and historical features are strictly lagged using `shift(1)`, ensuring that only information available prior to each match is used. This prevents information leakage and guarantees temporal validity of the predictive modelling setup.

# In[ ]:


df["rolling_form_3"] = df["win"].shift(1).rolling(3, min_periods=1).mean()
df["rolling_form_5"] = df["win"].shift(1).rolling(5, min_periods=1).mean()
df["rolling_margin_3"] = df["score_margin"].shift(1).rolling(3, min_periods=1).mean()

df["days_since_prev"] = df["date"].diff().dt.days


# ## Head-to-head feature

# In[ ]:


prev_wins = {}
prev_games = {}
h2h_values = []

for _, row in df.iterrows():
    opponent = row["opponent"]

    if prev_games.get(opponent, 0) > 0:
        h2h_values.append(prev_wins.get(opponent, 0) / prev_games[opponent])
    else:
        h2h_values.append(np.nan)

    prev_games[opponent] = prev_games.get(opponent, 0) + 1
    prev_wins[opponent] = prev_wins.get(opponent, 0) + int(row["win"] == 1)

df["h2h_winrate"] = h2h_values


# ## Elo rating feature
# 
# To capture opponent strength, a pre-match Elo rating difference is computed. Elo ratings are updated sequentially over time and stored before each match to ensure no future information is used.
# 
# The implementation is South Africa-centred, meaning that ratings are tracked relative to the Springboks and their opponents rather than as a fully global rating system.

# In[ ]:


def add_elo_features(df, k=20, initial_rating=1500):
    """
    Add pre-match Elo rating difference for South Africa vs. the opponent.
    Elo values are stored before the match update to avoid temporal leakage.
    """
    df = df.sort_values("date").copy()

    ratings = {}
    elo_diff_pre = []

    for _, row in df.iterrows():
        team = "South Africa"
        opponent = row["opponent"]

        team_rating = ratings.get(team, initial_rating)
        opponent_rating = ratings.get(opponent, initial_rating)

        # Store pre-match Elo difference before updating ratings
        elo_diff_pre.append(team_rating - opponent_rating)

        expected_team = 1 / (1 + 10 ** ((opponent_rating - team_rating) / 400))

        if row["springboks_score"] > row["opponent_score"]:
            actual_team = 1
        elif row["springboks_score"] == row["opponent_score"]:
            actual_team = 0.5
        else:
            actual_team = 0

        ratings[team] = team_rating + k * (actual_team - expected_team)
        ratings[opponent] = opponent_rating + k * ((1 - actual_team) - (1 - expected_team))

    df["elo_diff_pre"] = elo_diff_pre

    return df


df = add_elo_features(df)


# ## Save Gold and split

# In[ ]:


# handle missing values from lagged features explicitly
df["rolling_form_3"] = df["rolling_form_3"].fillna(0.5)
df["rolling_form_5"] = df["rolling_form_5"].fillna(0.5)
df["rolling_margin_3"] = df["rolling_margin_3"].fillna(0)
df["h2h_winrate"] = df["h2h_winrate"].fillna(0.5)
df["days_since_prev"] = df["days_since_prev"].fillna(df["days_since_prev"].median())

df_gold = df.reset_index(drop=True)

gold_path = GOLD_DIR / "gold_results.parquet"
df_gold.to_parquet(gold_path, index=False)

TRAIN_END = pd.Timestamp("2016-12-31")

train = df_gold[df_gold["date"] <= TRAIN_END].copy()
test = df_gold[df_gold["date"] > TRAIN_END].copy()

model_columns = [
    "date",
    "home",
    "rolling_form_3",
    "rolling_form_5",
    "rolling_margin_3",
    "h2h_winrate",
    "days_since_prev",
    "elo_diff_pre",
    "opponent",
    "tournament",
    "win"
]

train[model_columns].to_parquet(GOLD_DIR / "train.parquet", index=False)
test[model_columns].to_parquet(GOLD_DIR / "test.parquet", index=False)

print(f"[GOLD] Saved: {gold_path}")
print(f"[GOLD] Rows: {len(df_gold)}")
print(f"[GOLD] Split: train={len(train)}, test={len(test)}")
print(f"[GOLD] Train period: {train['date'].min().date()}–{train['date'].max().date()}")
print(f"[GOLD] Test period: {test['date'].min().date()}–{test['date'].max().date()}")

df_gold[["date", "opponent", "elo_diff_pre"]].head(10)

