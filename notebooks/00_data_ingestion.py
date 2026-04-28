#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


from pathlib import Path
import pandas as pd
import kagglehub


# ## Bronze ingestion

# In[ ]:


DATASET_ID = "lylebegbie/international-rugby-union-results-from-18712022"

BRONZE_DIR = Path("../data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
csv_files = sorted(dataset_path.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)

if not csv_files:
    raise FileNotFoundError("No CSV file found.")

df_bronze = pd.read_csv(csv_files[0])
df_bronze["date"] = pd.to_datetime(df_bronze["date"], errors="coerce")
df_bronze = df_bronze.dropna(subset=["date"]).reset_index(drop=True)

bronze_path = BRONZE_DIR / "bronze_results.parquet"
df_bronze.to_parquet(bronze_path, index=False)

print(f"[BRONZE] Saved: {bronze_path}")
print(f"[BRONZE] Rows: {len(df_bronze)}")
print(f"[BRONZE] Columns: {df_bronze.shape[1]}")

df_bronze.head()

