#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

BASE_DIR = Path("..")
GOLD_DIR = BASE_DIR / "data" / "gold"
FIG_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = "#075020"
SECONDARY = "#8A7A2F"
FIGSIZE = (6, 4)

sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ## Load data

# In[ ]:


train = pd.read_parquet(GOLD_DIR / "train.parquet")
test = pd.read_parquet(GOLD_DIR / "test.parquet")

print(f"train: {train.shape}")
print(f"test: {test.shape}")


# ## Define features

# In[ ]:


features_num_base = [
    "home",
    "rolling_form_3",
    "rolling_form_5",
    "rolling_margin_3",
    "h2h_winrate",
    "days_since_prev",
]

features_num_elo = features_num_base + ["elo_diff_pre"]

features_cat = [
    "opponent",
    "tournament",
]

target = "win"

y_train = train[target].astype(int).copy()
y_test = test[target].astype(int).copy()

X_train_base = train[features_num_base + features_cat].copy()
X_test_base = test[features_num_base + features_cat].copy()

X_train_elo = train[features_num_elo + features_cat].copy()
X_test_elo = test[features_num_elo + features_cat].copy()

assert "elo_diff_pre" in train.columns
assert "elo_diff_pre" in test.columns


# ## Model Training and Evaluation

# In[ ]:


def train_xgb_model(X_train, X_test, y_train, y_test, features_num, features_cat):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
            ("num", "passthrough", features_num),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )),
        ]
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "brier_score": brier_score_loss(y_test, prob),
    }

    return model, prob, pred, metrics


xgb_base_model, prob_xgb_base, pred_xgb_base, xgb_base_metrics = train_xgb_model(
    X_train_base, X_test_base, y_train, y_test, features_num_base, features_cat
)

xgb_elo_model, prob_xgb_elo, pred_xgb_elo, xgb_elo_metrics = train_xgb_model(
    X_train_elo, X_test_elo, y_train, y_test, features_num_elo, features_cat
)

print("\nXGBoost:")
for name, value in xgb_base_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nXGBoost + Elo:")
for name, value in xgb_elo_metrics.items():
    print(f"{name}: {value:.4f}")


# ## Load logistic regression results

# In[ ]:


logreg_comparison = pd.read_csv(REPORT_DIR / "logistic_model_comparison.csv")

logreg_base_metrics = (
    logreg_comparison[logreg_comparison["model"] == "Logistic regression"]
    .drop(columns="model")
    .iloc[0]
    .to_dict()
)

logreg_elo_metrics = (
    logreg_comparison[logreg_comparison["model"] == "Logistic regression + Elo"]
    .drop(columns="model")
    .iloc[0]
    .to_dict()
)


# ## Compare models

# In[ ]:


comparison = pd.DataFrame([
    {"model": "Logistic regression", **logreg_base_metrics},
    {"model": "Logistic regression + Elo", **logreg_elo_metrics},
    {"model": "XGBoost", **xgb_base_metrics},
    {"model": "XGBoost + Elo", **xgb_elo_metrics},
])

comparison


# ## Save outputs

# In[ ]:


comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

with open(REPORT_DIR / "xgboost_metrics.txt", "w", encoding="utf-8") as file:
    file.write("XGBoost model comparison\n")
    file.write("========================\n\n")

    file.write("XGBoost:\n")
    for name, value in xgb_base_metrics.items():
        file.write(f"{name}: {value:.4f}\n")

    file.write("\nXGBoost + Elo:\n")
    for name, value in xgb_elo_metrics.items():
        file.write(f"{name}: {value:.4f}\n")

print("Saved XGBoost outputs.")


# ## Plot model comparison

# In[ ]:


plot_df = comparison.melt(
    id_vars="model",
    value_vars=["accuracy", "roc_auc", "brier_score"],
    var_name="metric",
    value_name="value",
)

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    data=plot_df,
    x="metric",
    y="value",
    hue="model",
    palette=[PRIMARY, "#3E7A4D", SECONDARY, "#B7A65A"],
    ax=ax,
)

ax.set_title("Model comparison")
ax.set_xlabel("Metric")
ax.set_ylabel("Value")
ax.set_ylim(0, 1)
ax.legend(title="Model", loc="upper right", fontsize=8, title_fontsize=9, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / "model_comparison.pdf", bbox_inches="tight")
plt.show()

