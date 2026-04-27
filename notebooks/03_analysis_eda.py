#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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
    "legend.fontsize": 10
})

bok_cmap = LinearSegmentedColormap.from_list("bok_cmap", [SECONDARY, PRIMARY])


# ## Load Gold

# In[2]:


GOLD_PATH = Path("../data/gold/gold_results.parquet")
FIG_DIR = Path("../figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df_gold = pd.read_parquet(GOLD_PATH)


# ### Annual Win Rate
# 
# We analyse the evolution of the Springboks' win rate over time to identify long-term performance trends.

# In[3]:


annual = df_gold.groupby(df_gold["date"].dt.year)["win"].mean().reset_index()

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.plot(
    annual["date"],
    annual["win"],
    marker="o",
    color=PRIMARY,
    label="Annual win rate"
)

# smoothing
y_smooth = annual["win"].rolling(3, center=True, min_periods=1).mean()
ax.plot(
    annual["date"],
    y_smooth,
    linestyle="--",
    color=SECONDARY,
    label="3-year moving average"
)

ax.set_xlabel("Year")
ax.set_ylabel("Win rate")
ax.set_title("Annual win rate")
ax.set_ylim(0, 1)
ax.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "eda_winrate_trend.pdf", bbox_inches="tight")
plt.show()


# ### Score Margin Distribution
# 
# We examine the distribution of score margins to understand typical match outcomes and variability.

# In[4]:


fig, ax = plt.subplots(figsize=FIGSIZE)

sns.histplot(
    df_gold["score_margin"],
    bins=40,
    kde=True,
    color=PRIMARY,
    edgecolor="white",
    ax=ax
)

ax.set_xlabel("Score margin (South Africa – opponent)")
ax.set_ylabel("Frequency")
ax.set_title("Score margin distribution")

fig.tight_layout()
fig.savefig(FIG_DIR / "eda_score_margin.pdf", bbox_inches="tight")
plt.show()


# ### Rolling Form
# 
# We analyse recent team performance (last three matches) and its relationship with match outcomes.

# In[5]:


fig, ax = plt.subplots(figsize=FIGSIZE)

sns.boxplot(
    data=df_gold,
    x="win",
    y="rolling_form_3",
    hue="win",
    palette={0: SECONDARY, 1: PRIMARY},
    legend=False,
    ax=ax
)

ax.set_xlabel("Match outcome")
ax.set_ylabel("Rolling form")
ax.set_title("Rolling form by match outcome")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Loss/draw", "Win"])
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig(FIG_DIR / "eda_rolling_form.pdf", bbox_inches="tight")
plt.show()


# ### Home Advantage
# 
# We analyse whether playing at home is associated with a higher win probability.

# In[6]:


home_summary = (
    df_gold.groupby("home")["win"]
    .mean()
    .reindex([0, 1])
)

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.bar(
    ["Away/Neutral", "Home"],
    home_summary.values,
    color=[SECONDARY, PRIMARY],
    width=0.6
)

ax.set_xlabel("Match location")
ax.set_ylabel("Win rate")
ax.set_title("Win rate by match location")
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(FIG_DIR / "eda_home_advantage.pdf", bbox_inches="tight")
plt.show()


# ### Opponent-Specific Win Rate
# 
# We analyse how the win probability varies across different opponents.

# In[7]:


opponent_summary = (
    df_gold.groupby("opponent")["win"]
    .mean()
    .sort_values(ascending=False)
    .to_frame(name="Win rate")
)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    opponent_summary,
    annot=True,
    fmt=".2f",
    cmap=bok_cmap,
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("Opponent")
ax.set_title("Opponent-specific win rate")

fig.tight_layout()
fig.savefig(FIG_DIR / "eda_opponent_winrate.pdf", bbox_inches="tight")
plt.show()

