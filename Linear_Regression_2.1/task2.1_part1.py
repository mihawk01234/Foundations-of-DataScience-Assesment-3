"""
LINEAR REGRESSION 2.1 - GOAL DIFFERENCE (PART 1)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# STEP 1: READ AND MERGE THE DATA

features = pd.read_csv("match_prediction_features.csv").drop(columns=["home_score", "away_score"])
matches = pd.read_csv("matches.csv")
matches = matches[matches["status"] == "Completed"]  # only played matches

df = features.merge(matches[["match_id", "home_score", "away_score"]], on="match_id")
print("Matches:", len(df), " Missing values:", df.isnull().sum().sum())


# STEP 2: DATA CHECKS AND CORRECTIONS

# Switzerland vs Colombia went to penalties
df.loc[(df["home_team_name"] == "Switzerland") & (df["away_team_name"] == "Colombia"), "result_type"] = "Penalties"

# replace the squad values with the transfermarkt ones
transfermarkt = pd.read_csv("squad_values_transfermarkt.csv").set_index("fifa_code")["squad_total_value_eur"]
for side in ["home", "away"]:
    correct = df[side + "_fifa_code"].map(transfermarkt)
    ratio = df[side + "_squad_total_value_eur"] / correct
    print(side, "teams more than 25% wrong:", sorted(df.loc[(ratio > 1.25) | (ratio < 0.8), side + "_team_name"].unique()))
    df[side + "_squad_total_value_eur"] = correct

# first match of each team has default values (0 goals conceded, 10 rest days)
first_match = (df["home_prev_avg_goals_conceded"] == 0) & (df["away_prev_avg_goals_conceded"] == 0) \
              & (df["home_rest_days"] == 10) & (df["away_rest_days"] == 10)
print("Matches with first-match placeholders:", first_match.sum())

# scores, xG and results are only known after the match so they aren't used as predictors


# STEP 3: TARGET AND EXPLANATORY VARIABLES (HOME - AWAY)

df["goal_diff"] = df["home_score"] - df["away_score"]

df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["log_value_diff"] = np.log(df["home_squad_total_value_eur"]) - np.log(df["away_squad_total_value_eur"])
df["host_diff"] = df["home_is_host"] - df["away_is_host"]
df["age_diff"] = df["home_squad_avg_age"] - df["away_squad_avg_age"]
df["caps_diff"] = df["home_squad_total_caps"] - df["away_squad_total_caps"]
df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
df["goals_conceded_diff"] = df["home_prev_avg_goals_conceded"] - df["away_prev_avg_goals_conceded"]

features = ["fifa_rank_diff", "elo_diff", "log_value_diff", "host_diff",
            "age_diff", "caps_diff", "rest_days_diff", "goals_conceded_diff"]
df = df[["match_id", "home_team_name", "away_team_name"] + features + ["goal_diff"]]
print("Missing values in the model dataset:", df.isnull().sum().sum())


# STEP 4: DESCRIPTIVE STATISTICS AND PLOTS

print(df[features + ["goal_diff"]].describe().round(3))
print("Home wins:", (df["goal_diff"] > 0).sum(), " Draws:", (df["goal_diff"] == 0).sum(),
      " Away wins:", (df["goal_diff"] < 0).sum())

plt.hist(df["goal_diff"], bins=10, edgecolor="black")
plt.title("Goal difference (home - away)")
plt.xlabel("Goal difference")
plt.ylabel("Number of matches")
plt.show()

# scatterplots of each variable vs goal difference
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, features):
    ax.scatter(df[col], df["goal_diff"])
    ax.set_xlabel(col)
    ax.set_ylabel("goal_diff")
plt.tight_layout()
plt.show()

# correlation matrix
plt.figure(figsize=(9, 7))
sns.heatmap(df[features + ["goal_diff"]].corr(), vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f")
plt.tight_layout()
plt.show()

# VIF (above 5 means collinear)
X = sm.add_constant(df[features])
for i, col in enumerate(features, start=1):
    print("VIF", col, round(variance_inflation_factor(X.values, i), 2))


# STEP 5: SAVE THE DATASET

df.to_csv("task2.1_model_dataset.csv", index=False)
print("Saved task2.1_model_dataset.csv")
