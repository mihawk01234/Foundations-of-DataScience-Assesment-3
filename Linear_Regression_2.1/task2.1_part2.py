"""
LINEAR REGRESSION 2.1 - GOAL DIFFERENCE (PART 2)
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# STEP 1: READ THE DATASET

df = pd.read_csv("task2.1_model_dataset.csv")
features = ["fifa_rank_diff", "elo_diff", "log_value_diff", "host_diff",
            "age_diff", "caps_diff", "rest_days_diff", "goals_conceded_diff"]
X = df[features]
y = df["goal_diff"]


# STEP 2: FIT THE MODEL ON ALL 104 MATCHES

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

# standardised coefficients so the variables can be compared
coefs = pd.DataFrame({"coef": model.params, "std_err": model.bse, "p_value": model.pvalues})
coefs["std_coef"] = coefs["coef"] * X.std()
print(coefs.round(4))
coefs.to_csv("results/lr21_coefficients.csv")

# residual checks
bp_p = het_breuschpagan(model.resid, sm.add_constant(X))[1]
jb_p = jarque_bera(model.resid)[1]
print("Breusch-Pagan p:", round(bp_p, 3), " Jarque-Bera p:", round(jb_p, 3))


# STEP 3: 10-FOLD CROSS-VALIDATION

folds = KFold(n_splits=10, shuffle=True, random_state=42)
pred_model = cross_val_predict(LinearRegression(), X, y, cv=folds)
pred_baseline = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=folds)   # always predicts the mean
pred_elo = cross_val_predict(LinearRegression(), df[["elo_diff"]], y, cv=folds)      # elo difference only


def scores(pred):
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred)}


metrics = pd.DataFrame({
    "Baseline (mean)": scores(pred_baseline),
    "Elo difference only": scores(pred_elo),
    "Linear regression (8 variables)": scores(pred_model),
}).T
print(metrics.round(3))
metrics.to_csv("results/lr21_metrics.csv")

# repeat with 20 different splits
repeat_rmse = []
for seed in range(20):
    pred = cross_val_predict(LinearRegression(), X, y, cv=KFold(n_splits=10, shuffle=True, random_state=seed))
    repeat_rmse.append(np.sqrt(mean_squared_error(y, pred)))
print("RMSE over 20 repeats: mean", round(np.mean(repeat_rmse), 3), " sd", round(np.std(repeat_rmse), 3))

# result accuracy (1 = home win, 0 = draw, -1 = away win, within 0.5 counts as a draw)
predicted_result = np.where(np.abs(pred_model) <= 0.5, 0, np.sign(pred_model))
accuracy_model = np.mean(predicted_result == np.sign(y))
accuracy_home = np.mean(y > 0)  # always picking the home team
print("Result accuracy - model:", round(accuracy_model, 3), " always home win:", round(accuracy_home, 3))


# STEP 4: ACTUAL VS PREDICTED AND RESIDUAL PLOTS

residuals = y - pred_model
jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(y))  # so the points don't overlap
lims = [min(pred_model.min(), y.min()) - 0.5, max(pred_model.max(), y.max()) + 0.5]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].scatter(pred_model, y + jitter, s=14, color="#2a78d6", alpha=0.8)
axes[0].plot(lims, lims, color="#52514e", linestyle="--", linewidth=1)
axes[0].set_xlabel("Predicted goal difference (cross-validated)")
axes[0].set_ylabel("Actual goal difference")
axes[0].set_title("LR2.1: actual vs predicted", loc="left", fontweight="bold")

axes[1].scatter(pred_model, residuals, s=14, color="#2a78d6", alpha=0.8)
axes[1].axhline(0, color="#52514e", linewidth=1)
axes[1].set_xlabel("Predicted goal difference (cross-validated)")
axes[1].set_ylabel("Residual (actual - predicted)")
axes[1].set_title("LR2.1: residuals", loc="left", fontweight="bold")

for ax in axes:
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/lr21_actual_vs_predicted_residuals.png", dpi=300)
plt.show()


# STEP 5: SAVE THE SUMMARY

summary = {
    "n": len(df),
    "r2": model.rsquared, "adj_r2": model.rsquared_adj, "f_pvalue": model.f_pvalue,
    "breusch_pagan_p": bp_p, "jarque_bera_p": jb_p,
    "cv": metrics.to_dict(orient="index"),
    "cv_rmse_repeat_mean": np.mean(repeat_rmse), "cv_rmse_repeat_sd": np.std(repeat_rmse),
    "result_accuracy_model": accuracy_model, "result_accuracy_home": accuracy_home,
    "home_wins": int((y > 0).sum()), "draws": int((y == 0).sum()), "away_wins": int((y < 0).sum()),
}
with open("results/lr21_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("Saved the results/ and figures/ files")
