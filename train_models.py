import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

load = lambda n: pd.read_csv(f"data/{n}.csv", parse_dates=["Date"], index_col="Date")
train, val, test = load("train"), load("val"), load("test")

DROP = ["target", "gram_altin"]
FEATS = [c for c in train.columns if c not in DROP]
Xtr, Xva, Xte = train[FEATS], val[FEATS], test[FEATS]
ytr, yva, yte = train["target"], val["target"], test["target"]

# Clip extreme crisis days in the TRAINING target only (limits from train data)
lo, hi = ytr.quantile([0.01, 0.99])
ytr_c = ytr.clip(lo, hi)
print(f"Features: {len(FEATS)} | train {len(train)} | val {len(val)} | test {len(test)}")

def evaluate(name, pred_ret, df):
    y = df["target"].values
    p_now = df["gram_altin"].values
    actual = p_now * np.exp(y)            # tomorrow's real price
    pred = p_now * np.exp(pred_ret)       # tomorrow's predicted price
    return {
        "model": name,
        "MAE_TL": np.mean(np.abs(actual - pred)),
        "RMSE_TL": np.sqrt(np.mean((actual - pred) ** 2)),
        "MAPE_%": np.mean(np.abs(actual - pred) / actual) * 100,
        "RMSE_ret": np.sqrt(np.mean((y - pred_ret) ** 2)),
        "DirAcc_%": np.mean(np.sign(pred_ret) == np.sign(y)) * 100,
    }

preds = {}

# 1. Baselines
preds["Naive (tomorrow = today)"] = np.zeros(len(test))
preds["Drift (train mean return)"] = np.full(len(test), ytr.mean())

# 2. Ridge linear regression
ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
ridge.fit(Xtr, ytr_c)
preds["Ridge regression"] = ridge.predict(Xte)

# 3. Random forest
rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=20,
                           max_features=0.3, n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr_c)
preds["Random forest"] = rf.predict(Xte)

# 4. XGBoost: Huber loss (robust to crisis days) + early stopping on validation
xgb = XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=3,
                   subsample=0.8, colsample_bytree=0.6, min_child_weight=30,
                   reg_lambda=10.0, gamma=0.0,
                   objective="reg:pseudohubererror", huber_slope=0.01,
                   base_score=float(ytr_c.mean()),
                   early_stopping_rounds=300, eval_metric="mae", random_state=42)
xgb.fit(Xtr, ytr_c, eval_set=[(Xva, yva)], verbose=False)
print("XGBoost best iteration:", xgb.best_iteration)
preds["XGBoost"] = xgb.predict(Xte)

# 5. Results on the TEST set (2025 -> today)
res = pd.DataFrame([evaluate(n, p, test) for n, p in preds.items()]).set_index("model")
# Improvement vs simple baselines (easier to read)
nv, dr = res.loc["Naive (tomorrow = today)"], res.loc["Drift (train mean return)"]
res["MAE_vs_naive_%"] = (res["MAE_TL"] / nv["MAE_TL"] - 1) * 100
res["RMSE_vs_naive_%"] = (res["RMSE_TL"] / nv["RMSE_TL"] - 1) * 100
res["Dir_vs_drift_pts"] = res["DirAcc_%"] - dr["DirAcc_%"]
print("\n===== TEST RESULTS =====")
print(res.round(4).to_string())
res.to_csv("results/test_metrics.csv")

# 6. Feature importance (XGBoost)
imp = pd.Series(xgb.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nTop 15 XGBoost features:")
print(imp.head(15).round(4).to_string())
imp.to_csv("results/xgb_feature_importance.csv")

# 7. Save models + predictions
joblib.dump(ridge, "models/ridge.pkl")
joblib.dump(rf, "models/random_forest.pkl")
xgb.save_model("models/xgboost.json")
out = test[["gram_altin", "target"]].copy()
for n, p in preds.items():
    out[n] = test["gram_altin"].values * np.exp(p)
out["actual_next"] = test["gram_altin"].values * np.exp(test["target"].values)
out.to_csv("results/test_predictions.csv")

# 8. Plot last 120 days: actual vs XGBoost
last = out.tail(120)
plt.figure(figsize=(11, 4))
plt.plot(last.index, last["actual_next"], label="Actual")
plt.plot(last.index, last["XGBoost"], "--", label="XGBoost")
plt.title("Gram gold: actual vs predicted (next day, test set)")
plt.legend(); plt.tight_layout()
plt.savefig("results/xgb_actual_vs_pred.png", dpi=120)
print("\nSaved -> models/, results/test_metrics.csv, results/test_predictions.csv, results/xgb_actual_vs_pred.png")
