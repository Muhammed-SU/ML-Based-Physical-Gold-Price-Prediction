import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

load = lambda n: pd.read_csv(f"data/{n}.csv", parse_dates=["Date"], index_col="Date")
train, val, test, latest = load("train"), load("val"), load("test"), load("latest")
FEATS = [c for c in train.columns if c not in ("target", "gram_altin")]

# Label: 1 = price goes UP tomorrow, 0 = goes DOWN
up = lambda d: (d["target"] > 0).astype(int)

def models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "Random forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30,
                                                max_features=0.3, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8,
                                 colsample_bytree=0.6, min_child_weight=30, random_state=42),
    }

# 1. Train on train, compare on validation -> pick the best model (test stays untouched)
print("Validation accuracy (2023-2024):")
print(f"  {'Always UP (baseline)':22s} {up(val).mean()*100:.1f}%")
val_acc = {}
for name, m in models().items():
    m.fit(train[FEATS], up(train))
    val_acc[name] = accuracy_score(up(val), m.predict(val[FEATS])) * 100
    print(f"  {name:22s} {val_acc[name]:.1f}%")
best = max(val_acc, key=val_acc.get)
print(f"\nBest on validation: {best}")

# 2. Final check on test: train on train+val, predict 2025-2026
trval = pd.concat([train, val])
m = models()[best].fit(trval[FEATS], up(trval))
pred = m.predict(test[FEATS])
acc = accuracy_score(up(test), pred) * 100
base = up(test).mean() * 100
print(f"\n===== TEST (2025 -> 2026, {len(test)} days) =====")
print(f"Always UP (baseline): {base:.1f}%")
print(f"{best}: {acc:.1f}%   ({acc - base:+.1f} points vs baseline)")
cm = confusion_matrix(up(test), pred)
print("\nConfusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm, index=["actual DOWN", "actual UP"], columns=["pred DOWN", "pred UP"]))

# 3. Tomorrow: train on ALL data, predict from the latest day
alld = pd.concat([train, val, test])
m = models()[best].fit(alld[FEATS], up(alld))
p_up = m.predict_proba(latest[FEATS])[0, 1]
day = latest.index[-1].date()
print(f"\n===== PREDICTION FOR THE NEXT TRADING DAY after {day} =====")
print(f"Gram gold today: {latest['gram_altin'].iloc[-1]:,.2f} TL")
print(f"Model: {best}")
print(f"Direction: {'UP' if p_up >= 0.5 else 'DOWN'}  (probability of UP: {p_up*100:.1f}%)")
