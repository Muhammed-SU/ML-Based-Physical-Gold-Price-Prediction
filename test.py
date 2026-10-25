import pandas as pd
f = pd.read_csv("data/dataset_features.csv", parse_dates=["Date"], index_col="Date")

# Correlation without crisis days (target clipped to 1%-99%)
lo, hi = f["target"].quantile([0.01, 0.99])
calm = f[(f["target"] > lo) & (f["target"] < hi)]
print("All days :", round(f["usd_try_ret"].corr(f["target"]), 3))
print("Calm days:", round(calm["usd_try_ret"].corr(calm["target"]), 3))
print("Spearman :", round(f["usd_try_ret"].corr(f["target"], method="spearman"), 3))
