import numpy as np
import pandas as pd

df = pd.read_csv("data/dataset_raw.csv", parse_dates=["Date"], index_col="Date")
df["usd_try"] = df["usd_try"].shift(1)   # Stooq USD/TRY closes later than XAU/TRY -> use yesterday's value
f = pd.DataFrame(index=df.index)

# 1. Log returns of market prices + lags + rolling stats
PRICES = ["gram_altin", "ons_usd", "usd_try", "gumus_usd", "brent", "bist100", "dxy"]
for c in PRICES:
    r = np.log(df[c]).diff()
    f[f"{c}_ret"] = r
    for l in (1, 2, 3, 5):
        f[f"{c}_ret_lag{l}"] = r.shift(l)
    f[f"{c}_ret5"] = r.rolling(5).sum()      # 1-week momentum
    f[f"{c}_ret20"] = r.rolling(20).sum()    # 1-month momentum
    f[f"{c}_vol20"] = r.rolling(20).std()    # 1-month volatility

# 2. Technical indicators on gram gold
g = df["gram_altin"]
f["sma20_dist"] = g / g.rolling(20).mean() - 1
f["sma50_dist"] = g / g.rolling(50).mean() - 1
f["sma200_dist"] = g / g.rolling(200).mean() - 1
d = g.diff()
up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
f["rsi14"] = 100 - 100 / (1 + up / dn)
ema12, ema26 = g.ewm(span=12, adjust=False).mean(), g.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
f["macd"] = macd / g
f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / g

# 3. Risk, rates and macro (levels + changes)
f["vix_log"] = np.log(df["vix"])
f["vix_chg"] = f["vix_log"].diff()
f["us_real_yield"] = df["us_real_yield"]
f["us_real_yield_chg5"] = df["us_real_yield"].diff(5)
f["policy_rate"] = df["policy_rate"]
f["inflation_yoy"] = df["inflation_yoy"]
f["inflation_mom"] = df["inflation_mom"]
f["tr_real_rate"] = df["policy_rate"] - df["inflation_yoy"]
f["policy_chg20"] = df["policy_rate"].diff(20)
f["gold_silver_ratio"] = df["ons_usd"] / df["gumus_usd"]

# 4. Calendar
f["dow"] = df.index.dayofweek
f["month"] = df.index.month

# 5. Target: next-day log return of gram gold
f["target"] = np.log(g).shift(-1) - np.log(g)

# Keep price level for plotting / converting predictions back to TL
f["gram_altin"] = g

f = f.replace([np.inf, -np.inf], np.nan).dropna()
print("Dataset:", f.shape, "|", f.index.min().date(), "->", f.index.max().date())
print("Features:", f.shape[1] - 2)

# 6. Chronological split (never random)
train = f[f.index < "2023-01-01"]
val   = f[(f.index >= "2023-01-01") & (f.index < "2025-01-01")]
test  = f[f.index >= "2025-01-01"]
for n, s in [("train", train), ("val", val), ("test", test)]:
    print(f"{n:5s} {len(s):5d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    s.to_csv(f"data/{n}.csv")
f.to_csv("data/dataset_features.csv")

# 7. Sanity checks
print("\nTarget stats (daily log return):")
print(f["target"].describe().round(4))
print("\nTop 10 features by |correlation| with target:")
print(f.drop(columns=["target", "gram_altin"]).corrwith(f["target"]).abs().sort_values(ascending=False).head(10).round(3))
print("\nSaved -> data/dataset_features.csv, train.csv, val.csv, test.csv")
