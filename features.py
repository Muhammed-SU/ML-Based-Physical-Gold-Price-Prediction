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

# 2. Technical indicators on gram gold (ratios -> comparable across years)
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

# 3. Risk, rates, macro -> CHANGES or z-scores only (no raw levels = no "era" memorising)
def zscore(s, n=250):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()

vix_log = np.log(df["vix"])
f["vix_chg"] = vix_log.diff()
f["vix_chg5"] = vix_log.diff(5)
f["vix_z"] = zscore(vix_log)
f["us_real_yield_chg1"] = df["us_real_yield"].diff()
f["us_real_yield_chg5"] = df["us_real_yield"].diff(5)
f["us_real_yield_chg20"] = df["us_real_yield"].diff(20)
f["policy_chg20"] = df["policy_rate"].diff(20)
f["policy_chg60"] = df["policy_rate"].diff(60)
f["inflation_mom"] = df["inflation_mom"]                      # monthly %, similar range in all years
f["inflation_yoy_chg"] = df["inflation_yoy"].diff(21)         # change vs ~1 month ago
f["tr_real_rate_chg20"] = (df["policy_rate"] - df["inflation_yoy"]).diff(20)
f["gold_silver_z"] = zscore(df["ons_usd"] / df["gumus_usd"])

# 4. Calendar
f["dow"] = df.index.dayofweek
f["month"] = df.index.month

# 5. Target: next-day log return of gram gold
f["target"] = np.log(g).shift(-1) - np.log(g)
f["gram_altin"] = g   # kept for converting predictions back to TL (not a feature)

f = f.replace([np.inf, -np.inf], np.nan)
# Latest day (its "tomorrow" is unknown) -> used to predict tomorrow
f.drop(columns=["target"]).dropna().tail(1).to_csv("data/latest.csv")
f = f.dropna()
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

print("\nTop 10 features by |correlation| with target:")
print(f.drop(columns=["target", "gram_altin"]).corrwith(f["target"]).abs().sort_values(ascending=False).head(10).round(3))
print("\nSaved -> data/dataset_features.csv, train.csv, val.csv, test.csv")
