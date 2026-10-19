import pandas as pd

# file name -> columns to take
FILES = {
    "gram_altin.csv":    ["gram_altin"],          # target -> defines calendar
    "xauusd.csv":        ["ons_usd"],
    "usdtry.csv":        ["usd_try"],
    "xagusd.csv":        None,                    # None = all columns in file
    "brent.csv":         ["brent"],
    "xu100.csv":         ["bist100"],
    "dxy.csv":           ["dxy"],
    "vix.csv":           ["vix"],
    "us_real_yield.csv": ["us_real_yield"],
    "policy_rate.csv":   ["policy_rate"],
    "cpi.csv":           ["inflation_yoy", "inflation_mom"],
}

def load(name, cols):
    df = pd.read_csv(f"data/raw/{name}", parse_dates=["Date"], index_col="Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df if cols is None else df[cols]

# 1. Calendar = days with a gram gold price, up to yesterday (today's bar may be incomplete)
base = load("gram_altin.csv", ["gram_altin"])
today = pd.Timestamp.today().normalize()
base = base[(base.index >= "2005-01-01") & (base.index < today)]
df = base.copy()

# 2. Add every other series: align to calendar, carry last known value forward
for name, cols in FILES.items():
    if name == "gram_altin.csv":
        continue
    s = load(name, cols)
    s = s.reindex(s.index.union(df.index)).ffill(limit=10).reindex(df.index)
    df = df.join(s)
    print(f"{name:20s} -> {list(s.columns)}  missing: {int(s.isna().sum().sum())}")

# 3. Drop the first days where some series has not started yet
print("\nMissing values before dropna:")
print(df.isna().sum()[df.isna().sum() > 0])
df = df.dropna()
df.index.name = "Date"

print("\nFinal dataset:", df.shape)
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.tail(3).round(2).T)

df.to_csv("data/dataset_raw.csv")
print("\nSaved -> data/dataset_raw.csv")
