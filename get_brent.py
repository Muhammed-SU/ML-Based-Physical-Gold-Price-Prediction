import pandas as pd
import matplotlib.pyplot as plt

# 1. Read FRED file ("." means missing)
raw = pd.read_csv("data/raw/brent_original.csv", na_values=".")
print(raw.columns.tolist())

# 2. Rename columns
raw.columns = ["Date", "brent"]
df = raw.copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# 3. Clean
df = df[~df.index.duplicated(keep="last")]
df = df[df["brent"] > 0].dropna()
df = df[df.index >= "2005-01-01"]

# 4. Checks
print("Rows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe())
ret = df["brent"].pct_change()
print("Biggest daily moves:")
print(ret.abs().sort_values(ascending=False).head(5))

# 5. Save
df.to_csv("data/raw/brent.csv")
print("Saved -> data/raw/brent.csv")

# 6. Plot
df["brent"].plot(title="Brent oil (USD/barrel)", figsize=(10, 4))
plt.tight_layout()
plt.savefig("data/raw/brent.png", dpi=120)
print("Chart saved -> data/raw/brent.png")
