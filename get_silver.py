
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("data/raw", exist_ok=True)

# 1. Read the manually downloaded Stooq file
raw = pd.read_csv("data/raw/xagusd_original.csv")
print(raw.columns.tolist())

# Stooq may use English or Polish column names
raw = raw.rename(columns={"Data": "Date", "Zamkniecie": "Close"})

# 2. Keep only Date + Close
df = raw[["Date", "Close"]].rename(columns={"Close": "gumus_usd"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# 3. Clean
df = df[~df.index.duplicated(keep="last")]
df = df[df["gumus_usd"] > 0].dropna()
df = df[df.index >= "2005-01-01"]

# 4. Checks
print("Rows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe())
ret = df["gumus_usd"].pct_change()
print("Biggest daily moves:")
print(ret.abs().sort_values(ascending=False).head(5))

# 5. Save
df.to_csv("data/raw/xagusd.csv")
print("Saved -> data/raw/xagusd.csv")

# 6. Plot
df["gumus_usd"].plot(title="XAG/USD daily close", figsize=(10, 4))
plt.ylabel("USD / oz")
plt.tight_layout()
plt.savefig("data/raw/xagusd.png", dpi=120)
print("Chart saved -> data/raw/xagusd.png")
