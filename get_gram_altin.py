import pandas as pd
import matplotlib.pyplot as plt

OZ = 31.1035

# 1. Read Stooq XAU/TRY (TL per ounce)
raw = pd.read_csv("data/raw/xautry_original.csv")
raw = raw.rename(columns={"Data": "Date", "Zamkniecie": "Close"})
df = raw[["Date", "Close"]].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# 2. Ounce -> gram
df["gram_altin"] = df["Close"] / OZ
df = df[["gram_altin"]]

# 3. Clean
df = df[~df.index.duplicated(keep="last")]
df = df[df["gram_altin"] > 0].dropna()
df = df[df.index >= "2005-01-01"]

# 4. Checks
print("Rows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe().T.round(2))
ret = df["gram_altin"].pct_change()
print("Biggest daily moves:")
print(ret.abs().sort_values(ascending=False).head(5).round(3))

# 5. Consistency check vs XAU/USD x USD/TRY
xau = pd.read_csv("data/raw/xauusd.csv", parse_dates=["Date"], index_col="Date")["ons_usd"]
usd = pd.read_csv("data/raw/usdtry.csv", parse_dates=["Date"], index_col="Date")["usd_try"]
calc = (xau * usd / OZ).reindex(df.index)
diff = (df["gram_altin"] / calc - 1).dropna() * 100
print("\nDifference vs XAU/USD x USD/TRY (%):")
print(diff.describe().round(3))

# 6. Save + plot
df.to_csv("data/raw/gram_altin.csv")
df["gram_altin"].plot(title="Gram gold (TL/gram)", figsize=(10, 4), logy=True)
plt.tight_layout()
plt.savefig("data/raw/gram_altin.png", dpi=120)
print("Saved -> data/raw/gram_altin.csv")
