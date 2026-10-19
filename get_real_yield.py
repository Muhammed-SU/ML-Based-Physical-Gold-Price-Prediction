import pandas as pd
import matplotlib.pyplot as plt

# 1. Download US 10-year real yield (TIPS) from FRED
try:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
    raw = pd.read_csv(url, na_values=".")
    print("Downloaded from FRED")
except Exception as e:
    print("FRED download failed:", e)
    print("Reading manual file data/raw/dfii10_original.csv")
    raw = pd.read_csv("data/raw/dfii10_original.csv", na_values=".")

print(raw.columns.tolist())

# 2. Clean
raw.columns = ["Date", "us_real_yield"]
raw["Date"] = pd.to_datetime(raw["Date"])
df = raw.set_index("Date").sort_index()
df = df[~df.index.duplicated(keep="last")].dropna()
df = df[df.index >= "2005-01-01"]

# 3. Checks (use diff, because the yield can be negative)
print("Rows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe().T.round(2))
ch = df["us_real_yield"].diff()
print("Biggest daily changes (percentage points):")
print(ch.abs().sort_values(ascending=False).head(5).round(2))

# 4. Save + plot
df.to_csv("data/raw/us_real_yield.csv")
df["us_real_yield"].plot(title="US 10-year real yield (%)", figsize=(10, 4))
plt.axhline(0, color="gray", linewidth=0.8)
plt.tight_layout()
plt.savefig("data/raw/us_real_yield.png", dpi=120)
print("Saved -> data/raw/us_real_yield.csv")
