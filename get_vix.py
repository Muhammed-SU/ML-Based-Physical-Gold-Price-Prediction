import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. Download VIX (volatility index) from Yahoo
raw = yf.download("^VIX", start="2004-01-01", auto_adjust=True, progress=False)
print(raw.tail(3))

close = raw["Close"]
if isinstance(close, pd.DataFrame):      # new yfinance returns multi-level columns
    close = close.iloc[:, 0]

df = close.rename("vix").to_frame()
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"

# 2. Clean
df = df[~df.index.duplicated(keep="last")]
df = df[df["vix"] > 0].dropna()
df = df[df.index >= "2005-01-01"]

# 3. Checks
print("Rows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe().T)
ret = df["vix"].pct_change()
print("Biggest daily moves:")
print(ret.abs().sort_values(ascending=False).head(5))

# 4. Save + plot
df.to_csv("data/raw/vix.csv")
df["vix"].plot(title="VIX - volatility index", figsize=(10, 4))
plt.tight_layout()
plt.savefig("data/raw/vix.png", dpi=120)
print("Saved -> data/raw/vix.csv")
