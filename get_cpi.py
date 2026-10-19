import pandas as pd
import matplotlib.pyplot as plt

# 1. Read EVDS CPI (2003=100), cleaned CSV: Tarih, TP_GENENDEKS_T1
raw = pd.read_csv("data/raw/cpi_original.csv", thousands=",", sep=None, engine="python")
print(raw.head(3)); print(raw.tail(3))

raw = raw.iloc[:, :2]
raw.columns = ["Tarih", "cpi"]
raw["month"] = pd.to_datetime(raw["Tarih"].astype(str).str.strip(), format="%Y-%m", errors="coerce")
raw["cpi"] = pd.to_numeric(raw["cpi"], errors="coerce")
m = raw.dropna(subset=["month", "cpi"]).set_index("month")["cpi"].sort_index()

# 2. Inflation rates
cpi = pd.DataFrame({"cpi": m})
cpi["inflation_yoy"] = cpi["cpi"].pct_change(12) * 100
cpi["inflation_mom"] = cpi["cpi"].pct_change(1) * 100
print("\nLast months:"); print(cpi.tail(4).round(2))

# 3. Release date: month M is published on the 3rd of month M+1
cpi.index = cpi.index + pd.DateOffset(months=1, days=2)

# 4. Daily step series
days = pd.date_range("2005-01-01", pd.Timestamp.today().normalize(), freq="D")
df = cpi.reindex(days.union(cpi.index)).ffill().reindex(days)[["inflation_yoy", "inflation_mom"]]
df.index.name = "Date"
df = df.dropna()

print("\nRows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
print(df.describe().T.round(2))

# 5. Save + plot
df.to_csv("data/raw/cpi.csv")
df["inflation_yoy"].plot(title="Turkey CPI inflation, annual %", figsize=(10, 4), drawstyle="steps-post")
plt.tight_layout()
plt.savefig("data/raw/cpi.png", dpi=120)
print("Saved -> data/raw/cpi.csv")
