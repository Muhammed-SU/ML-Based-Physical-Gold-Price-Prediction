import pandas as pd
import matplotlib.pyplot as plt

# 1. Read TCMB table: Tarih | Borc Alma | Borc Verme
raw = pd.read_csv("data/raw/policy_original.csv")
raw = raw.iloc[:, :3]
raw.columns = ["Date", "alma", "verme"]

raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip(),
                             dayfirst=True, format="mixed")
for c in ["alma", "verme"]:
    raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", "."), errors="coerce")

# Before 18 May 2010: overnight borrowing rate. After: 1-week repo (lending column)
raw["rate"] = raw["alma"].where(raw["Date"] < "2010-05-18", raw["verme"])
raw = raw.dropna(subset=["rate"]).sort_values("Date").drop_duplicates("Date", keep="last")
print(raw[["Date", "rate"]].to_string())

# 2. Expand to daily step series
days = pd.date_range("2005-01-01", pd.Timestamp.today().normalize(), freq="D")
s = raw.set_index("Date")["rate"]
df = s.reindex(days.union(s.index)).ffill().reindex(days).to_frame("policy_rate")
df.index.name = "Date"
df = df.dropna()

# 3. Checks
print("\nRows:", len(df))
print("From:", df.index.min().date(), "To:", df.index.max().date())
ch = df["policy_rate"].diff()
print("Biggest changes (points):")
print(ch[ch != 0].abs().sort_values(ascending=False).head(8))

# 4. Save + plot
df.to_csv("data/raw/policy_rate.csv")
df["policy_rate"].plot(title="TCMB policy rate (%)", figsize=(10, 4), drawstyle="steps-post")
plt.tight_layout()
plt.savefig("data/raw/policy_rate.png", dpi=120)
print("Saved -> data/raw/policy_rate.csv")
