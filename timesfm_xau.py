"""Zero-shot XAU/USD forecasting with Google TimesFM 3.0 (no training needed)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from timesfm3 import TimesFM3Forecaster

CONTEXT = 1024          # how many past trading days the model sees (~4 years)
HORIZON = 5             # forecast 5 trading days = 1 week
TEST_START = "2025-01-01"
os.makedirs("results", exist_ok=True)

# 1. Data: daily XAU/USD close (drop today, its bar may be incomplete)
s = pd.read_csv("data/raw/xauusd.csv", parse_dates=["Date"], index_col="Date")["ons_usd"]
s = s[s.index < pd.Timestamp.today().normalize()].dropna()
values, dates = s.values.astype(np.float32), s.index
print(f"XAU/USD: {len(s)} days, {dates[0].date()} -> {dates[-1].date()}, last = {values[-1]:,.2f}")

# 2. Load TimesFM 3.0 (downloads ~1.3 GB from Hugging Face the first time)
model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
print("Model device:", model.device)

# 3. Backtest: for every test day t, give the model only data up to t, forecast t+1..t+5
origins = [i for i in range(len(s) - HORIZON) if dates[i] >= pd.Timestamp(TEST_START) and i >= CONTEXT]
contexts = [values[i - CONTEXT + 1: i + 1] for i in origins]
print(f"Backtest: {len(origins)} forecast origins from {dates[origins[0]].date()}")

fc, q10, q90 = [], [], []
B = 32
for b in range(0, len(contexts), B):
    for out in model.predict_batch(contexts[b:b + B], horizon=HORIZON, return_quantiles=True):
        fc.append(out.forecast[:HORIZON]); q10.append(out.quantiles[:HORIZON, 0]); q90.append(out.quantiles[:HORIZON, -1])
    print(f"  {min(b + B, len(contexts))}/{len(contexts)}", end="\r")
fc, q10, q90 = map(np.array, (fc, q10, q90))

now = values[origins]                                   # price on forecast day
actual = np.array([values[i + 1: i + 1 + HORIZON] for i in origins])

# 4. Evaluation vs naive ("price stays the same")
def report(h):                                          # h = steps ahead (1..HORIZON)
    a, p = actual[:, h - 1], fc[:, h - 1]
    mae_m, mae_n = np.mean(np.abs(a - p)), np.mean(np.abs(a - now))
    dir_m = np.mean(np.sign(p - now) == np.sign(a - now)) * 100
    dir_up = np.mean(a > now) * 100                     # baseline: always predict UP
    cover = np.mean((a >= q10[:, h - 1]) & (a <= q90[:, h - 1])) * 100
    return {"days_ahead": h, "MAE_TimesFM_$": mae_m, "MAE_naive_$": mae_n,
            "MAE_vs_naive_%": (mae_m / mae_n - 1) * 100,
            "Direction_TimesFM_%": dir_m, "Always_UP_%": dir_up,
            "80%_interval_coverage_%": cover}

res = pd.DataFrame([report(h) for h in (1, HORIZON)]).set_index("days_ahead")
print(f"\n===== BACKTEST {dates[origins[0]].date()} -> {dates[origins[-1]].date()} ({len(origins)} days) =====")
print(res.round(2).T.to_string())
res.to_csv("results/timesfm_xau_backtest.csv")

# 5. Forecast for the coming week from the latest data
out = next(model.predict_batch([values[-CONTEXT:]], horizon=HORIZON, return_quantiles=True))
future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON)
nxt = pd.DataFrame({"forecast": out.forecast[:HORIZON], "low_10%": out.quantiles[:HORIZON, 0],
                    "high_90%": out.quantiles[:HORIZON, -1]}, index=future)
chg = (nxt["forecast"].iloc[-1] / values[-1] - 1) * 100
print(f"\n===== NEXT WEEK FORECAST (last close {dates[-1].date()}: {values[-1]:,.2f} $) =====")
print(nxt.round(2).to_string())
print(f"\n1-week direction: {'UP' if chg > 0 else 'DOWN'} ({chg:+.2f}%)")
nxt.to_csv("results/timesfm_xau_next_week.csv")

# 6. Plot: last 120 days + next-week forecast with 80% band
plt.figure(figsize=(11, 4))
plt.plot(dates[-120:], values[-120:], label="XAU/USD")
plt.plot(nxt.index, nxt["forecast"], "--", label="TimesFM forecast")
plt.fill_between(nxt.index, nxt["low_10%"], nxt["high_90%"], alpha=0.25, label="80% range")
plt.title("XAU/USD - TimesFM 3.0 one-week forecast"); plt.legend(); plt.tight_layout()
plt.savefig("results/timesfm_xau_forecast.png", dpi=120)
print("\nSaved -> results/timesfm_xau_backtest.csv, timesfm_xau_next_week.csv, timesfm_xau_forecast.png")
