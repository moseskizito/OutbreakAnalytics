"""Run the full analysis on simulated data and write results/.

Usage:  python run_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from outbreak import simulate_surveillance, ALGORITHMS, evaluate_algorithm, lead_time
from outbreak.detect import ears_c2

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

df, register = simulate_surveillance()
print(f"{df['district'].nunique()} districts x {df['week'].nunique()} weeks; {len(register)} simulated outbreaks")

rows = []
for stream in ["community_alerts", "facility_cases"]:
    for name, fn in ALGORITHMS.items():
        m, _ = evaluate_algorithm(df, register, stream, fn)
        rows.append(dict(stream=stream.replace("_", " "), algorithm=name, **m))
res = pd.DataFrame(rows).round(2)
res.to_csv(out / "detection_performance.csv", index=False)
print("\n" + res.to_string(index=False))
(out / "detection_performance.md").write_text(res.to_markdown(index=False))

lt = lead_time(df, register, ears_c2)
lt.to_csv(out / "lead_time_ears_c2.csv", index=False)
both = lt.dropna(subset=["lead_weeks"])
print(f"\nEARS C2 - outbreaks detected in both streams: {len(both)}/{len(lt)}; "
      f"community earlier in {int((both['lead_weeks'] > 0).sum())}, same week in {int((both['lead_weeks'] == 0).sum())}, "
      f"later in {int((both['lead_weeks'] < 0).sum())}; median lead {both['lead_weeks'].median():.1f} weeks")

# Figure 1: one district, both streams, with alarms
ob = register.sort_values("size", ascending=False).iloc[0]
g = df[df["district"] == ob["district"]].sort_values("week")
win = g[(g["week"] >= ob["start_week"] - 40) & (g["week"] <= ob["end_week"] + 15)]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.axvspan(ob["start_week"], ob["end_week"], color="#F4D03F", alpha=0.25, label="true outbreak period")
ax.plot(win["week"], win["community_alerts"], color="#2E86C1", lw=1.6, label="community alerts")
ax.plot(win["week"], win["facility_cases"], color="#555", lw=1.6, label="facility-confirmed cases")
for col, colour in [("community_alerts", "#2E86C1"), ("facility_cases", "#555")]:
    al = ears_c2(g[col].to_numpy())
    sel = g[al & g["week"].isin(win["week"])]
    ax.scatter(sel["week"], sel[col], s=55, facecolors="none", edgecolors="#C0392B", lw=1.6, zorder=3)
ax.scatter([], [], s=55, facecolors="none", edgecolors="#C0392B", lw=1.6, label="EARS C2 alarm")
ax.set_xlabel("week")
ax.set_ylabel("weekly count")
ax.set_title(f"{ob['district']} (simulated data): community alerts flag the outbreak before facility data", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "example_district.png", dpi=150)

# Figure 2: weeks to detection by stream and algorithm
fig, ax = plt.subplots(figsize=(7.5, 3.4))
pivot = res.pivot(index="algorithm", columns="stream", values="median_weeks_to_detect")
y = np.arange(len(pivot))
ax.barh(y - 0.2, pivot["community alerts"], 0.4, color="#2E86C1", label="community alerts")
ax.barh(y + 0.2, pivot["facility cases"], 0.4, color="#888", label="facility cases")
ax.set_yticks(y, pivot.index)
ax.set_xlabel("median weeks from outbreak start to first alarm (lower is better)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "time_to_detection.png", dpi=150)
print(f"\nSaved results to {out}")
