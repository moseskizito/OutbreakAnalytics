"""Evaluate detection against the outbreak register."""
from __future__ import annotations

import numpy as np
import pandas as pd

EVAL_FROM = 104  # score only weeks after 2 years of history, when every method can run


def _alarms(df: pd.DataFrame, stream: str, fn) -> pd.DataFrame:
    out = []
    for d, g in df.groupby("district", sort=False):
        g = g.sort_values("week")
        out.append(g.assign(alarm=fn(g[stream].to_numpy())))
    return pd.concat(out)


def evaluate_algorithm(df: pd.DataFrame, register: pd.DataFrame, stream: str, fn) -> dict:
    """Sensitivity (outbreaks detected), median weeks to detection, false alarms per district-year."""
    a = _alarms(df, stream, fn)
    a = a[a["week"] >= EVAL_FROM]
    delays = []
    for _, ob in register.iterrows():
        g = a[(a["district"] == ob["district"]) & (a["week"] >= ob["start_week"]) & (a["week"] <= ob["end_week"]) & a["alarm"]]
        delays.append(g["week"].min() - ob["start_week"] if len(g) else np.nan)
    delays = np.array(delays, float)
    quiet = a[~a["in_outbreak"]]
    district_years = quiet.groupby("district").size().sum() / 52
    return dict(
        sensitivity=float(np.mean(~np.isnan(delays))),
        median_weeks_to_detect=float(np.nanmedian(delays)) if np.any(~np.isnan(delays)) else np.nan,
        false_alarms_per_district_year=float(quiet["alarm"].sum() / district_years),
    ), delays


def lead_time(df: pd.DataFrame, register: pd.DataFrame, fn) -> pd.DataFrame:
    """Per outbreak: weeks to detection in each stream and the community lead time."""
    rows = []
    _, dc = evaluate_algorithm(df, register, "community_alerts", fn)
    _, dfac = evaluate_algorithm(df, register, "facility_cases", fn)
    for (_, ob), c, f in zip(register.iterrows(), dc, dfac):
        rows.append(dict(district=ob["district"], start_week=ob["start_week"],
                         community_detect=c, facility_detect=f,
                         lead_weeks=(f - c) if not (np.isnan(c) or np.isnan(f)) else np.nan))
    return pd.DataFrame(rows)
