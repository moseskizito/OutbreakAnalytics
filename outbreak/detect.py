"""Aberration-detection algorithms used in public-health surveillance.

Each function takes a 1-D array of weekly counts for one district and returns
a boolean array of alarms, using only data available up to that week.
"""
from __future__ import annotations

import numpy as np


def ears_c2(y: np.ndarray, k: float = 3.0, baseline: int = 7, guard: int = 2) -> np.ndarray:
    """CDC EARS C2: compare each week to the mean/SD of 7 weeks ending 2 weeks earlier."""
    y = np.asarray(y, float)
    alarm = np.zeros(len(y), dtype=bool)
    for t in range(baseline + guard, len(y)):
        ref = y[t - guard - baseline:t - guard]
        sd = max(ref.std(ddof=1), 1.0)
        alarm[t] = (y[t] - ref.mean()) / sd > k
    return alarm


def historical_limits(y: np.ndarray, k: float = 2.0, window: int = 2, years: int = 2) -> np.ndarray:
    """Historical limits: compare to the same weeks (+/- window) in previous years."""
    y = np.asarray(y, float)
    alarm = np.zeros(len(y), dtype=bool)
    for t in range(52 * years, len(y)):
        ref = np.concatenate([y[t - 52 * j - window:t - 52 * j + window + 1] for j in range(1, years + 1)])
        sd = max(ref.std(ddof=1), 1.0)
        alarm[t] = y[t] > ref.mean() + k * sd
    return alarm


def cusum(y: np.ndarray, k: float = 0.5, h: float = 4.0, baseline: int = 26, guard: int = 2) -> np.ndarray:
    """One-sided CUSUM on standardised counts; resets after each alarm."""
    y = np.asarray(y, float)
    alarm = np.zeros(len(y), dtype=bool)
    s = 0.0
    for t in range(baseline + guard, len(y)):
        ref = y[t - guard - baseline:t - guard]
        z = (y[t] - ref.mean()) / max(ref.std(ddof=1), 1.0)
        s = max(0.0, s + z - k)
        if s > h:
            alarm[t] = True
            s = 0.0
    return alarm


def negbin_threshold(y: np.ndarray, q: float = 0.995, years: int = 2) -> np.ndarray:
    """Seasonal negative-binomial threshold (simplified Farrington-style).

    Mean and dispersion are estimated from the same season in past years, and an
    alarm is raised when the count exceeds the q-quantile of that distribution.
    """
    from scipy.stats import nbinom

    y = np.asarray(y, float)
    alarm = np.zeros(len(y), dtype=bool)
    for t in range(52 * years, len(y)):
        ref = np.concatenate([y[t - 52 * j - 3:t - 52 * j + 4] for j in range(1, years + 1)])
        m, v = ref.mean(), max(ref.var(ddof=1), ref.mean() + 1e-6)
        if v <= m:
            v = m + 1e-6
        r = m ** 2 / (v - m)
        p = r / (r + m)
        alarm[t] = y[t] > nbinom.ppf(q, r, p)
    return alarm


ALGORITHMS = {
    "EARS C2": ears_c2,
    "Historical limits": historical_limits,
    "CUSUM": cusum,
    "Negative-binomial threshold": negbin_threshold,
}
