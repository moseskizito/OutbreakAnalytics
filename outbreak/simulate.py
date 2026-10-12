"""Simulate weekly surveillance counts from two streams with known outbreaks.

Stream 1 - community alerts: reports from village health workers (e.g. eCHIS
           alerts for suspected cases). Noisy, but closer to the onset.
Stream 2 - facility-confirmed cases: aggregate counts reported by health
           facilities (e.g. weekly HMIS/IDSR). Cleaner, but delayed by care
           seeking, testing and reporting.

Districts carry real Ugandan names as labels only; each has a simulated seasonal baseline. Some districts get one outbreak with a
known start week, so detection algorithms can be scored. No real data are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Real Ugandan district names, used only as labels. All counts and outbreaks are simulated.
UGANDA_DISTRICTS = [
    "Arua", "Gulu", "Lira", "Kitgum", "Moroto", "Kotido", "Soroti", "Mbale",
    "Tororo", "Busia", "Jinja", "Iganga", "Kamuli", "Mayuge", "Mukono", "Wakiso",
    "Kampala", "Masaka", "Mubende", "Kassanda", "Kakumiro", "Kagadi", "Hoima",
    "Kikuube", "Masindi", "Kiryandongo", "Fort Portal", "Kasese", "Bundibugyo",
    "Ntoroko", "Kabarole", "Mbarara", "Bushenyi", "Kabale", "Kisoro", "Rukungiri",
    "Ntungamo", "Isingiro", "Kyotera", "Rakai", "Lyantonde", "Sembabule", "Gomba",
    "Mpigi", "Luwero", "Nakasongola", "Adjumani", "Yumbe", "Koboko", "Nebbi",
]


def simulate_surveillance(
    n_districts: int = 40,
    weeks: int = 208,
    outbreak_share: float = 0.5,
    facility_delay_weeks: int = 2,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly counts, outbreak register).

    counts:   district, week, community_alerts, facility_cases, in_outbreak
    register: district, start_week, peak_week, end_week, size
    """
    rng = np.random.default_rng(seed)
    t = np.arange(weeks)
    rows, register = [], []
    for d in range(n_districts):
        name = UGANDA_DISTRICTS[d % len(UGANDA_DISTRICTS)]
        base = rng.lognormal(2.3, 0.5)
        season = 1 + 0.3 * np.sin(2 * np.pi * (t - rng.integers(0, 52)) / 52)
        mu_fac = base * season
        mu_com = mu_fac * rng.uniform(1.2, 2.0)  # community reports more suspected cases

        extra = np.zeros(weeks)
        flag = np.zeros(weeks, dtype=bool)
        if rng.random() < outbreak_share:
            start = int(rng.integers(110, weeks - 25))  # leave >= 2 years of history
            duration = int(rng.integers(6, 14))
            size = rng.uniform(2.0, 5.0) * base  # peak excess per week
            k = np.arange(duration)
            curve = size * np.exp(-0.5 * ((k - duration / 3) / (duration / 5)) ** 2)
            extra[start:start + duration] = curve
            flag[start:start + duration] = True
            register.append(dict(district=name, start_week=start,
                                 peak_week=start + int(np.argmax(curve)),
                                 end_week=start + duration - 1, size=round(float(curve.sum()))))

        # community sees the excess immediately; facilities see it after a delay
        com_excess = extra * rng.uniform(1.0, 1.5)
        fac_excess = np.roll(extra, facility_delay_weeks)
        fac_excess[:facility_delay_weeks] = 0

        community = rng.negative_binomial(5, 5 / (5 + mu_com + com_excess))
        facility = rng.negative_binomial(10, 10 / (10 + mu_fac + fac_excess))
        for w in range(weeks):
            rows.append(dict(district=name, week=w, community_alerts=int(community[w]),
                             facility_cases=int(facility[w]), in_outbreak=bool(flag[w])))
    return pd.DataFrame(rows), pd.DataFrame(register)
