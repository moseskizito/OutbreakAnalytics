from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from outbreak import simulate_surveillance, ALGORITHMS, evaluate_algorithm
from outbreak.detect import ears_c2

df, register = simulate_surveillance(n_districts=12, seed=5)


def test_register_matches_flags():
    for _, ob in register.iterrows():
        g = df[(df["district"] == ob["district"]) & df["in_outbreak"]]
        assert g["week"].min() == ob["start_week"]
        assert g["week"].max() == ob["end_week"]


def test_algorithms_return_aligned_booleans():
    y = df[df["district"] == "Arua"]["facility_cases"].to_numpy()
    for name, fn in ALGORITHMS.items():
        a = fn(y)
        assert a.dtype == bool and len(a) == len(y), name


def test_ears_c2_flags_obvious_spike():
    y = np.full(60, 10.0) + np.tile([0, 1, -1, 2, -2], 12)
    y[50] = 60
    assert ears_c2(y)[50]


def test_district_names_are_unique():
    assert df["district"].nunique() == 12


def test_metrics_in_range():
    m, _ = evaluate_algorithm(df, register, "community_alerts", ALGORITHMS["EARS C2"])
    assert 0 <= m["sensitivity"] <= 1
    assert m["false_alarms_per_district_year"] >= 0
