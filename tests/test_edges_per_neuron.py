"""Tests for the A7 edges-per-neuron computation, on a toy roster."""
import pandas as pd

from src.edges_per_neuron import compare, per_neuron


def ps(d):
    return pd.Series(d, index=pd.MultiIndex.from_tuples(list(d)), name="syn")


def test_per_neuron_counts_outgoing_at_tau_and_keeps_zeros():
    s = ps({(1, 2): 10, (1, 3): 9, (2, 1): 30, (1, 9): 50})       # 9 is outside the roster
    assert per_neuron([1, 2, 3], s, 10).to_dict() == {1: 1, 2: 1, 3: 0}


def test_compare_shared_types_and_scaled_tau():
    kf = pd.Series({1: "A", 2: "A", 3: "B", 4: "X"})
    km = pd.Series({11: "A", 12: "B", 13: "B", 14: "Y"})
    pf = ps({(1, 2): 10, (2, 3): 12, (3, 1): 5, (1, 4): 99})
    pm = ps({(11, 12): 12, (11, 13): 20, (12, 11): 30, (13, 14): 99})
    r = compare([1, 2, 3, 4], kf, pf, [11, 12, 13, 14], km, pm, 10, {"same_tau": 10, "scaled_tau": 18})
    assert r["shared_types"] == 2 and r["neurons_fafb"] == 3 and r["neurons_malecns"] == 3
    assert r["fafb"]["mean"] == 2 / 3                              # neuron 1 -> 2, neuron 2 -> 3, neuron 3 none
    s, t = r["settings"]["same_tau"], r["settings"]["scaled_tau"]
    assert s["mean"] == 1.0 and t["mean"] == 2 / 3                 # tau 18 drops (11,12)=12
    assert abs(s["ratio_mean"] - 1.5) < 1e-12
    row = {x["type"]: x for x in s["per_type"]}
    assert row["A"]["n_fafb"] == 2 and row["A"]["n_malecns"] == 1 and row["A"]["mean_fafb"] == 1.0 and row["A"]["mean_malecns"] == 2
