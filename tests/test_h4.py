"""Tests for the H4 helpers (values by hand)."""
import numpy as np
import pandas as pd
import pytest

from src.h4 import matched_hit_rate, pair_type_status


def test_matched_hit_rate_standardises_controls_to_candidate_strata():
    # stratum x: controls hit 1 of 4; stratum y: controls hit 3 of 3; stratum z has no control
    hit = [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    cand = [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    strata = ["x", "x", "x", "x", "x", "x", "x", "y", "y", "y", "y", "z"]
    r = matched_hit_rate(hit, cand, strata)
    assert r["candidates"] == 5 and r["candidates_unmatched"] == 1
    assert r["candidate_hit_rate"] == pytest.approx(3 / 4)               # x: 1, 0, 1; y: 1
    assert r["control_hit_rate"] == pytest.approx((3 * 0.25 + 1 * 1.0) / 4)
    assert r["controls_used"] == 7


def test_pair_type_status_maps_neurons_to_type_pairs():
    table = pd.DataFrame({"max_pair": [12, 3], "total": [30, 5]},
                         index=pd.MultiIndex.from_tuples([("A", "B"), ("B", "A")], names=["A", "B"]))
    key = pd.Series({1: "A", 2: "B", 3: "A", 4: np.nan})
    st = pair_type_status([1, 2, 3, 1, 1], [2, 3, 1, 4, 3], key, table, tau=10)
    assert list(st[:3]) == ["recorded", "below_tau", "absent"]          # A→B, B→A, A→A
    assert st[3] != st[3] and st[4] == "absent"                           # no key -> NaN


def test_weights_count_pairs_and_bootstrap_brackets_the_estimate():
    from src.h4 import cluster_bootstrap
    hit, cand, strata = [1, 0, 1, 0], [1, 1, 0, 0], ["x"] * 4
    doubled = matched_hit_rate(hit, cand, strata, weight=[2, 1, 1, 0])     # pair 4 absent, pair 1 twice
    assert doubled["candidate_hit_rate"] == pytest.approx(2 / 3) and doubled["control_hit_rate"] == 1.0
    assert doubled["candidates"] == 3 and doubled["controls_used"] == 1
    rng = np.random.default_rng(1)
    n = 4000
    clusters = rng.integers(0, 80, n)
    cand = rng.random(n) < 0.1
    hit = rng.random(n) < np.where(cand, 0.4, 0.1)
    est = matched_hit_rate(hit, cand, ["s"] * n)
    b = cluster_bootstrap(hit, cand, ["s"] * n, clusters, n_boot=200, seed=0)
    assert b["valid"] == 200 and b["clusters"] == 80
    assert b["candidate_ci"][0] < est["candidate_hit_rate"] < b["candidate_ci"][1]
    assert b["ratio_ci"][0] > 1 and b["share_candidate_above_control"] == 1.0


def test_bootstrap_without_any_control_hit_returns_nan_instead_of_failing():
    from src.h4 import cluster_bootstrap
    b = cluster_bootstrap([1, 0, 0, 0], [1, 0, 0, 0], ["s"] * 4, [0, 1, 2, 3], n_boot=20)
    assert b["valid"] == 0 and b["ratio_ci"][0] != b["ratio_ci"][0]
