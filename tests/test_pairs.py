"""Tests for the step-15 pair sampler and discrepancy count, on hand-built toy tables."""
import numpy as np
import pandas as pd
import pytest

from src.discrepancy import compare, status, type_pair_table
from src.pairs import _pair_from_index, in_region, region_pair_counts, sample_pairs


def test_pair_index_covers_every_ordered_pair_once():
    n = 4
    i, j = _pair_from_index(np.arange(n * (n - 1)), n)
    pairs = set(zip(i.tolist(), j.tolist()))
    assert len(pairs) == 12 and all(a != b for a, b in pairs)


def test_region_counts_sum_cx_neuropils_in_both_spellings():
    er = pd.DataFrame({"pre_id": [1, 1, 1, 2, 3], "post_id": [2, 2, 2, 1, 3],
                       "neuropil": ["FB", "NO(L)", "LAL_R", "AB(R)", "EB"], "syn_count": [4, 3, 50, 2, 9]})
    assert list(in_region(er["neuropil"])) == [True, True, False, True, True]
    s = region_pair_counts(er, [1, 2, 3])
    assert s.to_dict() == {(1, 2): 7, (2, 1): 2}          # LAL dropped; AB counts as FB; self pair dropped


def test_sample_pairs_keeps_positives_and_weights_the_rest():
    ids = [10, 20, 30, 40]
    syn = pd.Series({(10, 20): 12, (20, 10): 3, (30, 40): 10})
    syn.index.names = ["pre_id", "post_id"]
    full = sample_pairs(ids, syn, tau=10, rate=1.0)
    assert len(full) == 12 and full.s.sum() == 2
    assert full.set_index(["pre_id", "post_id"]).loc[(20, 10), "stratum"] == "below_tau"
    assert (full.loc[full.s == 1, "weight"] == 1).all()
    a = sample_pairs(ids, syn, tau=10, rate=0.5, seed=3)
    b = sample_pairs(ids, syn, tau=10, rate=0.5, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a.loc[a.s == 0, "weight"] == 2.0).all() and a.s.sum() == 2
    with pytest.raises(ValueError):
        sample_pairs(ids, syn, rate=0.5, per_positive=10)


def test_discrepancy_status_and_compare():
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3, 4], "side": ["left", "left", "right", "right"]})
    key = pd.Series(["A", "B", "A", "B"])
    syn = pd.Series({(1, 2): 12, (3, 4): 4})
    tl, tr = type_pair_table(neurons, syn, key, "left"), type_pair_table(neurons, syn, key, "right")
    idx = pd.MultiIndex.from_tuples([("A", "B"), ("B", "A")])
    assert list(status(tl, idx, 10)) == ["recorded", "absent"]
    assert list(status(tr, idx, 10)) == ["below_tau", "absent"]
    r = compare(tl, tr, {"A", "B"}, {"A", "B"}, taus=(10,), names=("L", "R"))
    first = r[(r.direction == "L → R") & (r.unit == "type pairs")].iloc[0]
    assert first.recorded_in_first == 1 and first.below_tau_in_second == 1 and first.discrepancy_share == 1.0
    scaled = compare(tl, tr, {"A", "B"}, set(), taus=(5,), names=("L", "R"), scale_y=0.5)
    row = scaled[(scaled.threshold == "scaled") & (scaled.direction == "L → R") & (scaled.unit == "type pairs")].iloc[0]
    assert (row.tau_first, row.tau_second, row.recorded_in_second) == (5, 2, 1)   # 4 synapses pass τ 2.5 → 2


def test_type_mapping_uses_flywire_type_and_merges_compound_values():
    from src.type_mapping import coverage, mapping_table
    fafb = pd.DataFrame({"neuron_id": [1, 2, 3, 4], "cell_type": ["A", "B", "C", None],
                         "region": ["FB", "FB", "ME_L", "FB"]})
    mal = pd.DataFrame({"neuron_id": [11, 12, 13, 14], "cell_type": ["A_m", "AB_m", "D_m", "E_m"],
                        "flywire_type": ["A", "A,B", None, "Z"], "region": ["FB", "FB", "ME(L)", "FB"]})
    tab = mapping_table(fafb, mal)
    assert tab.loc["A,B", "status"] == "matched" and tab.loc["A,B", "fafb_neurons"] == 2
    assert tab.loc["A,B", "malecns_types"] == "AB_m,A_m"          # malecns names come along, never used to match
    assert tab.loc["C", "status"] == "fafb only" and tab.loc["Z", "status"] == "malecns only"
    overall, by_region = coverage(fafb, mal, tab)
    assert (overall["fafb_neurons_matched"], overall["malecns_neurons_matched"]) == (2, 2)
    assert overall["malecns_neurons_without_flywire_type"] == 1


def test_sampler_edge_cases_and_input_checks():
    syn = pd.Series({(1, 2): 10, (2, 1): 11})
    syn.index.names = ["pre_id", "post_id"]
    only_pos = sample_pairs([1, 2], syn, tau=10, per_positive=20)          # no unlabelled pairs: no division by zero
    assert len(only_pos) == 2 and only_pos.attrs["rate"] == 1.0
    with pytest.raises(ValueError, match="no positives"):
        sample_pairs([1, 2], syn, tau=50, per_positive=20)
    with pytest.raises(ValueError, match="unique"):
        sample_pairs([1, 1, 2], syn, rate=0.5)
    wide = pd.concat([syn, pd.Series({(1, 9): 40, (3, 3): 99})])           # out-of-roster and self pairs ignored
    wide.index.names = ["pre_id", "post_id"]
    s = sample_pairs([1, 2, 3], wide, tau=10, rate=1.0)
    assert s.attrs["positives"] == 2 and len(s) == 6 and not ((s.pre_id == s.post_id).any())


def test_discrepancy_uses_the_largest_neuron_pair_not_the_type_total():
    # type pair (A, B) on the left: two neuron pairs of 6 synapses (total 12, max 6) -> below τ = 10
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3, 4, 5, 6], "side": ["left"] * 4 + ["right"] * 2})
    key = pd.Series(["A", "A", "B", "B", "A", "B"])
    syn = pd.Series({(1, 3): 6, (2, 4): 6, (5, 6): 15})
    tl, tr = type_pair_table(neurons, syn, key, "left"), type_pair_table(neurons, syn, key, "right")
    assert tl.loc[("A", "B"), "total"] == 12
    assert list(status(tl, pd.MultiIndex.from_tuples([("A", "B")]), 10)) == ["below_tau"]
    r = compare(tl, tr, {"A", "B"}, {"B"}, taus=(10,), names=("L", "R"))
    rl = r[(r.direction == "R → L") & (r.unit == "type pairs")].iloc[0]     # reverse direction
    assert (rl.recorded_in_first, rl.below_tau_in_second) == (1, 1)
    single = r[(r.direction == "R → L") & (r.unit == "one neuron per type")].iloc[0]
    assert single.recorded_in_first == 0                                   # A is not a singleton type


def test_no_shared_type_pairs_gives_same_threshold_rows_only():
    from src.discrepancy import scale
    neurons = pd.DataFrame({"neuron_id": [1, 2], "side": ["right", "right"]})
    tx = type_pair_table(neurons, pd.Series({(1, 2): 12}), pd.Series(["A", "B"]), "right")
    ty = type_pair_table(neurons, pd.Series(dtype="int64", index=pd.MultiIndex.from_tuples([], names=["pre_id", "post_id"])),
                         pd.Series(["A", "B"]), "right")
    sc = scale(tx, ty, {"A", "B"})
    assert sc != sc                                                        # NaN: no type pair has synapses in both
    r = compare(tx, ty, {"A", "B"}, set(), taus=(10,), names=("x", "y"), scale_y=sc)
    assert set(r.threshold) == {"same"}
    assert r[(r.direction == "x → y") & (r.unit == "type pairs")].iloc[0].absent_in_second == 1
