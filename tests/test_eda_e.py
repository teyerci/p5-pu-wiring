"""Tests for the dataset-specific parts of src/eda_e.py, on toy malecns-like tables."""
import pandas as pd

from src.eda_e import (brain_part, columnar_cx_types, e4_malecns_input_ratio, e7_columnar_matrix,
                       flags, real_neuropil)


def toy():
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3, 4],
                            "region": ["FB", "ME(R)", "LegNp(T1)(L)", "unknown"]})
    # 3 -> 1 runs from a nerve-cord neuron into the brain: brain_part must drop it
    edges = pd.DataFrame({"pre_id": [1, 1, 2, 3, 3], "post_id": [2, 3, 1, 4, 1], "syn_count": [5, 7, 2, 1, 4]})
    edgeroi = pd.DataFrame({"pre_id": [1, 1, 1, 2], "post_id": [2, 2, 3, 1],
                            "neuropil": ["FB", "LegNp(T1)(L)", "FB", "CentralBrain-unspecified"],
                            "syn_count": [3, 2, 7, 2]})
    return neurons, edges, edgeroi


def test_flags_accept_malecns_nerve_cord_names():
    fl = flags(toy()[0]).set_index("neuron_id")
    assert fl.loc[3, "neuropil"] == "nerve_cord"
    assert list(fl["in_brain"]) == [True, True, False, False]
    assert fl.loc[2, "is_optic"] and not fl.loc[1, "is_optic"]


def test_brain_part_keeps_brain_neurons_edges_and_neuropils():
    n, e, er = brain_part(*toy())
    assert list(n["neuron_id"]) == [1, 2]
    assert sorted(zip(e["pre_id"], e["post_id"])) == [(1, 2), (2, 1)]
    # the nerve-cord row of pair (1, 2) is dropped; the unspecified row stays for real_neuropil to exclude
    assert list(er["neuropil"]) == ["FB", "CentralBrain-unspecified"]


def test_real_neuropil_excludes_unassigned_and_unspecified():
    names = pd.Series(["FB", "UNASGD", "unknown", "Optic-unspecified(L)", "<unspecified>", "ME(R)"])
    assert list(real_neuropil(names)) == [True, False, False, False, False, True]


def test_e4_counts_inputs_from_every_traced_partner():
    neurons, edges, _ = toy()
    neurons = neurons.assign(n_post=[10, 5, 5, 5], super_class="central")
    brain_n, brain_e = brain_part(neurons, edges)
    df, st = e4_malecns_input_ratio(brain_n, edges, brain_e)
    # neuron 1 (FB, central): inputs 2 from neuron 2 plus 4 from nerve-cord neuron 3, n_post 10
    assert df.loc[1, "ratio"] == 0.6
    assert st["median_ratio_brain_partners"] == 0.2
    assert list(df.index) == [1]                       # neuron 2 is optic, so not central


def test_columnar_types_are_whole_fafb_cx_types_matched_by_column():
    fafb = pd.DataFrame({"cell_type": ["PFNp", "PFNp", "EPG", "Mi1"],
                         "cell_class": ["CX", "CX", "CX", "optic"],
                         "cell_sub_class": ["columnar", None, "columnar", "columnar"]})
    types = columnar_cx_types(fafb)
    assert types == {"PFNp", "EPG"}
    mal = pd.DataFrame({"neuron_id": [1, 2, 3, 4], "region": ["NO", "FB", "EB", "LAL(R)"],
                        "cell_type": ["PFNp_a", "PFNp_b", "EPG", "EPG"],
                        "flywire_type": ["PFNp", "PFNp", "EPG", "EPG"]})
    edges = pd.DataFrame({"pre_id": [1, 2, 3], "post_id": [2, 3, 1], "syn_count": [9, 6, 7]})
    m, order = e7_columnar_matrix(mal, edges, types, type_column="flywire_type")
    assert m.shape == (3, 3) and sorted(order) == ["EPG", "PFNp_a", "PFNp_b"]   # neuron 4 is outside the CX
    assert m.sum() == 22


def test_match_groups_merge_compound_values_transitively():
    from src.eda_e import group_key, match_groups
    g = match_groups(pd.Series(["A,B", "C", None]), pd.Series(["A", "B,E", "D", "C"]))
    assert g["A"] == g["B"] == g["E"] == "A,B,E"
    assert g["C"] == "C" and g["D"] == "D"
    k = group_key(pd.Series(["B,E", "C", None]), g)
    assert list(k[:2]) == ["A,B,E", "C"] and pd.isna(k[2])


def test_keyed_type_edges_sum_over_neuron_pairs_with_a_key():
    from src.eda_e import keyed_type_edges
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3, 4]})
    key = pd.Series(["X", "X", "Y", None])
    edges = pd.DataFrame({"pre_id": [1, 2, 3, 1], "post_id": [3, 3, 1, 4], "syn_count": [2, 5, 1, 9]})
    w = keyed_type_edges(neurons, edges, key)
    assert w.to_dict() == {("X", "Y"): 7, ("Y", "X"): 1}          # the edge to neuron 4 (no key) is left out


def test_hemibrain_side_from_instance_token_then_region():
    from src.fetch_hemibrain import side_of
    inst = pd.Series(["EPG(PB08)_R4", "PFNv(PB05)_L4_C7", "Delta7(PB15)_L4R5_R", "FR2(FQ4)_C3_L_small",
                      "CL141_L", "LC10", "FB4A", None])
    region = pd.Series(["EB", "NO", "PB", "FB", "SMP(L)", "LO(R)", "FB", "unknown"])
    assert list(side_of(inst, region)) == ["right", "left", "right", "left", "left", "right", "unknown", "unknown"]


def test_e6_compare_filters_side_region_and_matches_compound_types():
    from src.eda_e import e6_compare
    # copy a: types X, Y on both sides; copy b names X as "X,Z" (one type matching two); W unmatched
    na = pd.DataFrame({"neuron_id": [1, 2, 3, 4, 5], "region": ["FB", "FB", "FB", "FB", "SMP(R)"],
                       "side": ["right", "right", "left", "left", "right"],
                       "type": ["X", "Y", "X", "Y", "W"]})
    nb = pd.DataFrame({"neuron_id": [11, 12, 13], "region": ["FB", "FB", "AMMC(R)"],
                       "side": ["right", "right", "right"], "type": ["X,Z", "Y", "X,Z"]})
    ea = pd.DataFrame({"pre_id": [1, 3, 1], "post_id": [2, 4, 5], "syn_count": [12, 40, 3]})
    eb = pd.DataFrame({"pre_id": [12, 13], "post_id": [11, 11], "syn_count": [15, 2]})
    res, st, info = e6_compare((na, ea, na["type"]), (nb, eb, nb["type"]), population="all")
    assert info["matched"] == {"X,Z", "Y"} and info["regions"] == {"FB"}    # AMMC and SMP only in one copy
    # right side only: a has X->Y (12, bin 10–19), b has Y->X (15); the left edge, W and AMMC are left out
    assert st["neurons_a"] == 2 and st["neurons_b"] == 2 and st["type_pairs"] == 2
    assert res.loc["10–19", "a_to_b"] == 0 and res.loc["10–19", "b_to_a"] == 0
