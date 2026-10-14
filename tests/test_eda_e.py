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
