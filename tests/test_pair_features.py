"""Leave-pair-out pair features on a hand-built toy graph (values computed by hand)."""
import numpy as np
import pandas as pd

from src.pair_features import build_pair_features, design, feature_columns


def toy():
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3], "cell_type": ["A", "B", "B"],
                            "cell_sub_class": ["columnar", None, "ring neuron"],
                            "side": ["left", "left", "right"], "nt_type": ["gaba", "acetylcholine", "glutamate"]})
    edgeroi = pd.DataFrame({"pre_id": [1, 1, 1, 3, 2], "post_id": [2, 2, 3, 2, 3],
                            "neuropil": ["FB", "EB", "FB", "FB", "UNASGD"], "syn_count": [8, 4, 3, 5, 9]})
    edges = edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()
    return neurons, edges, edgeroi


def test_leave_pair_out_removes_the_target_pair_everywhere():
    neurons, edges, edgeroi = toy()
    pairs = pd.DataFrame({"pre_id": [1, 1, 3], "post_id": [2, 3, 1]})
    f = build_pair_features(neurons, edges, edgeroi, pairs, [1, 2, 3], tau=10).set_index(["pre_id", "post_id"])
    # (1, 2): 12 synapses (8 FB + 4 EB) is the target. Left over: neuron 1 outputs FB 3 (to 3);
    # neuron 2's inputs: FB 5 (from 3); UNASGD never counts.
    r = f.loc[(1, 2)]
    assert (r["frac_out_FB"], r["frac_out_EB"]) == (1.0, 0.0)
    assert (r["frac_in_FB"], r["frac_in_EB"]) == (1.0, 0.0)
    assert r["out_deg_i"] == 0 and r["in_deg_j"] == 0                  # its own ≥ 10 edge removed
    assert r["spatial"] == np.log1p(3 * 5) and r["overlap"] == 1.0
    # (1, 3): neuron 1 keeps FB 8 + EB 4 (to 2); neuron 3 has no named-neuropil input left
    r = f.loc[(1, 3)]
    assert r["out_deg_i"] == 1 and r["no_in"] == 1 and r["frac_in_FB"] == 0 and r["spatial"] == 0
    assert (r["sub_i"], r["sub_j"], r["same_type"], r["same_side"]) == ("columnar", "ring neuron", 0, 0)
    # (3, 1): no synapses from 3 onto 1, so nothing is removed
    r = f.loc[(3, 1)]
    assert r["frac_out_FB"] == 1.0 and r["no_in"] == 1 and r["nt_i"] == "glutamate"
    assert f.loc[(1, 2), "sub_j"] == "missing"


def test_design_one_hot_levels_come_from_train_and_numerics_use_train_statistics():
    neurons, edges, edgeroi = toy()
    pairs = pd.DataFrame({"pre_id": [1, 1, 3, 2], "post_id": [2, 3, 1, 1]})
    f = build_pair_features(neurons, edges, edgeroi, pairs, [1, 2, 3])
    cols = feature_columns(f, ["identity", "degree"])
    Xtr, Xte, names = design(f.iloc[:2], f.iloc[2:], cols)
    assert Xtr.shape[1] == Xte.shape[1] == len(names)
    assert "nt_i=gaba" in names and "nt_i=glutamate" not in names    # levels from train only
    j = names.index("nt_i=gaba")
    assert Xte[:, j].tolist() == [0.0, 0.0]                           # unseen level -> all zero


def test_untyped_neurons_are_never_the_same_type():
    neurons, edges, edgeroi = toy()
    neurons.loc[[0, 1], "cell_type"] = None
    f = build_pair_features(neurons, edges, edgeroi, pd.DataFrame({"pre_id": [1], "post_id": [2]}), [1, 2, 3])
    assert f.loc[0, "same_type"] == 0 and f.loc[0, "type_i"] == "untyped"


def test_held_out_synapses_never_reach_training_features():
    from src.pair_features import split_features
    neurons, edges, edgeroi = toy()
    pairs = pd.DataFrame({"pre_id": [1, 3, 2], "post_id": [2, 2, 3]})     # (1, 2) is held out (source 1)
    tr_a, te_a, test = split_features(neurons, edges, edgeroi, pairs, [1, 2, 3], test_sources=[1])
    er_b = edgeroi.copy()
    er_b.loc[(er_b.pre_id == 1) & (er_b.post_id == 2) & (er_b.neuropil == "FB"), "syn_count"] = 80
    tr_b, te_b, _ = split_features(neurons, er_b.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum(),
                                   er_b, pairs, [1, 2, 3], test_sources=[1])
    assert list(test) == [True, False, False]
    pd.testing.assert_frame_equal(tr_a, tr_b)                             # training side is invariant
    pd.testing.assert_frame_equal(te_a, te_b)                             # and the held-out pair is leave-pair-out
    # (3, 2) in training: neuron 2's inputs no longer include anything from held-out source 1
    assert tr_a.set_index(["pre_id", "post_id"]).loc[(3, 2), "no_in"] == 1


def test_design_scales_with_training_statistics_only():
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    test = pd.DataFrame({"x": [100.0]})
    Xtr, Xte, _ = design(train, test, ["x"])
    assert np.allclose(Xtr[:, 0], [-1.0, 0.0, 1.0]) and np.isclose(Xte[0, 0], (100 - 2) / 2)
