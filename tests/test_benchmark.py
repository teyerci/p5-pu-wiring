"""Tests for the dropout-benchmark helpers."""
import pandas as pd
import pytest

from src.benchmark import edges_from_rows, score, thin_rows


def rows():
    return pd.DataFrame({"pre_id": [1, 1, 2, 3], "post_id": [2, 2, 3, 1],
                         "neuropil": ["FB", "EB", "FB", "NO"], "syn_count": [30, 20, 5, 12]})


def test_synapse_thinning_never_adds_and_connection_dropout_keeps_whole_pairs():
    er = rows()
    t = thin_rows(er, 0.5, "synapse", seed=1).merge(er, on=["pre_id", "post_id", "neuropil"], suffixes=("", "_orig"))
    assert (t.syn_count <= t.syn_count_orig).all()
    for seed in range(20):
        c = thin_rows(er, 0.5, "connection", seed=seed)
        kept = set(zip(c.pre_id, c.post_id))
        assert ((1, 2) in kept) == (len(c[(c.pre_id == 1) & (c.post_id == 2)]) == 2)   # both neuropil rows or none
        assert (c.syn_count == er.set_index(["pre_id", "post_id", "neuropil"]).loc[
            list(zip(c.pre_id, c.post_id, c.neuropil)), "syn_count"].to_numpy()).all()
    with pytest.raises(ValueError):
        thin_rows(er, 0.5, "other")


def test_edges_from_rows_and_weighted_scores():
    e = edges_from_rows(rows()).set_index(["pre_id", "post_id"])["syn_count"]
    assert e.to_dict() == {(1, 2): 50, (2, 3): 5, (3, 1): 12}
    sc = score(pd.Series([1.0, 0.0, 0.0]).to_numpy(), pd.Series([0.9, 0.1, 0.2]).to_numpy(),
               pd.Series([1.0, 2.0, 2.0]).to_numpy())
    assert sc["auc_y"] == 1.0 and abs(sc["mean_y"] - 0.2) < 1e-12


def test_run_condition_on_a_synthetic_roster():
    """Invariants on a small random roster: c = 1 and PU = naive without thinning, deleted
    connections scored, realised c ≈ q under connection dropout, strata carry the true label."""
    import numpy as np
    from src.benchmark import run_condition
    rng = np.random.default_rng(0)
    n = 60
    ids = np.arange(1, n + 1)
    neurons = pd.DataFrame({"neuron_id": ids, "cell_type": [f"T{i % 12}" for i in ids],
                            "cell_sub_class": "columnar", "side": np.where(ids % 2, "left", "right"),
                            "nt_type": "acetylcholine", "cell_class": "CX"})
    pre, post = rng.integers(1, n + 1, 1500), rng.integers(1, n + 1, 1500)
    er = pd.DataFrame({"pre_id": pre, "post_id": post, "neuropil": rng.choice(["FB", "EB", "LAL_L"], 1500),
                       "syn_count": rng.integers(1, 30, 1500)})
    er = er[er.pre_id != er.post_id].groupby(["pre_id", "post_id", "neuropil"], as_index=False)["syn_count"].sum()
    full = run_condition(neurons, er, ids, 1.0, seed=0, per_positive=3, lam=0.1, test_share=0.25)
    assert full["realised_c"] == 1.0 and full["deleted_in_eval"] == 0
    assert full["models"]["pu_true_c"]["brier_y"] == full["models"]["naive"]["brier_y"]
    thin = run_condition(neurons, er, ids, 0.6, mode="connection", seed=1, per_positive=3, lam=0.1, test_share=0.25)
    assert abs(thin["realised_c"] - 0.6) < 0.1 and thin["deleted_in_eval"] > 0
    m = thin["models"]["naive"]
    assert m["by_true_stratum"]["0"]["mean_y"] == 0.0 and m["by_true_stratum"]["10–19"]["mean_y"] == 1.0
    cv = run_condition(neurons, er, ids, 1.0, seed=0, per_positive=3, lam="cv", test_share=0.25, exclude=("type_i",))
    assert cv["lambda"] in cv["lambda_cv_losses"] and cv["excluded"] == ["type_i"]
    with pytest.raises(ValueError, match="unknown feature columns"):
        run_condition(neurons, er, ids, 1.0, seed=0, per_positive=3, lam=0.1, test_share=0.25, exclude=("typ_i",))
    folds = [run_condition(neurons, er, ids, 1.0, seed=2, per_positive=3, lam=0.1, fold=k, n_folds=3) for k in range(3)]
    assert sum(r["test_sources"] for r in folds) == n                  # every roster neuron held out once
    with pytest.raises(ValueError, match="fold must lie"):
        run_condition(neurons, er, ids, 1.0, seed=2, lam=0.1, fold=3, n_folds=3)


def test_evaluation_pairs_adds_every_deleted_connection_once_with_weight_one():
    from src.benchmark import evaluation_pairs
    idx = pd.MultiIndex.from_tuples([(1, 2), (1, 3), (4, 2), (5, 6)], names=["pre_id", "post_id"])
    pos_true = pd.Series([12, 15, 30, 11], index=idx)
    still = pd.Series([False, True, False, False], index=idx)        # (1,2), (4,2), (5,6) deleted
    sample = pd.DataFrame({"pre_id": [1, 1, 7], "post_id": [2, 3, 8], "syn": [4, 15, 0], "s": [0, 1, 0],
                           "stratum": ["below_tau", "positive", "unrecorded"], "weight": [25.0, 1.0, 25.0]})
    ev = evaluation_pairs(sample, pos_true, still, test_sources=[1, 5])  # source 4 is not held out
    keys = list(zip(ev.pre_id, ev.post_id))
    assert len(keys) == len(set(keys))                               # no pair counted twice
    deleted = ev[ev.stratum == "deleted"]
    assert set(zip(deleted.pre_id, deleted.post_id)) == {(1, 2), (5, 6)} and (deleted.weight == 1.0).all()
    assert ev.set_index(["pre_id", "post_id"]).loc[(7, 8), "weight"] == 25.0   # other unlabelled keep 1/ρ
    assert (4, 2) not in keys                                        # deleted, but a training source


def test_run_condition_refuses_q_outside_the_unit_interval():
    from src.benchmark import run_condition
    with pytest.raises(ValueError):
        run_condition(pd.DataFrame(), pd.DataFrame(), [1, 2], 1.2)


def test_reliability_bins_have_equal_weight_and_match_a_hand_count():
    import numpy as np
    from src.benchmark import reliability
    p = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    rel = reliability(p, y, np.ones(4), bins=2)
    assert [r["weight"] for r in rel] == [2.0, 2.0]
    assert rel[0]["rate_y"] == 0.0 and rel[1]["rate_y"] == 1.0 and abs(rel[1]["mean_p"] - 0.35) < 1e-12


def test_reliability_keeps_weighted_means_with_unequal_weights():
    import numpy as np
    from src.benchmark import reliability
    p = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.array([3.0, 1.0, 1.0, 1.0])
    rel = reliability(p, y, w, bins=2)
    assert [r["weight"] for r in rel] == [3.0, 3.0]                 # the heavy pair fills the first half
    total = sum(r["weight"] * r["rate_y"] for r in rel) / sum(r["weight"] for r in rel)
    assert abs(total - np.average(y, weights=w)) < 1e-12


def test_reliability_heavy_pair_can_leave_a_group_empty():
    import numpy as np
    from src.benchmark import reliability
    rel = reliability(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 0.0, 1.0, 1.0]),
                      np.array([5.0, 1.0, 1.0, 1.0]), bins=4)
    assert len(rel) < 4 and sum(r["weight"] for r in rel) == 8.0     # empty groups are skipped, weight conserved


def test_lambda_is_chosen_from_the_grid_on_training_pairs():
    import numpy as np
    from src.benchmark import LAMBDA_GRID, select_lambda
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    s = (rng.random(3000) < 1 / (1 + np.exp(-(X[:, 0] - 3)))).astype(float)
    lam, losses = select_lambda(X, s, np.ones(3000), groups=np.arange(3000) % 50)
    assert lam in LAMBDA_GRID and set(losses) == set(LAMBDA_GRID) and losses[lam] == min(losses.values())


def test_homologue_groups_and_folds_cover_each_neuron_once_and_keep_types_whole():
    import numpy as np
    from src.benchmark import fold_of_groups, homologue_groups
    roster = pd.DataFrame({"neuron_id": [1, 2, 3, 4, 5, 6, 7],
                           "cell_type": ["A", "A", "B", None, None, "C", "C"],
                           "side": ["left", "right", "left", "left", "right", "center", "left"]})
    g = homologue_groups(roster)
    assert list(g) == ["A", "A", "B", "neuron:4", "neuron:5", "neuron:6", "C"]
    f = fold_of_groups(g, 3, seed=0)
    assert f.notna().all() and set(f) <= {0, 1, 2}
    assert f.iloc[0] == f.iloc[1]                                     # a type stays in one fold
    sizes = np.bincount(f.groupby(g).first(), minlength=3)            # groups per fold
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == g.nunique()
    assert (fold_of_groups(g, 3, seed=0) == f).all()                   # seeded
