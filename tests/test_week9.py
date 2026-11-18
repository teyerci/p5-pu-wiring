"""Tests for the Week 9 naive run, the mirror c proxy and the sparse design."""
import numpy as np
import pandas as pd
import pytest

from src.c_estimate import mirror_map, mirror_share
from src.pair_features import design
from src.pu import PULogistic
from src.week9 import PUFit, benchmark_fold, missed_given_unlabelled, rescale, sklearn_fit


def test_sparse_design_has_the_same_values_as_the_dense_one():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "type_i": ["A", "B", "A", "C"]})
    te = pd.DataFrame({"a": [9.0, 0.0], "type_i": ["B", "Z"]})
    d = design(tr, te, ["a", "type_i"], clip=True)
    s = design(tr, te, ["a", "type_i"], clip=True, sparse=True)
    assert s[2] == d[2]
    assert np.allclose(s[0].toarray(), d[0]) and np.allclose(s[1].toarray(), d[1])
    assert s[1].toarray()[1, 1:].sum() == 0                       # an unseen level stays all-zero


def test_sklearn_naive_matches_the_numpy_naive_and_keeps_the_mean_level():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    s = (rng.random(4000) < 1 / (1 + np.exp(-(X @ [1.0, -0.5, 0.2] - 4)))).astype(float)
    a = rng.choice([1.0, 12.8], size=4000)
    lam = 1e-3
    sk = sklearn_fit(X, s, a, lam)
    npm = PULogistic(c=1.0, lam=lam).fit(X, s, a)
    assert np.allclose(sk.predict_proba(X)[:, 1], npm.predict_proba(X), atol=1e-5)
    # unpenalised intercept: at the optimum the weighted mean prediction equals the weighted label rate
    assert abs(np.average(sk.predict_proba(X)[:, 1], weights=a) - np.average(s, weights=a)) < 1e-6


def test_sparse_pu_fit_matches_the_numpy_pu_and_the_naive_at_c_one():
    from scipy import sparse
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4000, 3))
    y = rng.random(4000) < 1 / (1 + np.exp(-(X @ [1.0, -0.5, 0.2] - 3)))
    s = (y & (rng.random(4000) < 0.6)).astype(float)
    a = rng.choice([1.0, 12.8], size=4000)
    lam = 1e-3
    pu = PUFit(0.6, lam).fit(sparse.csr_matrix(X), s, a)
    ref = PULogistic(c=0.6, lam=lam).fit(X, s, a)
    assert pu.converged_ and np.allclose(pu.predict_proba(X), ref.predict_proba(X), atol=1e-5)
    one = PUFit(1.0, lam).fit(sparse.csr_matrix(X), s, a)              # c = 1: the naive model
    assert np.allclose(one.predict_proba(X), sklearn_fit(X, s, a, lam).predict_proba(X)[:, 1], atol=1e-5)


def test_pu_warm_start_from_naive_reaches_the_same_fit():
    from scipy import sparse
    from src.week9 import naive_start
    rng = np.random.default_rng(3)
    X = sparse.csr_matrix(rng.normal(size=(5000, 4)))
    y = rng.random(5000) < 1 / (1 + np.exp(-(X @ np.array([1.0, -0.5, 0.2, 0.0]) - 3)))
    s = (y & (rng.random(5000) < 0.6)).astype(float)
    a = rng.choice([1.0, 12.8], size=5000)
    for lam in (1e-2, 1e-5):
        cold = PUFit(0.6, lam).fit(X, s, a)
        warm = PUFit(0.6, lam).fit(X, s, a, w0=naive_start(sklearn_fit(X, s, a, lam), 0.6))
        assert warm.converged_ and np.allclose(warm.predict_proba(X), cold.predict_proba(X), atol=1e-6)


def test_lambda_views_share_folds_and_reduce_to_kfold():
    from src.benchmark import select_lambda_kfold, select_lambda_views
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3))
    s = (rng.random(3000) < 1 / (1 + np.exp(-(2 * X[:, 0] - 2)))).astype(float)
    a, g = np.ones(3000), rng.integers(0, 30, 3000)
    grid = (1e-1, 1e-2, 1e-3)
    one = select_lambda_kfold(X, s, a, g, grid=grid)
    many = select_lambda_views(X, s, a, g, {"s": None, "cap": lambda f: np.minimum(f, 0.3)}, grid=grid)
    assert many["s"] == one
    assert many["cap"][1] != one[1]                               # the capped view is scored on its own predictions


def test_rescale_and_missed_given_unlabelled():
    assert rescale([0.2, 0.9], 0.5).tolist() == [0.4, 1.0]
    g = np.array([0.0, 0.01, 0.3])
    assert np.allclose(missed_given_unlabelled(g, 1.0), 0.0)        # c = 1: nothing is missed
    c = 0.6
    py = g / c                                                    # P(y=1|x) under SCAR
    assert np.allclose(missed_given_unlabelled(g, c), py * (1 - c) / (1 - c * py))


def test_mirror_share_interval_brackets_the_share():
    from src.c_estimate import mirror_share_interval
    rng = np.random.default_rng(0)
    ids = np.arange(1, 41)
    roster = pd.DataFrame({"neuron_id": ids, "cell_type": [f"T{(i - 1) // 2}" for i in ids],
                           "side": np.where(ids % 2, "left", "right")})
    m = mirror_map(roster)
    pairs = {(i, j): int(rng.integers(5, 30)) for i in ids for j in ids if i != j and rng.random() < 0.2}
    ps = pd.Series(pairs).rename_axis(["pre_id", "post_id"])
    share = mirror_share(ps, m, 10)["share"]
    lo, hi = mirror_share_interval(ps, m, 10, roster.set_index("neuron_id")["cell_type"])
    assert lo <= share <= hi and hi - lo > 0


def test_mirror_map_and_share():
    roster = pd.DataFrame({"neuron_id": [1, 2, 3, 4, 5, 6, 7],
                           "cell_type": ["A", "A", "B", "B", "C", "C", "C"],
                           "side": ["left", "right", "left", "right", "left", "right", "right"]})
    m = mirror_map(roster)
    assert m.to_dict() == {1: 2, 2: 1, 3: 4, 4: 3}                # C has two on one side
    ps = pd.Series({(1, 3): 12, (2, 4): 15, (3, 1): 11, (4, 2): 3, (1, 5): 50},
                   ).rename_axis(["pre_id", "post_id"])
    r = mirror_share(ps, m, tau=10)
    assert r == {"share": 2 / 3, "pairs": 3}                      # (1,3),(2,4) hit; (3,1) -> (4,2) has 3
    assert mirror_share(ps, m, tau=10, sources=[3])["pairs"] == 1


def synthetic():
    rng = np.random.default_rng(0)
    n = 60
    ids = np.arange(1, n + 1)
    types = [f"T{i % 12}" for i in ids[:48]] + [f"M{i}" for i in range(6) for _ in range(2)]
    neurons = pd.DataFrame({"neuron_id": ids, "cell_type": types, "cell_sub_class": "columnar",
                            "side": np.where(ids % 2, "left", "right"), "nt_type_corrected": "acetylcholine", "cell_class": "CX"})
    pre, post = rng.integers(1, n + 1, 1500), rng.integers(1, n + 1, 1500)
    er = pd.DataFrame({"pre_id": pre, "post_id": post, "neuropil": rng.choice(["FB", "EB", "LAL_L"], 1500),
                       "syn_count": rng.integers(1, 30, 1500)})
    er = er[er.pre_id != er.post_id].groupby(["pre_id", "post_id", "neuropil"], as_index=False)["syn_count"].sum()
    return neurons, er, ids


def test_benchmark_fold_on_a_synthetic_roster():
    neurons, er, ids = synthetic()
    grid = (1e-1, 1e-2)
    full = benchmark_fold(neurons, er, ids, 1.0, seed=0, fold=0, n_folds=3, per_positive=3, grid=grid, keep_scores=True)
    assert full["realised_c_train"] == 1.0 and full["deleted_in_eval"] == 0 and full["lambda"]["naive"] in grid
    sc = full.pop("scores")
    assert np.array_equal(sc["p_naive"], sc["p_naive_c_true"])    # c = 1: naive/c is naive
    if full["lambda"]["pu"] == full["lambda"]["naive"]:             # and PU is naive, up to the solvers
        assert np.allclose(sc["p_pu"], sc["p_naive"], atol=1e-4)
    assert abs(full["train_mean_f"] - full["train_mean_s"]) < 1e-5
    assert set(full["real"]["expected_missed"]) >= {"0.53", "0.63", "0.78"}
    assert full["models"]["baseline"]["auc_y"] == 0.5
    thin = benchmark_fold(neurons, er, ids, 0.6, mode="connection", seed=1, fold=1, n_folds=3, per_positive=3, grid=grid,
                          keep_scores=True)
    sc = thin.pop("scores")
    assert thin["deleted_in_eval"] > 0 and "real" not in thin
    lam = thin["lambda"]
    if lam["naive_c_true"] == lam["naive"]:
        assert np.allclose(sc["p_naive_c_true"], np.minimum(sc["p_naive"] / thin["realised_c_train"], 1))
    assert set(lam) >= {"naive", "naive_c_true", "pu"} and all(v in grid for v in lam.values())
    assert thin["pu_fit"]["converged"] and thin["models"]["pu"]["brier_y"] > 0
    bt = thin["by_source_group"]                                  # per-group sums add up to the run's Brier
    for k in ("naive", "naive_c_true", "pu"):
        assert np.isclose(sum(v["sse"][k] for v in bt.values()) / sum(v["weight"] for v in bt.values()),
                          thin["models"][k]["brier_y"])
    folds = [benchmark_fold(neurons, er, ids, 1.0, seed=2, fold=k, n_folds=3, per_positive=3, grid=grid) for k in range(3)]
    assert sum(r["test_sources"] for r in folds) == len(ids)       # every roster neuron held out once
    with pytest.raises(ValueError):
        benchmark_fold(neurons, er, ids, 1.0, fold=3, n_folds=3)
