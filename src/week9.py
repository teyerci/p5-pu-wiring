"""Week 9: the naive result and the H2 test (lock note addendum A1, A2, A5, A8).

One run is (condition, seed, fold) of the dropout benchmark, set up as the lock note fixes
and as src.benchmark.run_condition does it: homologue-group folds, the held-out sources'
synapses removed from the training features, held-out target features from the training
graph, held-out numerics limited to the training range, about 20 unlabelled pairs per
positive.

Models, all scored against the benchmark truth y on the held-out pairs:
  baseline   intercept only: the weighted share of recorded training pairs;
  naive      scikit-learn LogisticRegression (lbfgs; L2 on the coefficients, not on the
             intercept), C = 1 / (λ Σa), sample_weight = a; λ from 3-fold inner CV over
             homologue groups with the 1-SE rule and the log-loss for s;
  naive/c    min(naive / c, 1) (Elkan and Noto 2008), at the realised c of the training
             sources (known on the benchmark; addendum A2). Each has its
             own λ by the same inner CV, scored on s through c · min(f/c, 1) = min(f, c)
             (addendum A1); the inner fits are shared;
  pu         logistic model for P(y=1|x) fitted with target s/c, the convex PU risk of
             src.pu (du Plessis, Niu and Sugiyama 2015) at the same c, minimised by L-BFGS on
             the sparse design; own λ by the same inner folds, scored on s through c · f.
             Fitting it in week 9 is the stated deviation of addendum A8.

The lock note's mirror estimate, recomputed on the training sources of the (thinned) graph
with its 95% interval over source types, is recorded but not scored against y: on a thinned
graph it is no estimate of the benchmark's c, whose truth is the recorded graph. Descriptive
only (addendum A5): under connection dropout it should be about the full-graph share x realised
c; under synapse dropout survival depends on strength, so there is no such target.

Per held-out source homologue group (the unit of addendum A2: cell type across sides, an
untyped neuron its own group), each model's weighted squared-error sum and weight are kept,
so the group-level sign tests (D0 = Brier(naive) - Brier(naive/c); D1 and D2 against PU) can
pool a group over seeds.

At q = 1 the truth is the recorded graph itself, so that run is also the real-data run:
it adds the expected number of unrecorded connections among the held-out unlabelled
pairs at the lock note's c scenarios (description only).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.benchmark import (LAMBDA_GRID, STRATA, edges_from_rows, evaluation_pairs, fold_of_groups, homologue_groups,
                           reliability, score, select_lambda_views, thin_rows)
from src.c_estimate import mirror_map, mirror_share, mirror_share_interval
from src.pair_features import design, feature_columns, split_features
from src.pu import class_prior, sigmoid
from src.pairs import region_pair_counts, sample_pairs

GROUPS = ["identity", "cell_type", "region", "degree", "spatial"]
SCENARIOS = (0.53, 0.63, 0.78)


def sklearn_fit(X, s, a, lam: float) -> LogisticRegression:
    """The naive model: weighted logistic regression whose objective, divided by Σa, is the
    weighted mean log-loss plus (λ/2)‖w‖² with the intercept unpenalised."""
    return LogisticRegression(C=1.0 / (lam * a.sum()), solver="lbfgs", tol=1e-8, max_iter=10000).fit(X, s, sample_weight=a)


def _sklearn_predictor(X, s, a, lam):
    m = sklearn_fit(X, s, a, lam)
    return lambda Z: m.predict_proba(Z)[:, 1]


class PUFit:
    """The PU risk of src.pu.PULogistic for a (sparse) design, minimised by L-BFGS:
    R(w) = -π E_P[z] + E_X[log(1 + e^z)] + (λ/2)‖w‖², intercept unpenalised, π = P(s=1)/c,
    the logistic loss with target s/c. c = 1 is the naive objective of `sklearn_fit`."""

    def __init__(self, c: float, lam: float):
        self.c, self.lam = c, lam

    def fit(self, X, s, a, w0=None):
        """`w0` (coefficients, then intercept) is a starting point only: the risk is convex."""
        s, a = np.asarray(s, float), np.asarray(a, float)
        self.pi_ = class_prior(s, a, self.c)
        wx, wp = a / a.sum(), a * s / (a * s).sum()
        cp = X.T @ wp

        def risk(w):
            z = X @ w[:-1] + w[-1]
            p = sigmoid(z)
            value = -self.pi_ * (wp @ z) + wx @ np.logaddexp(0.0, z) + 0.5 * self.lam * w[:-1] @ w[:-1]
            grad = np.append(X.T @ (wx * p) - self.pi_ * cp + self.lam * w[:-1], wx @ p - self.pi_)
            return value, grad

        if w0 is None:
            w0 = np.append(np.zeros(X.shape[1]), np.log(self.pi_ / (1 - self.pi_)))
        r = minimize(risk, w0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "gtol": 1e-10, "ftol": 1e-15})
        self.coef_, self.intercept_, self.n_iter_, self.converged_ = r.x[:-1], float(r.x[-1]), int(r.nit), bool(r.success)
        self.max_grad_, self.message_ = float(np.max(np.abs(r.jac))), str(r.message)
        if not self.converged_:
            warnings.warn(f"PUFit did not converge (λ {self.lam:g}, c {self.c:.3g}): {self.message_}", RuntimeWarning, stacklevel=2)
        return self

    def predict_proba(self, X):
        """f(x): estimated P(y=1|x)."""
        return sigmoid(X @ self.coef_ + self.intercept_)


def rescale(f, c: float) -> np.ndarray:
    """Elkan and Noto's simple correction: P(y=1|x) ≈ P(s=1|x) / c, clipped at 1."""
    return np.minimum(np.asarray(f, float) / c, 1.0)


def missed_given_unlabelled(g, c: float) -> np.ndarray:
    """P(y=1 | x, s=0) under SCAR, from g = P(s=1|x): (1-c)/c · g/(1-g), clipped to [0, 1]."""
    g = np.asarray(g, float)
    return np.clip((1 - c) / c * g / np.maximum(1 - g, 1e-12), 0.0, 1.0)


def naive_start(m, c: float) -> np.ndarray:
    """PU starting point from a naive fit: g ≈ c f, so f's intercept is about g's minus ln c."""
    return np.append(m.coef_[0], m.intercept_[0] - np.log(c))


def _joint_predictor(c: float, log: list):
    """Fit function for `select_lambda_views` that fits naive and, for c < 1, PU warm-started from it;
    predictions are (naive, PU) pairs. Appends (λ, PU converged) to `log`."""
    def fit(X, s, a, lam):
        m = sklearn_fit(X, s, a, lam)
        if c >= 1:
            return lambda Z: (m.predict_proba(Z)[:, 1], None)
        pu = PUFit(c, lam).fit(X, s, a, w0=naive_start(m, c))
        log.append((lam, pu.converged_))
        return lambda Z: (m.predict_proba(Z)[:, 1], pu.predict_proba(Z))
    return fit


def evaluate(y, p, a) -> dict:
    out = score(y, p, a)
    out["ap_y"] = float(average_precision_score(y, p, sample_weight=a))
    out["reliability"] = reliability(p, y, a)
    return out


def benchmark_fold(neurons, edgeroi_roster, ids, q: float, mode: str = "synapse", seed: int = 0, fold: int = 0,
                   tau: int = 10, per_positive: float = 20, n_folds: int = 5, grid=LAMBDA_GRID, keep_scores: bool = False, verbose: bool = False) -> dict:
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")
    if not 0 <= fold < n_folds:
        raise ValueError("fold must lie in [0, n_folds)")
    clock = [time.perf_counter()]

    def tick(stage):
        clock.append(time.perf_counter())
        if verbose:
            print(f"  {stage} {clock[-1] - clock[-2]:.0f} s", flush=True)

    truth = region_pair_counts(edgeroi_roster, ids)
    er = thin_rows(edgeroi_roster, q, mode, seed) if q < 1 else edgeroi_roster
    seen = region_pair_counts(er, ids)
    pos_true = truth[truth >= tau]
    still = seen.reindex(pos_true.index, fill_value=0) >= tau

    roster = neurons[neurons["neuron_id"].isin(ids)]
    groups = homologue_groups(roster)
    fold_of = fold_of_groups(groups, n_folds, seed)
    test_sources = pd.Index(roster.loc[(fold_of == fold).to_numpy(), "neuron_id"])
    train_sources = pd.Index(roster.loc[(fold_of != fold).to_numpy(), "neuron_id"])
    c_train = float(still[~pos_true.index.get_level_values(0).isin(test_sources)].mean())
    mirror = mirror_map(roster)
    c_mirror = mirror_share(seen, mirror, tau, sources=train_sources)

    sample = sample_pairs(ids, seen, tau=tau, per_positive=per_positive, seed=seed)
    pairs = evaluation_pairs(sample, pos_true, still, test_sources)
    f_tr, f_te, test = split_features(neurons, edges_from_rows(er), er, pairs, ids, test_sources, tau=tau, target_graph="train")
    pkey = pd.MultiIndex.from_arrays([pairs["pre_id"], pairs["post_id"]])
    true_syn = truth.reindex(pkey, fill_value=0).to_numpy()
    y = (true_syn >= tau).astype(float)
    s, a = pairs["s"].to_numpy(float), pairs["weight"].to_numpy()
    Xtr, Xte, names = design(f_tr, f_te, feature_columns(f_tr, GROUPS), clip=True, sparse=True)

    s_tr, a_tr = s[~test], a[~test]
    hg = pd.Series(groups.to_numpy(), index=roster["neuron_id"].to_numpy())
    inner_g = hg.reindex(pairs.loc[~test, "pre_id"].to_numpy()).to_numpy()
    has_mirror = c_mirror["pairs"] > 0 and c_mirror["share"] > 0      # otherwise there is no usable estimate
    cs = {"naive_c_true": c_train}
    if has_mirror:
        lo, hi = mirror_share_interval(seen, mirror, tau, roster.set_index("neuron_id")["cell_type"], train_sources, seed=seed)
        c_mirror["interval"] = [lo, hi]
        c_mirror["full_graph_share"] = mirror_share(truth, mirror, tau, sources=train_sources)["share"]
    # one inner pass: at each fold and λ naive is fitted, then PU from naive's solution (same folds, same rule)
    views = {"naive": lambda f: f[0]} | {k: (lambda f, c=c: np.minimum(f[0], c)) for k, c in cs.items()}
    if c_train < 1:
        views["pu"] = lambda f: c_train * f[1]
    inner_log = []
    tick("features")
    sel = select_lambda_views(Xtr, s_tr, a_tr, inner_g, views, grid=grid, n_inner=3, seed=seed, fit=_joint_predictor(c_train, inner_log))
    if c_train >= 1:                         # c = 1: the PU objective is the naive one, so is its λ choice
        sel["pu"] = sel["naive"]
    tick("inner CV")
    fits = {lam: sklearn_fit(Xtr, s_tr, a_tr, lam) for lam in {v[0] for v in sel.values()}}
    pu = PUFit(c_train, sel["pu"][0]).fit(Xtr, s_tr, a_tr, w0=naive_start(fits[sel["pu"][0]], c_train))
    tick("final fits")
    m = fits[sel["naive"][0]]
    f = m.predict_proba(Xte)[:, 1]
    f_train = m.predict_proba(Xtr)[:, 1]

    yt, at, st, s_te = y[test], a[test], true_syn[test], s[test]
    base = float(np.average(s_tr, weights=a_tr))
    preds = {"baseline": np.full(len(yt), base), "naive": f}
    for k, c in cs.items():
        preds[k] = rescale(fits[sel[k][0]].predict_proba(Xte)[:, 1], c)
    preds["pu"] = pu.predict_proba(Xte)
    out = {"q": q, "mode": mode, "seed": seed, "fold": fold, "n_folds": n_folds, "tau": tau,
           "realised_c": float(still.mean()), "realised_c_train": c_train, "c_mirror_train": c_mirror,
           "lambda": {k: v[0] for k, v in sel.items()},
           "lambda_cv": {k: {"loss": v[1], "se": v[2]} for k, v in sel.items()},
           "n_iter": {f"{lam:g}": int(fm.n_iter_[0]) for lam, fm in fits.items()},
           "pu_fit": {"pi": pu.pi_, "n_iter": pu.n_iter_, "converged": pu.converged_, "max_grad": pu.max_grad_,
                      "inner_fits": len(inner_log), "inner_not_converged": [lam for lam, ok in inner_log if not ok]}, "features": len(names),
           "test_sources": len(test_sources), "train_pairs": int((~test).sum()), "eval_pairs": int(test.sum()),
           "deleted_in_eval": int((pairs["stratum"].to_numpy()[test] == "deleted").sum()),
           # mean-level identity: with an unpenalised intercept, the weighted mean fit equals the weighted label rate
           "train_mean_f": float(np.average(f_train, weights=a_tr)), "train_mean_s": base,
           "seconds_parts": dict(zip(("features", "inner_cv", "final_fits"), np.round(np.diff(clock), 1).tolist())),
           "models": {}}
    for name, p in preds.items():
        res = evaluate(yt, p, at)
        res["by_true_stratum"] = {lab: {"weight": float(at[(st >= lo) & (st <= hi)].sum()),
                                        "mean_p": float(np.average(p[(st >= lo) & (st <= hi)], weights=at[(st >= lo) & (st <= hi)])),
                                        "mean_y": float(np.average(yt[(st >= lo) & (st <= hi)], weights=at[(st >= lo) & (st <= hi)]))}
                                  for lo, hi, lab in STRATA if ((st >= lo) & (st <= hi)).any()}
        out["models"][name] = res
    src_group = hg.reindex(pairs.loc[test, "pre_id"].to_numpy()).to_numpy()
    out["by_source_group"] = {g: {"weight": float(at[src_group == g].sum()),
                                  "sse": {k: float(at[src_group == g] @ (p[src_group == g] - yt[src_group == g]) ** 2) for k, p in preds.items()}}
                              for g in np.unique(src_group)}
    if q == 1:
        unl = s_te == 0
        scen = {f"{c:g}": c for c in SCENARIOS}
        if has_mirror:
            scen |= {"mirror_fold": c_mirror["share"]} | {k: v for k, v in zip(("mirror_fold_lo", "mirror_fold_hi"), c_mirror["interval"]) if v > 0}
        out["real"] = {"heldout_recorded": float(at[~unl].sum()), "heldout_unlabelled_weight": float(at[unl].sum()),
                       "expected_missed": {k: float(at[unl] @ missed_given_unlabelled(f[unl], c)) for k, c in scen.items()}}
    if keep_scores:
        out["scores"] = pd.DataFrame({"pre_id": pairs.loc[test, "pre_id"].to_numpy(), "post_id": pairs.loc[test, "post_id"].to_numpy(),
                                      "y": yt, "s": s_te, "weight": at, "true_syn": st, **{f"p_{k}": v for k, v in preds.items()}})
    return out
