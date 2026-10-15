"""Dropout benchmark (proposal §5): the recorded region is the truth, a thinned copy
is what a model sees, and both models are scored against the truth.

One condition = (q, mode, seed):
  1. truth: roster pairs with ≥ τ synapses over the region's neuropils (y = 1);
  2. thin the roster's edge-neuropil rows (synapse dropout: Binomial per row;
     connection dropout: whole pair kept with probability q), rebuild the pair
     counts and the per-pair edge totals from the thinned rows;
  3. label s from the thinned counts at τ, sample pairs from s (src.pairs), build
     leave-pair-out features on the thinned graph (src.pair_features);
  4. fit naive (c = 1) and PU models on training pairs, score P(y=1|x) against y on
     test pairs. The realised c is known here, so PU is fitted at that c and at
     the Elkan–Noto e1 estimate from held-out training positives.

Only rows touching the roster are thinned: features use each roster neuron's
synapses with all partners, and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.pair_features import design, feature_columns, split_features
from src.pairs import region_pair_counts, sample_pairs
from src.pu import PULogistic, elkan_noto_c
from src.ranking import ranking_agreement

GROUPS = ["identity", "cell_type", "region", "degree", "spatial"]


def thin_rows(edgeroi: pd.DataFrame, q: float, mode: str = "synapse", seed: int = 0) -> pd.DataFrame:
    """Thinned edge-neuropil rows (rows left with 0 synapses dropped)."""
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    er = edgeroi.copy()
    if mode == "synapse":
        er["syn_count"] = rng.binomial(er["syn_count"].to_numpy().astype(np.int64), q)
    elif mode == "connection":
        pairs = er[["pre_id", "post_id"]].drop_duplicates()
        keep = pairs[rng.random(len(pairs)) < q]
        er = er.merge(keep, on=["pre_id", "post_id"])
    else:
        raise ValueError(mode)
    return er[er["syn_count"] > 0].reset_index(drop=True)


def edges_from_rows(edgeroi: pd.DataFrame) -> pd.DataFrame:
    return edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()


def score(y, p, a) -> dict:
    return {"auc_y": float(roc_auc_score(y, p, sample_weight=a)),
            "brier_y": float(np.average((p - y) ** 2, weights=a)),
            "mean_p": float(np.average(p, weights=a)), "mean_y": float(np.average(y, weights=a))}


LAMBDA_GRID = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def select_lambda(X, s, a, groups, grid=LAMBDA_GRID, share=0.2, seed=0) -> tuple[float, dict]:
    """λ chosen on training pairs only: an inner split by group (source neuron), a naive
    fit per λ on the rest, and the weighted log-loss for the observable label s on the
    held-out groups. The true label is never used, so the choice is valid for the
    benchmark and for real data alike."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    val_groups = set(rng.choice(uniq, size=max(1, int(share * len(uniq))), replace=False))
    val = np.isin(groups, list(val_groups))
    losses = {}
    for lam in grid:
        m = PULogistic(c=1.0, lam=lam).fit(X[~val], s[~val], a[~val])
        p = np.clip(m.predict_proba(X[val]), 1e-12, 1 - 1e-12)
        losses[lam] = float(-np.average(s[val] * np.log(p) + (1 - s[val]) * np.log(1 - p), weights=a[val]))
    return min(losses, key=losses.get), losses


def _weighted_log_loss(s, p, a) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.average(s * np.log(p) + (1 - s) * np.log(1 - p), weights=a))


def select_lambda_kfold(X, s, a, groups, grid=LAMBDA_GRID, n_inner=3, seed=0) -> tuple[float, dict, dict]:
    """λ by grouped K-fold on training pairs (groups: source neurons), with the one-standard-error
    rule: the largest λ whose mean held-out log-loss for s is within one standard error (over
    folds) of the smallest mean. A steadier choice than `select_lambda` where the loss is flat
    below the optimum and rises steeply past it. Returns (λ, mean loss per λ, SE per λ)."""
    uniq = np.unique(groups)
    if n_inner < 2 or n_inner > len(uniq):
        raise ValueError("n_inner must lie in [2, number of groups]")
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = pd.Series(np.empty(len(uniq), int), index=uniq)
    fold_of.iloc[order] = np.arange(len(uniq)) % n_inner
    fold = fold_of.reindex(groups).to_numpy()
    per = {lam: [] for lam in grid}
    for k in range(n_inner):
        val = fold == k
        for lam in grid:
            m = PULogistic(c=1.0, lam=lam).fit(X[~val], s[~val], a[~val])
            per[lam].append(_weighted_log_loss(s[val], m.predict_proba(X[val]), a[val]))
    mean = {lam: float(np.mean(v)) for lam, v in per.items()}
    se = {lam: float(np.std(v, ddof=1) / np.sqrt(n_inner)) for lam, v in per.items()}
    best = min(mean, key=mean.get)
    return max(lam for lam in grid if mean[lam] <= mean[best] + se[best]), mean, se


def evaluation_pairs(sample, pos_true, still, test_sources) -> pd.DataFrame:
    """The sampled pairs with every deleted connection of a held-out source added exhaustively.

    Deleted connections (true positives no longer ≥ τ after thinning) of held-out sources
    get weight 1 and stratum "deleted"; any of them already in the sample is removed from
    the sampled part, so none is counted twice. Every other unlabelled pair keeps its
    Bernoulli(ρ) inclusion and weight 1/ρ, and recorded positives keep weight 1, so
    weighted totals are Horvitz–Thompson estimates of the held-out population."""
    deleted = pos_true.index[(~still.to_numpy()) & pos_true.index.get_level_values(0).isin(test_sources)]
    key = pd.MultiIndex.from_arrays([sample["pre_id"], sample["post_id"]])
    return pd.concat([sample[~key.isin(deleted)],
                      pd.DataFrame({"pre_id": deleted.get_level_values(0), "post_id": deleted.get_level_values(1),
                                    "syn": 0, "s": 0, "stratum": "deleted", "weight": 1.0})], ignore_index=True)


def reliability(p, y, w, bins: int = 10) -> list[dict]:
    """Weighted reliability table: pairs sorted by predicted p and cut into `bins` groups of
    approximately equal total weight (pairs are indivisible: a pair goes to the group where
    its cumulative weight starts, so a heavy pair can leave a group empty); per group the
    weighted mean prediction and the weighted true rate."""
    order = np.argsort(p)
    start = (np.cumsum(w[order]) - w[order]) / w.sum()          # cumulative weight before each pair
    group = np.minimum((start * bins + 1e-12).astype(int), bins - 1)
    out = []
    for g in range(bins):
        m = order[group == g]
        if len(m):
            out.append({"bin": g, "weight": float(w[m].sum()), "mean_p": float(np.average(p[m], weights=w[m])),
                        "rate_y": float(np.average(y[m], weights=w[m]))})
    return out


def homologue_groups(roster: pd.DataFrame) -> pd.Series:
    """Worksheet D18: cell_type across sides is one group; an untyped neuron, or one on the
    center/unknown side, is a group of its own. Indexed like `roster`."""
    own = roster["cell_type"].isna() | ~roster["side"].isin(["left", "right"])
    return roster["cell_type"].astype(object).where(~own, "neuron:" + roster["neuron_id"].astype(str))


def fold_of_groups(groups: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """Each distinct group to one of `n_folds` folds, in a seeded random order (sizes differ by ≤ 1 group)."""
    uniq = np.sort(groups.unique())
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold = pd.Series(np.empty(len(uniq), int), index=uniq)
    fold.iloc[order] = np.arange(len(uniq)) % n_folds
    return groups.map(fold)


STRATA = [(0, 0, "0"), (1, 9, "1–9"), (10, 19, "10–19"), (20, 49, "20–49"), (50, np.inf, "50+")]


def run_condition(neurons, edgeroi_roster, ids, q, mode="synapse", seed=0, tau=10, per_positive=10,
                  lam=1e-2, test_share=0.2, exclude=(), fold=None, n_folds=5, target_graph="full", keep_scores=False, clip=False) -> dict:
    """One benchmark condition; `edgeroi_roster` holds the rows touching the roster.
    `lam="cv"` chooses λ with `select_lambda`, `lam="cv3"` with `select_lambda_kfold` (3 folds, 1-SE rule); `exclude` drops feature columns (e.g. "type_i").
    With `fold` set, the held-out sources are homologue-group fold `fold` of `n_folds`
    (seeded by `seed`), so the folds of one seed cover every roster neuron once; otherwise
    `test_share` of the cell types (untyped neurons as one type) is held out.
    `target_graph` is passed to `split_features` ("train": held-out target features on the training graph).
    `clip=True` limits held-out numeric features to the training range (`design`).
    `keep_scores=True` adds "scores", a DataFrame of the evaluation pairs (pre_id, post_id, y,
    weight, s, f_<model>) for diagnostics; drop it before writing JSON.

    Evaluation set (held-out sources): every deleted connection (true positive left
    unlabelled by the thinning) with weight 1, plus the sampled held-out pairs that are
    not deleted connections with their sampling weights. Unlabelled non-deleted pairs
    are then still a Bernoulli(ρ) sample, so weighted means estimate population values,
    and nothing rests on the few deleted connections that happen to be sampled.
    """
    if not 0 < q <= 1:                                        # also refuses NaN
        raise ValueError("q must lie in (0, 1]")
    truth = region_pair_counts(edgeroi_roster, ids)
    er = thin_rows(edgeroi_roster, q, mode, seed) if q < 1 else edgeroi_roster
    seen = region_pair_counts(er, ids)
    pos_true = truth[truth >= tau]
    still = seen.reindex(pos_true.index, fill_value=0) >= tau
    realised_c = float(still.mean())

    sample = sample_pairs(ids, seen, tau=tau, per_positive=per_positive, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    roster = neurons[neurons["neuron_id"].isin(ids)]
    if fold is None:
        rtypes = roster["cell_type"].fillna("untyped")
        types = np.sort(rtypes.unique())
        test_types = set(rng.choice(types, size=int(test_share * len(types)), replace=False))
        test_sources = pd.Index(roster.loc[rtypes.isin(test_types).to_numpy(), "neuron_id"])
    else:
        if not 0 <= fold < n_folds:
            raise ValueError("fold must lie in [0, n_folds)")
        in_fold = fold_of_groups(homologue_groups(roster), n_folds, seed).to_numpy() == fold
        test_sources = pd.Index(roster.loc[in_fold, "neuron_id"])
    c_train = float(still[~pos_true.index.get_level_values(0).isin(test_sources)].mean())

    pairs = evaluation_pairs(sample, pos_true, still, test_sources)
    er_edges = edges_from_rows(er)
    f_tr, f_te, test = split_features(neurons, er_edges, er, pairs, ids, test_sources, tau=tau,
                                       target_graph=target_graph)
    pkey = pd.MultiIndex.from_arrays([pairs["pre_id"], pairs["post_id"]])
    true_syn = truth.reindex(pkey, fill_value=0).to_numpy()
    y = (true_syn >= tau).astype(float)
    s, a = pairs["s"].to_numpy(float), pairs["weight"].to_numpy()
    columns = feature_columns(f_tr, GROUPS)
    if set(exclude) - set(columns):
        raise ValueError(f"unknown feature columns: {sorted(set(exclude) - set(columns))}")
    columns = [c for c in columns if c not in exclude]
    Xtr, Xte, _ = design(f_tr, f_te, columns, clip=clip)
    lam_losses, lam_se = None, None
    if lam == "cv":
        lam, lam_losses = select_lambda(Xtr, s[~test], a[~test], pairs.loc[~test, "pre_id"].to_numpy(), seed=seed)
    elif lam == "cv3":
        lam, lam_losses, lam_se = select_lambda_kfold(Xtr, s[~test], a[~test], pairs.loc[~test, "pre_id"].to_numpy(),
                                                      n_inner=3, seed=seed)

    # Elkan–Noto e1: naive model on 80% of training source neurons, mean over the other 20%'s positives
    tr_src = pairs.loc[~test, "pre_id"].to_numpy()
    src_ids = np.unique(tr_src)
    inner_src = set(rng.choice(src_ids, size=int(0.8 * len(src_ids)), replace=False))
    inner = np.isin(tr_src, list(inner_src))
    s_tr, a_tr = s[~test], a[~test]
    g = PULogistic(c=1.0, lam=lam).fit(Xtr[inner], s_tr[inner], a_tr[inner])
    held_pos = (~inner) & (s_tr == 1)
    e1 = elkan_noto_c(g.predict_proba(Xtr[held_pos]))["e1"] if held_pos.any() else float("nan")
    p_floor = float(s_tr @ a_tr / a_tr.sum())                    # c must exceed P(s=1)

    yt, at, st = y[test], a[test], true_syn[test]
    is_del = (pairs["stratum"].to_numpy()[test] == "deleted")
    unl = s[test] == 0
    out = {"q": q, "mode": mode, "seed": seed, "realised_c": realised_c, "realised_c_train": c_train,
           "e1": e1, "e1_clipped": bool(e1 < 1.01 * p_floor), "p_s_train": p_floor,
           "lambda": lam, "lambda_cv_losses": lam_losses, "lambda_cv_se": lam_se, "excluded": list(exclude),
           "fold": fold, "n_folds": n_folds if fold is not None else None, "test_sources": int(len(test_sources)),
           "target_graph": target_graph, "clip": clip,
           "positives_seen": int(sample.attrs["positives"]),
           "positives_true": int(len(pos_true)), "eval_pairs": int(test.sum()), "deleted_in_eval": int(is_del.sum()),
           "models": {}}
    fitted = {}
    for name, c in [("naive", 1.0), ("pu_true_c", c_train), ("pu_e1", max(e1, 1.01 * p_floor))]:
        m = PULogistic(c=min(c, 1.0), lam=lam).fit(Xtr, s_tr, a_tr)
        f = m.predict_proba(Xte)
        fitted[name] = (m, f)
        # top-1% threshold: the weighted 99th percentile of the unlabelled scores; scores tied with
        # the threshold are included, so with ties the selected share can exceed 1%
        order = np.argsort(-f[unl])
        cw = np.cumsum(at[unl][order]) / at[unl].sum()
        thr = f[unl][order][min(np.searchsorted(cw, 0.01), len(order) - 1)]
        by_stratum = {}
        for lo, hi, lab in STRATA:
            msk = (st >= lo) & (st <= hi)
            if msk.any():
                by_stratum[lab] = {"weight": float(at[msk].sum()), "mean_f": float(np.average(f[msk], weights=at[msk])),
                                   "mean_y": float(np.average(yt[msk], weights=at[msk]))}
        out["models"][name] = {"c": float(min(c, 1.0)), "pi": m.pi_, "converged": m.converged_, **score(yt, f, at),
                               "reliability": reliability(f, yt, at),
                               "recall_deleted_top1pct": float((f[is_del] >= thr).mean()) if is_del.any() else float("nan"),
                               "mean_f_deleted": float(f[is_del].mean()) if is_del.any() else float("nan"),
                               "by_true_stratum": by_stratum}
    # H3: does PU at the true c keep the naive ranking of the held-out unlabelled pairs?
    (mn, fn), (mp, fp) = fitted["naive"], fitted["pu_true_c"]
    out["h3_naive_vs_pu_true_c"] = ranking_agreement(fn[unl], fp[unl], mn.coef_, mp.coef_, ks=(100, 1000))
    if keep_scores:
        out["scores"] = pd.DataFrame({"pre_id": pairs.loc[test, "pre_id"].to_numpy(), "post_id": pairs.loc[test, "post_id"].to_numpy(),
                                      "y": yt, "weight": at, "s": s[test], **{f"f_{k}": v[1] for k, v in fitted.items()}})
    return out
