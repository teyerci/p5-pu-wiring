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

GROUPS = ["identity", "cell_type", "region", "degree", "spatial"]


def thin_rows(edgeroi: pd.DataFrame, q: float, mode: str = "synapse", seed: int = 0) -> pd.DataFrame:
    """Thinned edge-neuropil rows (rows left with 0 synapses dropped)."""
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


def run_condition(neurons, edgeroi_roster, ids, q, mode="synapse", seed=0, tau=10, per_positive=10,
                  lam=1e-2, test_share=0.2) -> dict:
    """One benchmark condition; `edgeroi_roster` holds the rows touching the roster."""
    truth = region_pair_counts(edgeroi_roster, ids)
    er = thin_rows(edgeroi_roster, q, mode, seed) if q < 1 else edgeroi_roster
    seen = region_pair_counts(er, ids)
    pos_true = truth[truth >= tau]
    realised_c = float((seen.reindex(pos_true.index, fill_value=0) >= tau).mean())

    sample = sample_pairs(ids, seen, tau=tau, per_positive=per_positive, seed=seed)
    key = pd.MultiIndex.from_arrays([sample["pre_id"], sample["post_id"]])
    y = (truth.reindex(key, fill_value=0) >= tau).to_numpy().astype(float)
    # split first: held-out source types; training features never see a held-out pair's synapses
    rng = np.random.default_rng(1000 + seed)
    roster = neurons[neurons["neuron_id"].isin(ids)]
    rtypes = roster["cell_type"].fillna("untyped")
    types = np.sort(rtypes.unique())
    test_types = set(rng.choice(types, size=int(test_share * len(types)), replace=False))
    test_sources = roster.loc[rtypes.isin(test_types).to_numpy(), "neuron_id"]
    f_tr, f_te, test = split_features(neurons, edges_from_rows(er), er, sample, ids, test_sources, tau=tau)
    s, a = sample["s"].to_numpy(float), sample["weight"].to_numpy()
    Xtr, Xte, _ = design(f_tr, f_te, feature_columns(f_tr, GROUPS))

    # Elkan–Noto e1 from a naive model fitted on 80% of the training pairs, averaged over the rest's positives
    tr_idx = np.flatnonzero(~test)
    inner = rng.random(len(tr_idx)) < 0.8
    g = PULogistic(c=1.0, lam=lam).fit(Xtr[inner], s[tr_idx][inner], a[tr_idx][inner])
    held_pos = (~inner) & (s[tr_idx] == 1)
    e1 = elkan_noto_c(g.predict_proba(Xtr[held_pos]))["e1"]

    out = {"q": q, "mode": mode, "seed": seed, "realised_c": realised_c, "e1": e1,
           "positives_seen": int(sample.attrs["positives"]), "positives_true": int(len(pos_true)),
           "test_pairs": int(test.sum()), "models": {}}
    p_floor = s[~test] @ a[~test] / a[~test].sum()               # c must exceed P(s=1)
    unl = s[test] == 0
    deleted = unl & (y[test] == 1)                                # true connections the thinning removed
    for name, c in [("naive", 1.0), ("pu_true_c", realised_c), ("pu_e1", max(e1, 1.01 * p_floor))]:
        m = PULogistic(c=min(c, 1.0), lam=lam).fit(Xtr, s[~test], a[~test])
        f = m.predict_proba(Xte)
        k = max(1, int(0.01 * unl.sum()))
        top = np.zeros(unl.sum(), bool)
        top[np.argsort(-f[unl])[:k]] = True
        out["models"][name] = {"c": float(min(c, 1.0)), "converged": m.converged_,
                               **score(y[test], f, a[test]),
                               "deleted_in_test": int(deleted.sum()),
                               "recall_deleted_top1pct": float(top[deleted[unl]].mean()) if deleted.any() else float("nan")}
    return out
