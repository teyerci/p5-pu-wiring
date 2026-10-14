"""Hemibrain type-level check of high-scoring unrecorded fafb pairs (proposal H4).

A fafb pair (i, j) on one side is mapped to its type pair (A, B) through fafb's
`hemibrain_type` (compound names merged into match groups, src.eda_e.match_groups). In the
hemibrain, the type pair is "recorded at τ" if some neuron pair (a in A, b in B) on the same
side has at least τ synapses from a onto b in the central-complex neuropils (the definition of
src.discrepancy). The hit rate of the top-k unrecorded pairs is compared with control pairs
matched on strata (direct standardisation: each candidate stratum's control hit rate,
weighted by the number of candidates in it).

Corroborative only: a hemibrain hit does not confirm a single missed fafb connection, and a
miss can be biological variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.discrepancy import status, type_pair_table
from src.eda_e import group_key, match_groups
from src.pairs import region_pair_counts


def type_keys(fafb_neurons, other_neurons, fafb_col="hemibrain_type", other_col="cell_type"):
    """Match-group keys for both copies (NaN where a neuron has no usable type)."""
    groups = match_groups(fafb_neurons[fafb_col], other_neurons[other_col])
    return group_key(fafb_neurons[fafb_col], groups), group_key(other_neurons[other_col], groups)


def other_type_table(other_neurons, other_edgeroi, other_key, keep_groups, side="right") -> pd.DataFrame:
    """Type-pair table (max neuron-pair count, total) of the other copy on one side,
    over neurons whose group is in `keep_groups`."""
    keep = other_key.isin(keep_groups).to_numpy()
    on, ko = other_neurons[keep].reset_index(drop=True), other_key[keep].reset_index(drop=True)
    return type_pair_table(on, region_pair_counts(other_edgeroi, on["neuron_id"]), ko, side)


def pair_type_status(pre, post, key_by_id: pd.Series, table: pd.DataFrame, tau: int) -> pd.Series:
    """Status of each pair's type pair in `table` ("recorded", "below_tau", "absent"), NaN where
    either neuron has no key. `key_by_id` maps neuron_id -> group key."""
    a = key_by_id.reindex(np.asarray(pre)).to_numpy()
    b = key_by_id.reindex(np.asarray(post)).to_numpy()
    ok = pd.notna(a) & pd.notna(b)
    out = pd.Series(np.nan, index=range(len(a)), dtype=object)
    if ok.any():
        idx = pd.MultiIndex.from_arrays([a[ok], b[ok]])
        out[np.flatnonzero(ok)] = status(table, idx, tau).to_numpy()
    return out


def matched_hit_rate(hit, candidate, strata, weight=None) -> dict:
    """Candidate hit rate and the control rate standardised to the candidates' strata.

    `hit` (bool), `candidate` (bool) and `strata` (hashable labels) are aligned over the
    eligible pairs; controls are the non-candidates. Candidates in a stratum without any
    control are dropped from both rates (reported as `candidates_unmatched`). `weight`
    (default 1) counts each pair that many times, as in a bootstrap replicate; pairs of
    weight 0 are absent."""
    w_all = np.ones(len(hit)) if weight is None else np.asarray(weight, dtype=float)
    df = pd.DataFrame({"hit": np.asarray(hit, dtype=float), "cand": np.asarray(candidate, dtype=bool),
                       "stratum": list(strata), "w": w_all})
    df = df[df["w"] > 0]
    df["wh"] = df["w"] * df["hit"]
    ctrl = df[~df["cand"]].groupby("stratum")[["wh", "w"]].sum()
    cand = df[df["cand"]]
    matched = cand["stratum"].isin(ctrl.index)
    cm = cand[matched]
    w = cm.groupby("stratum")["w"].sum()
    rate = ctrl["wh"] / ctrl["w"]
    return {"candidates": int(round(cand["w"].sum())), "candidates_unmatched": int(round(cand.loc[~matched, "w"].sum())),
            "candidate_hit_rate": float(cm["wh"].sum() / cm["w"].sum()) if len(cm) else float("nan"),
            "control_hit_rate": float((rate[w.index] * w).sum() / w.sum()) if len(w) else float("nan"),
            "controls_used": int(round(ctrl.loc[w.index, "w"].sum())) if len(w) else 0}


def cluster_bootstrap(hit, candidate, strata, clusters, n_boot=500, seed=0) -> dict:
    """Percentile intervals for the candidate and control hit rates and their ratio, resampling
    whole clusters (e.g. the source neuron's type group) with replacement; every pair of a
    drawn cluster enters with the number of times its cluster was drawn. Candidate membership
    is fixed (the intervals condition on the original selection); replicates with a zero or
    undefined control rate are not used (`valid` counts the rest; all NaN if none)."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    cand_r, ctrl_r = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        r = matched_hit_rate(hit, candidate, strata, weight=counts[inv])
        cand_r.append(r["candidate_hit_rate"])
        ctrl_r.append(r["control_hit_rate"])
    cand_r, ctrl_r = np.array(cand_r), np.array(ctrl_r)
    ok = np.isfinite(cand_r) & np.isfinite(ctrl_r) & (ctrl_r > 0)
    if not ok.any():                                       # e.g. no control hit in any replicate
        nan2 = [float("nan"), float("nan")]
        return {"n_boot": n_boot, "valid": 0, "clusters": int(len(uniq)), "candidate_ci": nan2,
                "control_ci": nan2, "ratio_ci": nan2, "share_candidate_above_control": float("nan")}
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"n_boot": n_boot, "valid": int(ok.sum()), "clusters": int(len(uniq)),
            "candidate_ci": q(cand_r[ok]), "control_ci": q(ctrl_r[ok]), "ratio_ci": q(cand_r[ok] / ctrl_r[ok]),
            "share_candidate_above_control": float((cand_r[ok] > ctrl_r[ok]).mean())}
