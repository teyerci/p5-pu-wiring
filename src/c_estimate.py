"""The lock note's c proxy: mirror recurrence of labels at neuron-pair level.

In cell types with exactly one neuron on each side, every neuron has a known mirror.
Of the pairs recorded (≥ τ) among these neurons, the share whose mirror pair is also
recorded estimates c = P(recorded | connected), if both sides share one true wiring and
are recorded independently (lock note, c estimate; addendum A3). The same computation
as `mirror_pairs` in scripts/lock_note_numbers.py, as a function a fold can call on its
training sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mirror_map(roster: pd.DataFrame) -> pd.Series:
    """neuron_id -> its mirror, for the neurons of types with one neuron on each side."""
    typed = roster[roster["cell_type"].notna() & roster["side"].isin(["left", "right"])]
    n = typed.groupby(["cell_type", "side"], observed=True).size().unstack(fill_value=0)
    for side in ("left", "right"):
        if side not in n:
            n[side] = 0
    one = typed[typed["cell_type"].isin(n.index[(n["left"] == 1) & (n["right"] == 1)])]
    out = {}
    for _, g in one.groupby("cell_type", observed=True):
        a, b = g["neuron_id"].tolist()
        out[a], out[b] = b, a
    return pd.Series(out, dtype="int64")


def mirror_share(pair_syn: pd.Series, mirror: pd.Series, tau: int, sources=None) -> dict:
    """Share of recorded pairs among mirrored neurons whose mirror pair is recorded too.
    `pair_syn` is indexed by (pre_id, post_id); `sources`, if given, keeps only pairs whose
    source neuron is in it (a fold's training sources)."""
    kept = pair_syn[pair_syn >= tau]
    pre, post = kept.index.get_level_values(0), kept.index.get_level_values(1)
    keep = pre.isin(mirror.index) & post.isin(mirror.index)
    if sources is not None:
        keep &= pre.isin(pd.Index(sources))
    kept = kept[keep]
    other = pd.MultiIndex.from_arrays([kept.index.get_level_values(0).map(mirror), kept.index.get_level_values(1).map(mirror)])
    hit = pair_syn.reindex(other).fillna(0).to_numpy() >= tau
    return {"share": float(hit.mean()) if len(hit) else float("nan"), "pairs": int(len(hit))}


def mirror_share_interval(pair_syn: pd.Series, mirror: pd.Series, tau: int, source_type: pd.Series,
                          sources=None, b: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of `mirror_share` from a bootstrap over source cell types (the lock note's
    unit for this estimate). `source_type` maps neuron_id -> cell type."""
    kept = pair_syn[pair_syn >= tau]
    pre, post = kept.index.get_level_values(0), kept.index.get_level_values(1)
    keep = pre.isin(mirror.index) & post.isin(mirror.index)
    if sources is not None:
        keep &= pre.isin(pd.Index(sources))
    kept = kept[keep]
    other = pd.MultiIndex.from_arrays([kept.index.get_level_values(0).map(mirror), kept.index.get_level_values(1).map(mirror)])
    hit = pair_syn.reindex(other).fillna(0).to_numpy() >= tau
    if not len(hit):
        return float("nan"), float("nan")
    t = source_type.reindex(kept.index.get_level_values(0)).to_numpy()
    types = np.unique(t)
    n = np.array([(t == k).sum() for k in types])
    h = np.array([hit[t == k].sum() for k in types])
    draw = np.random.default_rng(seed).integers(0, len(types), size=(b, len(types)))
    share = h[draw].sum(1) / n[draw].sum(1)
    return float(np.quantile(share, 0.025)), float(np.quantile(share, 0.975))
