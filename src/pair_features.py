"""Pair features for P5, computed leave-pair-out (proposal §4, "Features").

For a candidate pair (i, j) every connectivity feature is computed after removing
the i → j synapses themselves, so a feature never contains the target pair's own
synapses (the label). The same code runs on a thinned graph in the dropout
benchmark, because every feature is a pure function of the edge tables.

Groups (columns):
  identity      sub_i, sub_j (cell_sub_class, "missing" as its own level), nt_i,
                same_type, same_side
  cell type     type_i, type_j (one-hot later, separately for i and j)
  region        frac_out_<np> (i's outputs), frac_in_<np> (j's inputs) over all
                partners and named neuropils, after removing i → j; overlap =
                Σ frac_out · frac_in; no_out / no_in flag a neuron left with no
                synapses (its fractions are then 0, D16-style "null" coded as 0 + flag)
  degree        out_deg_i, in_deg_j at τ over all partners (edges summed over
                neuropils), minus the pair itself if i → j reaches τ
  spatial       log1p(Σ_np out_i(np) · in_j(np)) on leave-pair-out counts, a proxy
                for cable overlap

These are P5's own pair features; the shared per-neuron feature table
(features.py) is written by the group after the week 3 session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda_e import real_neuropil

CATEGORICAL = ["sub_i", "sub_j", "nt_i", "type_i", "type_j"]


def neuropil_counts(edgeroi: pd.DataFrame, ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per roster neuron, output and input synapses per named neuropil, over all partners."""
    er = edgeroi[real_neuropil(edgeroi["neuropil"])]
    out = er[er["pre_id"].isin(ids)].groupby(["pre_id", "neuropil"])["syn_count"].sum().unstack(fill_value=0)
    inp = er[er["post_id"].isin(ids)].groupby(["post_id", "neuropil"])["syn_count"].sum().unstack(fill_value=0)
    cols = sorted(set(out.columns) | set(inp.columns))
    ids = pd.Index(ids)
    return (out.reindex(index=ids, columns=cols, fill_value=0).astype("float64"),
            inp.reindex(index=ids, columns=cols, fill_value=0).astype("float64"))


def _own_synapses(edgeroi, ids, key, cols) -> np.ndarray:
    """The pairs' own synapses per named neuropil, one row per pair in `key` order."""
    er = edgeroi[real_neuropil(edgeroi["neuropil"])]
    er = er[er["pre_id"].isin(ids) & er["post_id"].isin(ids)]
    row = pd.Series(np.arange(len(key)), index=key)
    er = er[pd.MultiIndex.from_arrays([er["pre_id"], er["post_id"]]).isin(key)]
    E = np.zeros((len(key), len(cols)))
    if len(er):
        r = row.reindex(pd.MultiIndex.from_arrays([er["pre_id"], er["post_id"]])).to_numpy()
        c = pd.Index(cols).get_indexer(er["neuropil"].to_numpy())
        np.add.at(E, (r, c), er["syn_count"].to_numpy())
    return E


def _strong_degree(edges, ids, key, tau, level):
    """Partners at ≥ τ (edges summed over neuropils) per roster neuron, and whether each pair is one."""
    tot = edges.groupby(["pre_id", "post_id"])["syn_count"].sum()
    strong = tot[tot >= tau]
    return (strong.groupby(level=level).size().reindex(ids, fill_value=0).to_numpy(),
            strong.reindex(key).notna().to_numpy().astype(int))


def build_pair_features(neurons: pd.DataFrame, edges: pd.DataFrame, edgeroi: pd.DataFrame,
                        pairs: pd.DataFrame, ids, tau: int = 10,
                        edges_in: pd.DataFrame | None = None, edgeroi_in: pd.DataFrame | None = None) -> pd.DataFrame:
    """Leave-pair-out features for `pairs` (pre_id, post_id; e.g. src.pairs.sample_pairs).

    Source-side features (i's outputs) come from `edges`/`edgeroi`; target-side ones (j's
    inputs) from `edges_in`/`edgeroi_in`, by default the same graph. `split_features`
    uses the second pair so held-out target features match the training graph."""
    edges_in = edges if edges_in is None else edges_in
    edgeroi_in = edgeroi if edgeroi_in is None else edgeroi_in
    ids = pd.Index(np.sort(np.asarray(ids, dtype="int64")))
    O, I = neuropil_counts(edgeroi, ids)
    if edgeroi_in is not edgeroi:
        _, I = neuropil_counts(edgeroi_in, ids)
    cols = sorted(set(O.columns) | set(I.columns))
    O, I = O.reindex(columns=cols, fill_value=0.0), I.reindex(columns=cols, fill_value=0.0)
    pi = ids.get_indexer(pairs["pre_id"].to_numpy())
    pj = ids.get_indexer(pairs["post_id"].to_numpy())
    if (pi < 0).any() or (pj < 0).any():
        raise ValueError("every pair must be inside the roster")

    # the pair's own synapses per neuropil, removed on each side from that side's graph
    key = pd.MultiIndex.from_arrays([pairs["pre_id"].to_numpy(), pairs["post_id"].to_numpy()])
    E = _own_synapses(edgeroi, ids, key, cols)
    E_in = E if edgeroi_in is edgeroi else _own_synapses(edgeroi_in, ids, key, cols)

    out_lpo = O.to_numpy()[pi] - E
    in_lpo = I.to_numpy()[pj] - E_in
    so, si = out_lpo.sum(1, keepdims=True), in_lpo.sum(1, keepdims=True)
    frac_out = np.divide(out_lpo, so, out=np.zeros_like(out_lpo), where=so > 0)
    frac_in = np.divide(in_lpo, si, out=np.zeros_like(in_lpo), where=si > 0)

    # degrees at τ over all partners, minus the pair itself
    outdeg, self_out = _strong_degree(edges, ids, key, tau, level=0)
    indeg, self_in = _strong_degree(edges_in, ids, key, tau, level=1)

    nn = neurons.set_index("neuron_id").reindex(ids)
    sub = nn["cell_sub_class"].astype("object").where(nn["cell_sub_class"].notna(), "missing").to_numpy()
    known = nn["cell_type"].notna().to_numpy()
    ctype = nn["cell_type"].astype("object").where(nn["cell_type"].notna(), "untyped").to_numpy()
    side = nn["side"].astype(str).to_numpy()
    feats = pd.DataFrame({
        "pre_id": pairs["pre_id"].to_numpy(), "post_id": pairs["post_id"].to_numpy(),
        "sub_i": sub[pi], "sub_j": sub[pj], "nt_i": nn["nt_type"].astype(str).to_numpy()[pi],
        "type_i": ctype[pi], "type_j": ctype[pj],
        "same_type": ((ctype[pi] == ctype[pj]) & known[pi] & known[pj]).astype(int), "same_side": (side[pi] == side[pj]).astype(int),
        "out_deg_i": outdeg[pi] - self_out, "in_deg_j": indeg[pj] - self_in,
        "no_out": (so[:, 0] == 0).astype(int), "no_in": (si[:, 0] == 0).astype(int),
        "overlap": (frac_out * frac_in).sum(1),
        "spatial": np.log1p((out_lpo * in_lpo).sum(1)),
    })
    region = pd.DataFrame(np.hstack([frac_out, frac_in]),
                          columns=[f"frac_out_{c}" for c in cols] + [f"frac_in_{c}" for c in cols])
    return pd.concat([feats, region], axis=1)


GROUPS = {
    "identity": ["sub_i", "sub_j", "nt_i", "same_type", "same_side"],
    "cell_type": ["type_i", "type_j"],
    "degree": ["out_deg_i", "in_deg_j"],
    "spatial": ["spatial"],
}


def feature_columns(feats: pd.DataFrame, groups) -> list[str]:
    cols = []
    for g in groups:
        if g == "region":
            cols += [c for c in feats.columns if c.startswith(("frac_out_", "frac_in_"))] + ["overlap", "no_out", "no_in"]
        else:
            cols += GROUPS[g]
    return cols


def design(train: pd.DataFrame, test: pd.DataFrame | None, columns: list[str]):
    """Numeric design matrices: one-hot categoricals (levels from train; unseen levels
    in test become all-zero), numerics standardised with train statistics (fold-local)."""
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    levels = {c: sorted(train[c].astype(str).unique()) for c in cat}
    mu = train[num].mean()
    sd = train[num].std().replace(0, 1.0).fillna(1.0)

    def build(df):
        parts = [((df[num] - mu) / sd).to_numpy(dtype=float)]
        names = list(num)
        for c in cat:
            v = df[c].astype(str).to_numpy()
            parts.append((v[:, None] == np.asarray(levels[c])[None, :]).astype(float))
            names += [f"{c}={lv}" for lv in levels[c]]
        return np.hstack(parts), names

    Xtr, names = build(train)
    Xte = build(test)[0] if test is not None else None
    return Xtr, Xte, names


def split_features(neurons, edges, edgeroi, pairs, ids, test_sources, tau: int = 10, target_graph: str = "full"):
    """Features for a split by source neuron, with no held-out synapse in training.

    Held-out candidates are all pairs (i, j) with i in `test_sources` and j in the
    roster. Training features are built on a graph from which every such pair's
    synapses are removed (all neuropils), so neither the training rows nor the
    standardisation they feed can carry a held-out label. Held-out features are
    built on the full graph, leave-pair-out, as at prediction time (the neuron's
    other edges are known). With `target_graph="train"` their target-side features
    (j's inputs) come from the training graph instead: training targets have lost the
    held-out sources' inputs, so full-graph target features would sit outside the
    training distribution. Returns (train features, test features, test mask).
    """
    if target_graph not in ("full", "train"):
        raise ValueError("target_graph must be 'full' or 'train'")
    test_sources, ids = pd.Index(test_sources), pd.Index(ids)
    test = pairs["pre_id"].isin(test_sources).to_numpy()
    drop_er = edgeroi["pre_id"].isin(test_sources) & edgeroi["post_id"].isin(ids)
    drop_e = edges["pre_id"].isin(test_sources) & edges["post_id"].isin(ids)
    train = build_pair_features(neurons, edges[~drop_e], edgeroi[~drop_er], pairs[~test].reset_index(drop=True), ids, tau)
    kw = {"edges_in": edges[~drop_e], "edgeroi_in": edgeroi[~drop_er]} if target_graph == "train" else {}
    held = build_pair_features(neurons, edges, edgeroi, pairs[test].reset_index(drop=True), ids, tau, **kw)
    return train, held, test
