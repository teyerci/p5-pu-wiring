"""Edges per neuron on the cell types present in both datasets (lock note addendum A7, descriptive)."""
import numpy as np
import pandas as pd


def per_neuron(ids, pair_syn: pd.Series, tau: int) -> pd.Series:
    """Outgoing partners with >= tau synapses inside `ids` (zeros kept), indexed by neuron id."""
    ids = pd.Index(ids)
    s = pair_syn[(pair_syn >= tau) & pair_syn.index.get_level_values(0).isin(ids) & pair_syn.index.get_level_values(1).isin(ids)]
    return s.groupby(level=0).size().reindex(ids, fill_value=0)


def compare(ids_f, key_f, ps_f, ids_m, key_m, ps_m, tau_f: int, tau_ms: dict) -> dict:
    """Shared types = keys on both sides; the population on each side is the neurons of those types.
    `key_*` is indexed by neuron id; `tau_ms` maps a setting name to the other dataset's tau."""
    kf, km = pd.Series(key_f).dropna(), pd.Series(key_m).dropna()
    kf, km = kf[kf.index.isin(ids_f)], km[km.index.isin(ids_m)]
    shared = sorted(set(kf) & set(km))
    kf, km = kf[kf.isin(shared)], km[km.isin(shared)]
    f = per_neuron(kf.index, ps_f, tau_f)
    out = {"shared_types": len(shared), "neurons_fafb": len(kf), "neurons_malecns": len(km), "tau_fafb": tau_f,
           "fafb": {"mean": float(f.mean()), "median": float(f.median())}, "settings": {}}
    for name, tau in tau_ms.items():
        m = per_neuron(km.index, ps_m, tau)
        tf = f.groupby(kf.reindex(f.index).to_numpy()).agg(["size", "mean"])
        tm = m.groupby(km.reindex(m.index).to_numpy()).agg(["size", "mean"])
        out["settings"][name] = {
            "tau_malecns": tau, "mean": float(m.mean()), "median": float(m.median()),
            "ratio_mean": float(m.mean() / f.mean()) if f.mean() > 0 else float("nan"),
            # type-weighted: each shared type counts once, so differing type sizes do not decide the ratio
            "type_weighted": {"mean_fafb": float(tf["mean"].mean()), "mean_malecns": float(tm.reindex(shared)["mean"].mean()),
                              "median_type_ratio": float((tm.reindex(shared)["mean"] / tf.reindex(shared)["mean"]).replace(
                                  [float("inf")], float("nan")).median())},
            "per_type": [{"type": t, "n_fafb": int(tf.loc[t, "size"]), "n_malecns": int(tm.loc[t, "size"]),
                          "mean_fafb": float(tf.loc[t, "mean"]), "mean_malecns": float(tm.loc[t, "mean"])} for t in shared]}
    return out
