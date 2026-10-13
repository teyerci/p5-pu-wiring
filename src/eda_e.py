"""EDA workstream E — regions, completeness and discrepancy (Week 3 EDA Plan).

Each item has a compute function (returns tables and the numbers the report
quotes) and a plot function (writes figures/eda/<item>_<dataset>_<release>_<population>_min<τ>.png).
Every function takes one dataset's tables, built to the Week 2 Data Conventions.
Neuropil names keep FlyWire's side suffix (ME_L, ME_R) unless a plot says otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_shared.eda_helpers import add_provisional_flags, load_neuropil_table

OPTIC = {"LA", "ME", "AME", "LO", "LOP"}
CX_NEUROPILS = ["PB", "FB", "EB", "NO"]
NOT_A_NEUROPIL = {"UNASGD", "None", "unknown"}
WEIGHT_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"),
               (10, 19, "10–19"), (20, 49, "20–49"), (50, np.inf, "50+")]


def base_neuropil(names: pd.Series) -> pd.Series:
    return names.astype("string").str.replace(r"(_[LR]|\([LR]\))$", "", regex=True)


def flags(neurons: pd.DataFrame) -> pd.DataFrame:
    """neurons with is_optic, region_group, neuropil (base name) from eda_shared."""
    return add_provisional_flags(neurons)


def population_ids(neurons_flagged: pd.DataFrame, population: str) -> pd.Index:
    if population == "all":
        keep = np.ones(len(neurons_flagged), bool)
    elif population == "central":
        keep = ~neurons_flagged["is_optic"]
    elif population == "optic":
        keep = neurons_flagged["is_optic"]
    else:
        raise ValueError(population)
    return pd.Index(neurons_flagged.loc[keep, "neuron_id"])


def weight_bin(w: pd.Series) -> pd.Categorical:
    labels = [b[2] for b in WEIGHT_BINS]
    edges = [0.5] + [b[1] + 0.5 for b in WEIGHT_BINS]
    return pd.cut(w, bins=edges, labels=labels)


# --------------------------------------------------------------------------- E1
def per_neuron_neuropil_synapses(edgeroi: pd.DataFrame) -> pd.DataFrame:
    """Long table neuron_id, neuropil, syn (inputs plus outputs), real neuropils only."""
    er = edgeroi[~edgeroi["neuropil"].isin(NOT_A_NEUROPIL)]
    out = er.groupby(["pre_id", "neuropil"], observed=True)["syn_count"].sum()
    inp = er.groupby(["post_id", "neuropil"], observed=True)["syn_count"].sum()
    out.index.names = inp.index.names = ["neuron_id", "neuropil"]
    s = out.astype("int64").add(inp.astype("int64"), fill_value=0)
    return s.rename("syn").reset_index()


def e1_compute(neurons, edgeroi, population="central"):
    ids = population_ids(flags(neurons), population)
    syn = per_neuron_neuropil_synapses(edgeroi)
    syn = syn[syn["neuron_id"].isin(ids)].sort_values(["neuron_id", "syn"], ascending=[True, False])
    total = syn.groupby("neuron_id")["syn"].transform("sum")
    syn["share"] = syn["syn"] / total
    rank = syn.groupby("neuron_id").cumcount()
    top = syn[rank == 0].set_index("neuron_id")["share"]
    second = syn[rank == 1].set_index("neuron_id")["share"].reindex(top.index).fillna(0)
    per = pd.DataFrame({"top_share": top, "gap": top - second})
    stats = {
        "neurons": int(len(per)),
        "no_synapses": int(len(ids) - len(per)),
        "median_top_share": float(per["top_share"].median()),
        "gap_below_0.10": int((per["gap"] < 0.10).sum()),
        "gap_below_0.10_pct": float((per["gap"] < 0.10).mean() * 100),
    }
    return per, stats


def e1_worked_example(neurons, edges, edgeroi, neuron_ids, tau=10):
    """frac_in_<neuropil> for a few neurons, before (τ = 1) and after thresholding pairs at τ."""
    strong = edges.loc[edges["syn_count"] >= tau, ["pre_id", "post_id"]]
    er = edgeroi[edgeroi["post_id"].isin(neuron_ids) & ~edgeroi["neuropil"].isin(NOT_A_NEUROPIL)]
    er_after = er.merge(strong, on=["pre_id", "post_id"])
    rows = []
    for nid in neuron_ids:
        for label, table in [("before (τ = 1)", er), (f"after (τ = {tau})", er_after)]:
            x = table[table["post_id"] == nid].groupby("neuropil", observed=True)["syn_count"].sum()
            total = x.sum()
            frac = (x / total) if total else x * 0.0
            top = frac.sort_values(ascending=False).head(3)
            rows.append({
                "neuron_id": nid,
                "cell_type": neurons.loc[neurons.neuron_id == nid, "cell_type"].iloc[0],
                "edges": label,
                "input_synapses": int(total),
                "top_frac_in": ", ".join(f"{k} {v:.2f}" for k, v in top.items()) or "—",
                "sum_of_fractions": round(float(frac.sum()), 3),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- E2
def e2_compute(neurons, edgeroi):
    syn = per_neuron_neuropil_synapses(edgeroi)
    syn["optic"] = base_neuropil(syn["neuropil"]).isin(OPTIC)
    total = syn.groupby("neuron_id")["syn"].sum()
    optic = syn[syn["optic"]].groupby("neuron_id")["syn"].sum().reindex(total.index, fill_value=0)
    share = (optic / total).reindex(neurons["neuron_id"])
    cls = pd.Series("no synapses", index=share.index)
    cls[share > 0.8] = "clean optic"
    cls[share < 0.2] = "clean central"
    cls[(share >= 0.2) & (share <= 0.8)] = "ambiguous"
    counts = cls.value_counts().reindex(["clean central", "ambiguous", "clean optic", "no synapses"]).fillna(0)
    fl = flags(neurons)
    stats = {
        "neurons": int(len(neurons)),
        **{k: int(v) for k, v in counts.items()},
        "optic_by_super_class_pct": float((neurons["super_class"] == "optic").mean() * 100),
        "optic_by_region_flag_pct": float(fl["is_optic"].mean() * 100),
        "optic_by_synapse_share_pct": float((cls == "clean optic").mean() * 100),
        "ambiguous_pct": float((cls == "ambiguous").mean() * 100),
    }
    return share, cls, stats


# --------------------------------------------------------------------------- E3
def e3_compute(neurons, edges, tau=10, population="central"):
    fl = flags(neurons)
    ids = population_ids(fl, population)
    region = fl.set_index("neuron_id")["neuropil"]
    e = edges[(edges["syn_count"] >= tau) & edges["pre_id"].isin(ids) & edges["post_id"].isin(ids)]
    m = (pd.DataFrame({"r1": region.reindex(e["pre_id"]).to_numpy(),
                       "r2": region.reindex(e["post_id"]).to_numpy(),
                       "syn": e["syn_count"].astype("int64").to_numpy()})
         .groupby(["r1", "r2"])["syn"].sum().unstack(fill_value=0))
    table = load_neuropil_table()
    order = [r for g in ["sensory_periphery", "higher_order", "motor_periphery"]
             for r in sorted(table.index[table["region_group"] == g]) if r in m.index or r in m.columns]
    order += [r for r in sorted(set(m.index) | set(m.columns)) if r not in order]
    m = m.reindex(index=order, columns=order, fill_value=0)
    diag = float(np.trace(m.to_numpy()) / m.to_numpy().sum())
    return m, {"regions": len(order), "within_region_share": diag,
               "synapses": int(m.to_numpy().sum())}


# --------------------------------------------------------------------------- E4 (FAFB supplement)
def e4_fafb_input_ratio(neurons, edges, post_counts: pd.DataFrame, population="central"):
    """Supplementary to E4 (specified for malecns only): proofread-partner inputs / all input synapses.

    `post_counts` is FlyWire's per-neuron neuropil count of postsynapses (all partners,
    including unproofread fragments), summed per neuron.
    """
    ids = population_ids(flags(neurons), population)
    n_post = post_counts.groupby("post_pt_root_id")["count"].sum()
    total_in = edges.groupby("post_id")["syn_count"].sum().astype("int64")
    df = pd.DataFrame({"n_post": n_post.reindex(ids), "total_in": total_in.reindex(ids).fillna(0)}).dropna()
    df = df[df["n_post"] > 0]
    df["ratio"] = df["total_in"] / df["n_post"]
    df = df.join(neurons.set_index("neuron_id")[["super_class"]])
    stats = {"neurons": int(len(df)), "median_ratio": float(df["ratio"].median()),
             "pooled_ratio": float(df["total_in"].sum() / df["n_post"].sum()),
             "over_1": int((df["ratio"] > 1).sum())}
    return df, stats


# --------------------------------------------------------------------------- E5
def type_level_edges(neurons, edges, ids):
    """Type-level edges per side: sum of syn_count over neuron pairs (A, side s) -> (B, side s)."""
    nt = neurons.set_index("neuron_id")[["cell_type", "side"]]
    nt = nt[nt["cell_type"].notna() & nt["side"].isin(["left", "right"])]
    nt = nt[nt.index.isin(ids)]
    e = edges[edges["pre_id"].isin(nt.index) & edges["post_id"].isin(nt.index)]
    pre, post = nt.reindex(e["pre_id"]), nt.reindex(e["post_id"])
    same = pre["side"].to_numpy() == post["side"].to_numpy()
    df = pd.DataFrame({"A": pre["cell_type"].to_numpy()[same], "B": post["cell_type"].to_numpy()[same],
                       "side": pre["side"].to_numpy()[same],
                       "w": e["syn_count"].astype("int64").to_numpy()[same]})
    return df.groupby(["A", "B", "side"])["w"].sum().unstack("side", fill_value=0)


def recurrence_by_bin(w_from: pd.Series, w_to: pd.Series) -> pd.DataFrame:
    present = w_from > 0
    d = pd.DataFrame({"bin": weight_bin(w_from[present]), "found": (w_to[present] >= 1)})
    return d.groupby("bin", observed=False)["found"].agg(["mean", "size"])


def e5_compute(neurons, edges):
    fl = flags(neurons)
    out, stats = {}, {}
    for pop in ["central", "optic"]:
        tl = type_level_edges(neurons, edges, population_ids(fl, pop))
        lr = recurrence_by_bin(tl["left"], tl["right"])
        rl = recurrence_by_bin(tl["right"], tl["left"])
        res = pd.DataFrame({"share": (lr["mean"] + rl["mean"]) / 2,
                            "pairs_left": lr["size"], "pairs_right": rl["size"]})
        out[pop] = res
        stats[pop] = {"type_pairs": int(len(tl)),
                      "share_1syn": float(res.loc["1", "share"]),
                      "share_10plus": float(((lr.loc[["10–19", "20–49", "50+"], "mean"] * lr.loc[["10–19", "20–49", "50+"], "size"]).sum()
                                             + (rl.loc[["10–19", "20–49", "50+"], "mean"] * rl.loc[["10–19", "20–49", "50+"], "size"]).sum())
                                            / (lr.loc[["10–19", "20–49", "50+"], "size"].sum() + rl.loc[["10–19", "20–49", "50+"], "size"].sum()))}
    return out, stats


# --------------------------------------------------------------------------- E7
def e7_compute(neurons, edges, tau=5):
    cx = neurons[neurons["region"].isin(CX_NEUROPILS)].copy()
    ids = pd.Index(cx["neuron_id"])
    per_np = cx["region"].value_counts().reindex(CX_NEUROPILS).fillna(0).astype(int)
    typed = cx[cx["cell_type"].notna()]
    per_type_side = (typed[typed["side"].isin(["left", "right"])]
                     .groupby(["cell_type", "side"], observed=True).size().unstack(fill_value=0))
    per_type_side["total"] = per_type_side.sum(axis=1)
    per_type_side = per_type_side.sort_values("total", ascending=False)
    e = edges[(edges["syn_count"] >= tau) & edges["pre_id"].isin(ids) & edges["post_id"].isin(ids)]
    t = typed.set_index("neuron_id")["cell_type"]
    et = e[e["pre_id"].isin(t.index) & e["post_id"].isin(t.index)]
    tm = (pd.DataFrame({"A": t.reindex(et["pre_id"]).to_numpy(), "B": t.reindex(et["post_id"]).to_numpy(),
                        "w": et["syn_count"].astype("int64").to_numpy()})
          .groupby(["A", "B"])["w"].sum().unstack(fill_value=0))
    order = per_type_side.index.intersection(tm.index.union(tm.columns))
    tm = tm.reindex(index=order, columns=order, fill_value=0)
    roster = set(neurons.loc[neurons["cell_class"] == "CX", "neuron_id"])
    stats = {
        "neurons_region_cx": int(len(cx)),
        "neurons_cell_class_cx": len(roster),
        "overlap": int(cx["neuron_id"].isin(roster).sum()),
        "cell_types": int(typed["cell_type"].nunique()),
        "untyped": int(cx["cell_type"].isna().sum()),
        "types_with_8_to_16_per_side": int(((per_type_side[["left", "right"]].max(axis=1) >= 8)
                                             & (per_type_side[["left", "right"]].max(axis=1) <= 16)).sum()),
        "types_one_per_side": int((per_type_side[["left", "right"]].max(axis=1) == 1).sum()),
        "edges_at_tau": int(len(e)),
    }
    return {"per_neuropil": per_np, "per_type_side": per_type_side, "type_matrix": tm}, stats


def e7_columnar_matrix(neurons, edges, tau=5):
    """Neuron x neuron synapse matrix for columnar CX neurons, sorted by type, then by a
    ring position estimated from the connectivity itself (angle of the top-2 singular
    vectors), since FlyWire gives no protocerebral-bridge glomerulus label."""
    cx = neurons[neurons["region"].isin(CX_NEUROPILS) & (neurons["cell_sub_class"] == "columnar")
                 & neurons["cell_type"].notna()]
    ids = cx["neuron_id"].to_numpy()
    pos = {n: i for i, n in enumerate(ids)}
    e = edges[(edges["syn_count"] >= tau) & edges["pre_id"].isin(ids) & edges["post_id"].isin(ids)]
    m = np.zeros((len(ids), len(ids)))
    m[e["pre_id"].map(pos).to_numpy(), e["post_id"].map(pos).to_numpy()] = e["syn_count"].to_numpy()
    feats = np.log1p(np.hstack([m, m.T]))
    feats -= feats.mean(axis=0)
    u, s, _ = np.linalg.svd(feats, full_matrices=False)
    angle = np.arctan2(u[:, 1], u[:, 0])
    order = pd.DataFrame({"type": cx["cell_type"].to_numpy(), "angle": angle}).sort_values(["type", "angle"]).index
    types = cx["cell_type"].to_numpy()[order]
    return m[np.ix_(order, order)], types
