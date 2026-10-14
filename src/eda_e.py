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
UNSPECIFIED = "unspecified"   # malecns labels for synapses outside a named neuropil
# PROVISIONAL, to agree with the group (README open issue 6): malecns nerve-cord
# neuropils and nerves, as base names (eda_shared strips one side suffix). Passed
# to add_provisional_flags as `nerve_cord`. CRN (16 neurons, 8 synapses) is a nerve
# label of unclear position; it is outside the brain table, so it goes here.
MALECNS_NERVE_CORD = frozenset({
    "ANm", "IntTct", "LTct", "Ov", "CRN", "VNC-unspecified", "CV-unspecified",
    "LegNp(T1)", "LegNp(T2)", "LegNp(T3)", "mVAC(T1)", "mVAC(T2)", "mVAC(T3)",
    "NTct(UTct-T1)", "WTct(UTct-T2)", "HTct(UTct-T3)",
    "ADMN", "AbN1", "AbN2", "AbN3", "AbN4", "AbNT", "CvN", "DMetaN", "DProN",
    "MesoAN", "MesoLN", "MetaLN", "PDMN", "PrN", "ProAN", "ProCN", "ProLN", "VProN",
})
WEIGHT_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"),
               (10, 19, "10–19"), (20, 49, "20–49"), (50, np.inf, "50+")]


def base_neuropil(names: pd.Series) -> pd.Series:
    return names.astype("string").str.replace(r"(_[LR]|\([LR]\))$", "", regex=True)


def flags(neurons: pd.DataFrame) -> pd.DataFrame:
    """neurons with is_optic, region_group, neuropil (base name) and in_brain from eda_shared."""
    return add_provisional_flags(neurons, nerve_cord=MALECNS_NERVE_CORD)


def brain_part(neurons, edges, edgeroi=None):
    """The plan's malecns population: neurons with in_brain, the edges among them and,
    for edgeroi, only the rows in brain neuropils (nerve-cord rows dropped, so a
    neuron's per-neuropil synapses cover the same volume as in fafb).

    Neurons with region unknown have in_brain False and are left out as well.
    """
    fl = flags(neurons)
    keep = fl.loc[fl["in_brain"], "neuron_id"]
    n = neurons[neurons["neuron_id"].isin(keep)].reset_index(drop=True)
    e = edges[edges["pre_id"].isin(keep) & edges["post_id"].isin(keep)].reset_index(drop=True)
    if edgeroi is None:
        return n, e
    er = edgeroi[edgeroi["pre_id"].isin(keep) & edgeroi["post_id"].isin(keep)]
    nerve = base_neuropil(er["neuropil"]).isin(MALECNS_NERVE_CORD) | er["neuropil"].isin(MALECNS_NERVE_CORD)
    return n, e, er[~nerve.to_numpy()].reset_index(drop=True)


def real_neuropil(names: pd.Series) -> np.ndarray:
    """True for a named neuropil: not UNASGD / None / unknown, not a malecns `unspecified` label."""
    s = names.astype("string")
    return (~s.isin(NOT_A_NEUROPIL) & ~s.str.contains(UNSPECIFIED, regex=False)).to_numpy()


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
    er = edgeroi[real_neuropil(edgeroi["neuropil"])]
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
    er = edgeroi[edgeroi["post_id"].isin(neuron_ids)]
    er = er[real_neuropil(er["neuropil"])]
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


def e4_malecns_input_ratio(neurons, edges_all, edges_brain=None, population="central"):
    """E4: input ratio = total_in_syn / n_post per neuron (malecns; n_post from body-stats).

    `neurons` is the brain part; inputs are counted over `edges_all`, every Traced
    partner. Deviation from the plan's brain-part rule, which exists to compare with
    fafb: E4 is malecns-only, and dropping nerve-cord partners would read as lost
    input (ascending neurons fall from a median 0.93 to 0.27). `edges_brain`, if
    given, adds the brain-partners-only figures for comparison.
    """
    fl = flags(neurons)
    ids = population_ids(fl, population)
    nn = fl.set_index("neuron_id").reindex(ids)
    total_in = edges_all.groupby("post_id")["syn_count"].sum().astype("int64")
    df = pd.DataFrame({"n_post": nn["n_post"].astype("float64"),
                       "total_in": total_in.reindex(ids).fillna(0).to_numpy()}, index=ids)
    df = df[df["n_post"] > 0]
    df["ratio"] = df["total_in"] / df["n_post"]
    df["super_class"] = nn["super_class"].reindex(df.index).astype(str)
    df["region_group"] = nn["region_group"].reindex(df.index).astype(str)
    stats = {"neurons": int(len(df)), "median_ratio": float(df["ratio"].median()),
             "pooled_ratio": float(df["total_in"].sum() / df["n_post"].sum()),
             "over_1": int((df["ratio"] > 1).sum()), "below_0.5": int((df["ratio"] < 0.5).sum())}
    if edges_brain is not None:
        brain_in = edges_brain.groupby("post_id")["syn_count"].sum().reindex(df.index).fillna(0)
        r = brain_in / df["n_post"]
        stats["median_ratio_brain_partners"] = float(r.median())
        stats["pooled_ratio_brain_partners"] = float(brain_in.sum() / df["n_post"].sum())
        stats["median_ratio_ascending_brain_partners"] = float(r[df["super_class"] == "ascending"].median())
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


def e5_by_type_size(neurons, edges, population="central"):
    """E5 recurrence split by type size: the smaller of the two types' per-side neuron counts.

    A type edge between one-neuron-per-side types is a single neuron pair, so its
    chance of recurring is lower than for large types, where many neuron pairs
    stand behind the same type edge. Type pairs where a type is missing on one
    side (min side count 0) are reported in their own row.
    """
    ids = population_ids(flags(neurons), population)
    tl = type_level_edges(neurons, edges, ids)
    nt = neurons[neurons["cell_type"].notna() & neurons["side"].isin(["left", "right"])
                 & neurons["neuron_id"].isin(ids)]
    cnt = nt.groupby(["cell_type", "side"], observed=True).size().unstack(fill_value=0)[["left", "right"]]
    size = cnt.min(axis=1)
    a, b = tl.index.get_level_values(0), tl.index.get_level_values(1)
    pair_size = np.minimum(size.reindex(a).to_numpy(), size.reindex(b).to_numpy())
    rows = {}
    for lo, hi, label in [(0, 0, "min side count 0"), (1, 1, "min side count 1"), (2, 4, "2–4"),
                          (5, 19, "5–19"), (20, np.inf, "20+")]:
        sub = tl[(pair_size >= lo) & (pair_size <= hi)]
        lr = recurrence_by_bin(sub["left"], sub["right"])
        rl = recurrence_by_bin(sub["right"], sub["left"])
        rows[label] = {"type_pairs": int(len(sub)), **((lr["mean"] + rl["mean"]) / 2).round(3).to_dict()}
    return pd.DataFrame(rows).T.astype({"type_pairs": int})


def e5_restricted(neurons, edges, population="central"):
    """E5 recurrence on restricted type sets: both types present on both sides,
    equal left/right counts, and both types hemibrain-matched (plus present on both sides)."""
    ids = population_ids(flags(neurons), population)
    tl = type_level_edges(neurons, edges, ids)
    nt = neurons[neurons["cell_type"].notna() & neurons["side"].isin(["left", "right"])
                 & neurons["neuron_id"].isin(ids)]
    cnt = nt.groupby(["cell_type", "side"], observed=True).size().unstack(fill_value=0)[["left", "right"]]
    both = cnt.index[(cnt["left"] > 0) & (cnt["right"] > 0)]
    equal = cnt.index[(cnt["left"] == cnt["right"]) & (cnt["left"] > 0)]
    matched = pd.Index(neurons.loc[neurons["hemibrain_type"].notna(), "cell_type"].dropna().unique())
    a, b = tl.index.get_level_values(0), tl.index.get_level_values(1)
    masks = {"all types": np.ones(len(tl), bool),
             "both types on both sides": a.isin(both) & b.isin(both),
             "equal left/right counts": a.isin(equal) & b.isin(equal),
             "hemibrain-matched, both sides": a.isin(both) & b.isin(both) & a.isin(matched) & b.isin(matched)}
    rows = {}
    for label, m in masks.items():
        sub = tl[m]
        lr = recurrence_by_bin(sub["left"], sub["right"])
        rl = recurrence_by_bin(sub["right"], sub["left"])
        rows[label] = {"type_pairs": int(len(sub)), **((lr["mean"] + rl["mean"]) / 2).round(3).to_dict()}
    return pd.DataFrame(rows).T.astype({"type_pairs": int})


# --------------------------------------------------------------------------- E6
def match_groups(*value_series) -> dict:
    """Type name -> match group. A compound value such as "A,B" (one type matching the
    union of several on the other side) joins its names; groups sharing a name merge."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for values in value_series:
        for v in pd.Series(values).dropna().unique():
            parts = [p.strip() for p in str(v).split(",") if p.strip()]
            for p in parts[1:]:
                ra, rb = find(parts[0]), find(p)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            find(parts[0])
    members = {}
    for x in parent:
        members.setdefault(find(x), []).append(x)
    label = {root: ",".join(sorted(m)) for root, m in members.items()}
    return {x: label[find(x)] for x in parent}


def group_key(values: pd.Series, groups: dict) -> pd.Series:
    first = values.astype("string").str.split(",").str[0].str.strip()
    return first.map(groups)


def keyed_type_edges(neurons, edges, key: pd.Series) -> pd.Series:
    """Type-level edge weights between match groups: sum of syn_count over edges whose
    neurons both carry a key (`key` indexed like `neurons`)."""
    k = pd.Series(key.to_numpy(), index=neurons["neuron_id"].to_numpy()).dropna()
    e = edges[edges["pre_id"].isin(k.index) & edges["post_id"].isin(k.index)]
    df = pd.DataFrame({"A": k.reindex(e["pre_id"]).to_numpy(), "B": k.reindex(e["post_id"]).to_numpy(),
                       "w": e["syn_count"].astype("int64").to_numpy()})
    return df.groupby(["A", "B"])["w"].sum()


def _e6_side(neurons, key_values, groups, matched, regions, population, side):
    fl = flags(neurons)
    key = group_key(key_values, groups)
    keep = ((fl["side"] == side) & fl["neuropil"].isin(regions) & key.isin(matched)
            & fl["neuron_id"].isin(population_ids(fl, population)))
    return neurons[keep.to_numpy()], key[keep.to_numpy()]


def e6_compare(a, b, population="central", side="right"):
    """E5's recurrence curve between two copies, for matched types in shared regions.

    a, b: (neurons, edges, key_values), key_values aligned with neurons: the column
    that names the shared type (fafb cell_type vs malecns flywire_type; fafb
    hemibrain_type vs hemibrain cell_type). Only `side` neurons, the population, and
    neurons whose region (base neuropil) has neurons in both copies are used.
    Averaged over both directions, as in E5.
    """
    (na, ea, va), (nb, eb, vb) = a, b
    groups = match_groups(va, vb)
    matched = set(group_key(va, groups).dropna()) & set(group_key(vb, groups).dropna())
    regions = (set(flags(na)["neuropil"]) & set(flags(nb)["neuropil"])) - {"unknown", "nerve_cord"}
    na_, ka = _e6_side(na, va, groups, matched, regions, population, side)
    nb_, kb = _e6_side(nb, vb, groups, matched, regions, population, side)
    wa, wb = keyed_type_edges(na_, ea, ka), keyed_type_edges(nb_, eb, kb)
    idx = wa.index.union(wb.index)
    wa, wb = wa.reindex(idx, fill_value=0), wb.reindex(idx, fill_value=0)
    ab, ba = recurrence_by_bin(wa, wb), recurrence_by_bin(wb, wa)
    res = pd.DataFrame({"share": (ab["mean"] + ba["mean"]) / 2, "a_to_b": ab["mean"], "b_to_a": ba["mean"],
                        "pairs_a": ab["size"], "pairs_b": ba["size"]})
    hi = ["10–19", "20–49", "50+"]
    stats = {
        "match_groups": len(matched), "regions": len(regions),
        "neurons_a": int(len(na_)), "neurons_b": int(len(nb_)),
        "type_pairs": int(len(idx)),
        "share_1syn": float(res.loc["1", "share"]),
        "share_10plus": float(((ab.loc[hi, "mean"] * ab.loc[hi, "size"]).sum() + (ba.loc[hi, "mean"] * ba.loc[hi, "size"]).sum())
                              / (ab.loc[hi, "size"].sum() + ba.loc[hi, "size"].sum())),
        "share_10_19": float(res.loc["10–19", "share"]),
        "a_to_b_10_19": float(res.loc["10–19", "a_to_b"]),
        "b_to_a_10_19": float(res.loc["10–19", "b_to_a"]),
    }
    return res, stats, {"matched": matched, "regions": regions, "groups": groups}


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
        "types_8_to_16_on_both_sides": int(((per_type_side[["left", "right"]].min(axis=1) >= 8)
                                             & (per_type_side[["left", "right"]].max(axis=1) <= 16)).sum()),
        "types_one_on_both_sides": int(((per_type_side["left"] == 1) & (per_type_side["right"] == 1)).sum()),
        "edges_at_tau": int(len(e)),
    }
    return {"per_neuropil": per_np, "per_type_side": per_type_side, "type_matrix": tm}, stats


def columnar_cx_types(fafb_neurons) -> set:
    """fafb central-complex types with at least one neuron labelled columnar. Selecting
    whole types matters: fafb labels only 4 of its 244 PFNp neurons columnar."""
    n = fafb_neurons
    return set(n.loc[(n["cell_class"] == "CX") & (n["cell_sub_class"] == "columnar"), "cell_type"].dropna())


def e7_columnar_matrix(neurons, edges, columnar_types, type_column="cell_type", tau=5):
    """Neuron x neuron synapse matrix for columnar CX neurons, sorted by type, then by a
    ring position estimated from the connectivity itself (angle of the top-2 singular
    vectors), since no protocerebral-bridge glomerulus label is used.

    Columnar = `type_column` in `columnar_types` (columnar_cx_types of fafb): cell_type
    for fafb, flywire_type for malecns, which has no cell_sub_class."""
    columnar = neurons[type_column].isin(columnar_types)
    cx = neurons[neurons["region"].isin(CX_NEUROPILS) & columnar & neurons["cell_type"].notna()]
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
