"""Figures for EDA workstream E. Each function computes its item and writes its figure(s).

Returns (paths, stats, tables) so the notebook and the report quote the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src import eda_e as E
from src.eda_plots import INK_2, MUTED, SERIES, SEQ, finish, style
from eda_shared.eda_helpers import load_neuropil_table

style()


def _cap(ds, release, tau, population, extra=""):
    return f"Dataset {ds}, release {release}; threshold τ = {tau}; population: {population}.{(' ' + extra) if extra else ''}"


# ------------------------------------------------------------------------ E1
def fig_e1(neurons, edges, edgeroi, ds="fafb", release="783"):
    per, st = E.e1_compute(neurons, edgeroi, "central")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.arange(0, 1.0001, 0.05)
    a.hist(per["top_share"], bins=bins, color=SERIES[0], edgecolor="white", linewidth=1)
    a.set(xlabel="Share of a neuron's synapses in its top neuropil", ylabel="Neurons", xlim=(0, 1))
    b.hist(per["gap"], bins=bins, color=SERIES[0], edgecolor="white", linewidth=1)
    b.axvline(0.10, color=INK_2, lw=1, ls="--")
    b.text(0.11, b.get_ylim()[1] * 0.92, f"gap < 0.10: {st['gap_below_0.10']:,} neurons "
           f"({st['gap_below_0.10_pct']:.0f}%)", fontsize=8, color=INK_2)
    b.set(xlabel="Top share minus second share", ylabel="Neurons", xlim=(0, 1))
    p = finish(fig, f"E1_{ds}_{release}_central_min1",
               f"For {st['gap_below_0.10_pct']:.0f}% of central neurons the 'most synapses' region is a close call",
               _cap(ds, release, 1, f"central ({st['neurons']:,} neurons with synapses)",
                    "Synapses = inputs + outputs per neuropil, proofread partners only; UNASGD excluded."))
    # worked example: one central-complex neuron, the closest call, and one that loses all input at τ = 10
    gap = per["gap"]
    cx = neurons.loc[neurons["cell_type"] == "EPG", "neuron_id"]
    weak = edges[edges["syn_count"] < 10].groupby("post_id").size()
    strong = edges[edges["syn_count"] >= 10].groupby("post_id").size()
    only_weak = weak.index.difference(strong.index).intersection(per.index)
    example = [int(cx.iloc[0]), int(gap.idxmin()), int(only_weak[0])]
    worked = E.e1_worked_example(neurons, edges, edgeroi, example, tau=10)
    return [p], st, {"per_neuron": per, "worked_example": worked}


# ------------------------------------------------------------------------ E2
def fig_e2(neurons, edgeroi, ds="fafb", release="783"):
    share, cls, st = E.e2_compute(neurons, edgeroi)
    order = ["clean central", "ambiguous", "clean optic", "no synapses"]
    colors = [SERIES[0], SERIES[1], SERIES[2], MUTED]
    pct = cls.value_counts(normalize=True).reindex(order).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 3.2))
    left = 0
    for k, c in zip(order, colors):
        ax.barh([ds], [pct[k]], left=left, color=c, edgecolor="white", linewidth=2, height=0.5, label=k)
        ax.text(left + pct[k] / 2, 0.36, f"{k}\n{pct[k]:.1f}%", ha="center", va="bottom", fontsize=8,
                color=INK_2) if pct[k] > 1 else None
        left += pct[k]
    ax.set(xlim=(0, 100), ylim=(-0.4, 0.75), xlabel="Neurons, %")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    p = finish(fig, f"E2_{ds}_{release}_all_min1",
               f"{st['optic_by_synapse_share_pct']:.0f}% of neurons are clean optic by synapses; "
               f"only {st['ambiguous_pct']:.1f}% straddle the boundary",
               _cap(ds, release, 1, f"all ({st['neurons']:,} neurons)",
                    f"Optic share of each neuron's synapses: > 0.8 clean optic, < 0.2 clean central. "
                    f"By super_class {st['optic_by_super_class_pct']:.1f}% optic; by region flag {st['optic_by_region_flag_pct']:.1f}%."))
    return [p], st, {"optic_share": share, "class": cls}


# ------------------------------------------------------------------------ E3
def fig_e3(neurons, edges, ds="fafb", release="783"):
    m, st = E.e3_compute(neurons, edges, tau=10, population="central")
    table = load_neuropil_table()
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(np.log10(m.to_numpy() + 1), cmap=SEQ, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(m)), m.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(m)), m.index, fontsize=6.5)
    ax.grid(False)
    groups = table["region_group"].reindex(m.index).fillna("unknown").to_numpy()
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color=INK_2, lw=0.8)
            ax.axvline(i - 0.5, color=INK_2, lw=0.8)
    ax.set(xlabel="Postsynaptic neuron's region", ylabel="Presynaptic neuron's region")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("log10(synapses + 1)", color=INK_2, fontsize=8)
    p = finish(fig, f"E3_{ds}_{release}_central_min10",
               f"{st['within_region_share'] * 100:.0f}% of central synapses stay within one region",
               _cap(ds, release, 10, "central",
                    "Regions are base neuropils (side merged), grouped sensory periphery | higher order | motor periphery."))
    return [p], st, {"matrix": m}


# ------------------------------------------------------------------------ E4 (FAFB supplement)
def fig_e4_fafb(neurons, edges, post_counts, ds="fafb", release="783"):
    df, st = E.e4_fafb_input_ratio(neurons, edges, post_counts)
    groups = df.groupby("super_class", observed=True)["ratio"]
    order = groups.median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([df.loc[df.super_class == g, "ratio"] for g in order], tick_labels=list(order), vert=True,
               showfliers=False, widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
               medianprops={"color": SERIES[0], "linewidth": 2},
               whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    ax.set(ylabel="Input ratio: proofread-partner inputs / all inputs", ylim=(0, 1.02))
    ax.tick_params(axis="x", rotation=20)
    p = finish(fig, f"E4supp_{ds}_{release}_central_min1",
               f"Central neurons keep a median {st['median_ratio']:.0%} of their inputs as edges to proofread partners",
               _cap(ds, release, 1, f"central ({st['neurons']:,} neurons)",
                    "Supplement: the plan specifies E4 for malecns; FAFB's all-synapse counts (Zenodo 783) allow the same ratio."))
    return [p], st, {"ratio": df}


# ------------------------------------------------------------------------ E5
def fig_e5(neurons, edges, ds="fafb", release="783"):
    res, st = E.e5_compute(neurons, edges)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(res["central"]))
    for (pop, r), c in zip(res.items(), SERIES[:2]):
        ax.plot(x, r["share"], color=c, marker="o", markersize=7, label=pop,
                markeredgecolor="white", markeredgewidth=1.5)
        dy = 0.05 if pop == "optic" else -0.06
        ax.text(x[0] + 0.15, r["share"].iloc[0] + dy, pop, color=INK_2, fontsize=8, va="center")
    ax.axhline(0.9, color=INK_2, lw=1, ls="--")
    ax.axvline(list(res["central"].index).index("10–19"), color=MUTED, lw=1, ls=":")
    ax.set_xticks(x, res["central"].index)
    ax.set(xlabel="Type-level edge weight on one side (synapses)", ylabel="Share found on the other side",
           ylim=(0, 1.02), xlim=(-0.3, len(x) - 0.4))
    ax.legend(loc="lower right")
    p = finish(fig, f"E5_{ds}_{release}_central-optic_min1",
               f"One-synapse type edges recur on the other side {st['central']['share_1syn']:.0%} of the time; "
               f"edges of 10+ recur {st['central']['share_10plus']:.0%}",
               _cap(ds, release, 1, "central and optic, type-level edges, left vs right averaged both ways",
                    "Neurons without cell_type or with side center/unknown left out."))
    return [p], st, res


# ------------------------------------------------------------------------ E7
def fig_e7(neurons, edges, ds="fafb", release="783"):
    tabs, st = E.e7_compute(neurons, edges, tau=5)
    paths = []
    # a. neurons per neuropil
    fig, ax = plt.subplots(figsize=(6, 3.4))
    pn = tabs["per_neuropil"]
    ax.bar(pn.index, pn.values, color=SERIES[0], width=0.55)
    for i, v in enumerate(pn.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set(ylabel="Neurons (region = this neuropil)")
    ax.grid(axis="x", visible=False)
    paths.append(finish(fig, f"E7a_{ds}_{release}_cx_min1",
                        f"The fan-shaped body holds {pn['FB'] / pn.sum():.0%} of central-complex neurons",
                        _cap(ds, release, 1, f"central complex by region ({st['neurons_region_cx']:,} neurons)")))
    # b. neurons per type and side
    pts = tabs["per_type_side"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(pts))
    ax.scatter(x, pts["left"], s=12, color=SERIES[0], label="left", zorder=3)
    ax.scatter(x, pts["right"], s=12, color=SERIES[1], label="right", zorder=3, marker="D")
    ax.set_yscale("log")
    ax.axhspan(8, 16, color="#cde2fb", alpha=0.5, lw=0)
    ax.text(len(x) * 0.62, 11, "8–16 neurons per side", fontsize=8, color=INK_2)
    ax.set(xlabel=f"Cell type, sorted by size ({len(pts)} types)", ylabel="Neurons per side (log)")
    ax.set_xticks([])
    ax.legend(loc="upper right")
    paths.append(finish(fig, f"E7b_{ds}_{release}_cx_min1",
                        f"{st['types_with_8_to_16_per_side']} central-complex types have 8–16 neurons per side; "
                        f"{st['types_one_per_side']} are one per side",
                        _cap(ds, release, 1, "central complex by region, typed neurons with side left/right")))
    # c. type x type matrix
    tm = tabs["type_matrix"]
    fig, ax = plt.subplots(figsize=(9, 8.5))
    im = ax.imshow(tm.to_numpy() + 1, cmap=SEQ, norm=LogNorm(vmin=1), interpolation="nearest", aspect="equal")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set(xlabel=f"Postsynaptic type ({len(tm)} types, sorted by size)", ylabel="Presynaptic type")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("synapses + 1 (log)", color=INK_2, fontsize=8)
    dens = (tm.to_numpy() > 0).mean()
    paths.append(finish(fig, f"E7c_{ds}_{release}_cx_min5",
                        f"Only {dens:.0%} of central-complex type pairs are connected at 5 synapses",
                        _cap(ds, release, 5, "central complex by region; type × type synapse sums")))
    # d. columnar neuron x neuron
    m, types = E.e7_columnar_matrix(neurons, edges, tau=5)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    im = ax.imshow(m + 1, cmap=SEQ, norm=LogNorm(vmin=1), interpolation="nearest", aspect="equal")
    ax.grid(False)
    bounds = np.flatnonzero(types[1:] != types[:-1]) + 0.5
    for b_ in bounds:
        ax.axhline(b_, color=MUTED, lw=0.4); ax.axvline(b_, color=MUTED, lw=0.4)
    starts = np.concatenate([[0], bounds + 0.5]).astype(int)
    ends = np.concatenate([bounds + 0.5, [len(types)]]).astype(int)
    big = [(s, e) for s, e in zip(starts, ends) if e - s >= 25]
    ax.set_xticks([]); ax.set_yticks([(s + e) / 2 for s, e in big], [types[s] for s, e in big], fontsize=6)
    ax.set(xlabel=f"Postsynaptic neuron ({len(types)} columnar neurons)", ylabel="Presynaptic neuron, grouped by type")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("synapses + 1 (log)", color=INK_2, fontsize=8)
    # fill of each connected (pre type, post type) block: share of its neuron pairs that are connected
    fills = []
    for s1, e1 in zip(starts, ends):
        for s2, e2 in zip(starts, ends):
            blk = m[s1:e1, s2:e2] > 0
            if s1 == s2:
                blk = blk[~np.eye(e1 - s1, dtype=bool)]
            if blk.any():
                fills.append(blk.mean())
    st["columnar_block_fill_median"] = float(np.median(fills))
    st["columnar_connected_type_pairs"] = len(fills)
    paths.append(finish(fig, f"E7d_{ds}_{release}_cx-columnar_min5",
                        f"Type alone does not fix a connection: in a connected pair of columnar types, a median "
                        f"{np.median(fills):.0%} of the neuron pairs are connected",
                        _cap(ds, release, 5, "columnar central-complex neurons",
                             "Within a type, neurons are ordered by a ring position estimated from connectivity "
                             "(FlyWire gives no glomerulus label).")))
    return paths, st, tabs
