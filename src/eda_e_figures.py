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


DS_COLOR = {"fafb": SERIES[0], "malecns": SERIES[1]}
BRAIN_NOTE = {"fafb": "", "malecns": "Brain part only: in_brain neurons, edges among them, brain neuropils."}


def _tags(data):
    """'fafb-malecns' and 'fafb 783, malecns v1.0' for a {ds: (release, ...)} dict."""
    return "-".join(data), ", ".join(f"{ds} {v[0]}" for ds, v in data.items())


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
                    "Synapses = inputs + outputs per neuropil, partners in the cache only (fafb proofread, malecns Traced); synapses outside "
                    "named neuropils (UNASGD, unspecified) excluded. " + BRAIN_NOTE.get(ds, "")))
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
def fig_e2(data):
    """data: {ds: (release, neurons, edgeroi)}; one stacked bar per dataset."""
    order = ["clean central", "ambiguous", "clean optic", "no synapses"]
    colors = [SERIES[0], SERIES[1], SERIES[2], MUTED]
    stats, tabs = {}, {}
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.9 * len(data)))
    for y, (ds, (release, neurons, edgeroi)) in enumerate(data.items()):
        share, cls, st = E.e2_compute(neurons, edgeroi)
        stats[ds], tabs[ds] = st, {"optic_share": share, "class": cls}
        pct = cls.value_counts(normalize=True).reindex(order).fillna(0) * 100
        left = 0
        for k, c in zip(order, colors):
            ax.barh([y], [pct[k]], left=left, color=c, edgecolor="white", linewidth=2, height=0.5,
                    label=k if y == 0 else None)
            if pct[k] > 1:
                ax.text(left + pct[k] / 2, y, f"{pct[k]:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if k != "no synapses" else INK_2)
            left += pct[k]
    ax.set_yticks(range(len(data)), [f"{ds}\n({stats[ds]['neurons']:,})" for ds in data])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Neurons, %")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    tag, rel = _tags(data)
    title = "; ".join(f"{ds}: {st['optic_by_synapse_share_pct']:.0f}% clean optic, {st['ambiguous_pct']:.1f}% ambiguous"
                      for ds, st in stats.items())
    extra = " ".join(f"{ds}: by super_class {st['optic_by_super_class_pct']:.1f}% optic, by region flag "
                     f"{st['optic_by_region_flag_pct']:.1f}%." for ds, st in stats.items())
    p = finish(fig, f"E2_{tag}_all_min1", f"Few neurons straddle the optic boundary — {title}",
               f"Datasets {rel}; threshold τ = 1; population: all (malecns: brain part). Optic share of each "
               f"neuron's synapses: > 0.8 clean optic, < 0.2 clean central. {extra}")
    return [p], stats, tabs


# ------------------------------------------------------------------------ E3
def fig_e3(data):
    """data: {ds: (release, neurons, edges)}; one heatmap panel per dataset, same region order."""
    table = load_neuropil_table()
    mats, stats = {}, {}
    for ds, (release, neurons, edges) in data.items():
        mats[ds], stats[ds] = E.e3_compute(neurons, edges, tau=10, population="central")
    order = list(dict.fromkeys(r for m in mats.values() for r in m.index))
    groups = table["region_group"].reindex(order).fillna("unknown")
    rank = {"sensory_periphery": 0, "higher_order": 1, "motor_periphery": 2}
    order = sorted(order, key=lambda r: (rank.get(groups[r], 3), r))
    groups = groups.reindex(order).to_numpy()
    vmax = max(np.log10(m.to_numpy().max() + 1) for m in mats.values())
    fig, axs = plt.subplots(1, len(data) + 1, figsize=(8.5 * len(data) + 0.8, 9.4),
                            gridspec_kw={"width_ratios": [1] * len(data) + [0.035]})
    axes, cax = axs[:-1], axs[-1]
    for ax, (ds, m) in zip(axes, mats.items()):
        m = mats[ds] = m.reindex(index=order, columns=order, fill_value=0)
        im = ax.imshow(np.log10(m.to_numpy() + 1), cmap=SEQ, vmin=0, vmax=vmax, aspect="equal",
                       interpolation="nearest")
        ax.set_xticks(range(len(m)), m.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(m)), m.index, fontsize=6)
        ax.grid(False)
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color=INK_2, lw=0.8)
                ax.axvline(i - 0.5, color=INK_2, lw=0.8)
        ax.set_title(f"{ds}: {stats[ds]['within_region_share']:.0%} within one region", fontsize=10)
        ax.set(xlabel="Postsynaptic neuron's region", ylabel="Presynaptic neuron's region")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log10(synapses + 1)", color=INK_2, fontsize=8)
    tag, rel = _tags(data)
    title = " and ".join(f"{s_['within_region_share'] * 100:.0f}% ({ds})" for ds, s_ in stats.items())
    p = finish(fig, f"E3_{tag}_central_min10",
               f"Most central synapses stay within one region: {title}",
               f"Datasets {rel}; threshold τ = 10; population: central (malecns: brain part). Regions are base "
               "neuropils (side merged), grouped sensory periphery | higher order | motor periphery; "
               "one colour scale for both panels. Empty rows: malecns v1.0 has no AMMC, OCG or GA neuropil.")
    return [p], stats, {"matrix": mats}


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


def fig_e4_malecns(neurons, edges_all, edges_brain=None, ds="malecns", release="v1.0"):
    df, st = E.e4_malecns_input_ratio(neurons, edges_all, edges_brain)
    paths = []
    for col, name in [("super_class", "E4a"), ("region_group", "E4b")]:
        groups = df.groupby(col)["ratio"]
        order = groups.median().sort_values().index
        sizes = groups.size()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot([df.loc[df[col] == g, "ratio"] for g in order],
                   tick_labels=[f"{g}\n({sizes[g]:,})" for g in order],
                   showfliers=False, widths=0.5, patch_artist=True,
                   boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
                   medianprops={"color": SERIES[0], "linewidth": 2},
                   whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        ax.set(ylabel="Input ratio: total_in_syn / n_post", ylim=(0, 1.02))
        ax.tick_params(axis="x", rotation=20 if col == "super_class" else 0)
        extra = ""
        if "median_ratio_brain_partners" in st:
            extra = (f" Counting brain partners only, the median is {st['median_ratio_brain_partners']:.0%} "
                     f"(ascending {st['median_ratio_ascending_brain_partners']:.0%}).")
        paths.append(finish(
            fig, f"{name}_{ds}_{release}_central_min1",
            f"Central neurons keep a median {st['median_ratio']:.0%} of their input synapses as edges "
            f"from Traced neurons",
            _cap(ds, release, 1, f"central ({st['neurons']:,} neurons with n_post > 0), by {col}",
                 "n_post counts synapses at confidence ≥ 0.5, the same filter as the edges (checked per neuron). "
                 "Neurons: brain part; inputs: every Traced partner, nerve cord included (E4 is malecns-only)."
                 + extra)))
    return paths, st, {"ratio": df}


# ------------------------------------------------------------------------ E5
def fig_e5(neurons, edges, ds="fafb", release="783", item="E5", note=""):
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
    p = finish(fig, f"{item}_{ds}_{release}_central-optic_min1",
               f"One-synapse type edges recur on the other side {st['central']['share_1syn']:.0%} of the time; "
               f"edges of 10+ recur {st['central']['share_10plus']:.0%}",
               _cap(ds, release, 1, "central and optic, type-level edges, left vs right averaged both ways",
                    "Neurons without cell_type or with side center/unknown left out. " + note))
    return [p], st, res


# ------------------------------------------------------------------------ E7
def fig_e7a(data):
    """data: {ds: (release, neurons)}; neurons per CX neuropil, one bar per dataset."""
    per = {ds: n["region"].value_counts().reindex(E.CX_NEUROPILS).fillna(0).astype(int)
           for ds, (release, n) in data.items()}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(E.CX_NEUROPILS))
    w = 0.8 / len(per)
    for i, (ds, pn) in enumerate(per.items()):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, pn.values, width=w - 0.03, color=DS_COLOR[ds], label=f"{ds} ({pn.sum():,})")
        for xi, v in zip(xs, pn.values):
            ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, E.CX_NEUROPILS)
    ax.set(ylabel="Neurons (region = this neuropil)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    tag, rel = _tags(data)
    shares = ", ".join(f"{pn['FB'] / pn.sum():.0%} in {ds}" for ds, pn in per.items())
    p = finish(fig, f"E7a_{tag}_cx_min1",
               f"The fan-shaped body holds most central-complex neurons ({shares})",
               f"Datasets {rel}; threshold τ = 1; population: neurons whose region is PB, FB, EB or NO.")
    return [p], {ds: pn.to_dict() for ds, pn in per.items()}, per


def fig_e7(neurons, edges, columnar_types, ds="fafb", release="783", type_column="cell_type"):
    tabs, st = E.e7_compute(neurons, edges, tau=5)
    paths = []
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
                        f"{st['types_8_to_16_on_both_sides']} central-complex types have 8–16 neurons on each side; "
                        f"{st['types_one_on_both_sides']} have exactly one on each side",
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
    m, types = E.e7_columnar_matrix(neurons, edges, columnar_types, type_column, tau=5)
    st["columnar_neurons"] = int(len(types))
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
                             "(no glomerulus label is used)."
                             + f" Columnar = {type_column} is a fafb central-complex type with a neuron labelled "
                               "columnar (whole types).")))
    return paths, st, tabs
