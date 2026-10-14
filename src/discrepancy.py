"""P5 discrepancy count (course step 15): connections recorded in one copy but not in
the matched pair of another, in the central complex.

Unit. Neurons are matched across copies only by type, so the matched pair is a
type pair (A, B) on one side; for types with exactly one neuron on that side in
both copies the type pair is a single neuron pair (reported as its own row).
In each copy, a type pair is

    recorded at τ  if some neuron pair (a in A, b in B) has at least τ synapses
                   from a onto b, summed over the central-complex neuropils
                   (the P5 label);
    below τ        if it has synapses there but no neuron pair reaches τ;
    absent         if no neuron pair has any synapse there.

For type pairs recorded at τ in copy X, the count is how many are below τ or
absent in copy Y. "Absent" is the strict count: no synapse at all where X has a
connection of at least τ.

How to read it. If the two copies shared their true wiring at type level, every
discrepancy would be a connection missed in one copy, and the count would be a
lower bound on missed positives, because connections missed in both copies stay
invisible. Real copies differ (individual variation, asymmetry; sex in fafb vs
malecns), and every such difference also shows up as a discrepancy. So the count
mixes recording loss with biological difference and is not an unconditional bound
in either direction. The same holds for the recurrence share as an estimate of c:
correlated misses push it up, biological differences push it down. fafb vs
malecns is reported and never used to claim a missed connection.
"""
from __future__ import annotations

import pandas as pd

from src.eda_e import group_key, match_groups
from src.pairs import CX_NEUROPILS, cx_roster, region_pair_counts

STATUSES = ["recorded", "below_tau", "absent"]


def type_pair_table(neurons, pair_syn: pd.Series, key: pd.Series, side: str) -> pd.DataFrame:
    """Per type pair on one side: the largest neuron-pair count and the total, over
    neurons carrying a key (`key` aligned with `neurons`)."""
    k = pd.Series(key.to_numpy(), index=neurons["neuron_id"].to_numpy())
    k = k[(neurons["side"] == side).to_numpy() & k.notna().to_numpy()]
    ps = pair_syn[pair_syn.index.get_level_values(0).isin(k.index) & pair_syn.index.get_level_values(1).isin(k.index)]
    df = pd.DataFrame({"A": k.reindex(ps.index.get_level_values(0)).to_numpy(),
                       "B": k.reindex(ps.index.get_level_values(1)).to_numpy(), "syn": ps.to_numpy()})
    return df.groupby(["A", "B"])["syn"].agg(max_pair="max", total="sum")


def status(table: pd.DataFrame, index, tau: int) -> pd.Series:
    t = table.reindex(index)
    out = pd.Series("absent", index=index)
    out[t["total"].fillna(0).to_numpy() > 0] = "below_tau"
    out[t["max_pair"].fillna(0).to_numpy() >= tau] = "recorded"
    return out


def compare(tab_x, tab_y, groups_both, singleton, taus=(1, 5, 10), names=("x", "y"), scale_y: float = 1.0) -> pd.DataFrame:
    """Rows per direction and τ: type pairs recorded in the first copy, and how they
    stand in the second. Only type pairs whose two groups occur in both copies.

    `scale_y` is the median type-pair total ratio y / x (a proxy for how many more
    synapses y records per connection). Rows with threshold "scaled" anchor τ in x and
    use round(τ · scale_y) in y, whatever the value of scale_y (at least 1 synapse);
    rows with "same" use τ in both. Without a usable scale (NaN), only "same" rows."""
    rows = []
    usable = scale_y == scale_y and scale_y != 1.0              # NaN when no type pair is shared
    for mode in (["same", "scaled"] if usable else ["same"]):
        for (a, ta, sa), (b, tb, sb) in [((names[0], tab_x, 1.0), (names[1], tab_y, scale_y)),
                                         ((names[1], tab_y, scale_y), (names[0], tab_x, 1.0))]:
            idx = ta.index[ta.index.get_level_values(0).isin(groups_both) & ta.index.get_level_values(1).isin(groups_both)]
            single = idx.get_level_values(0).isin(singleton) & idx.get_level_values(1).isin(singleton)
            for tau in taus:
                t_a = max(1, round(tau * sa)) if mode == "scaled" else tau
                t_b = max(1, round(tau * sb)) if mode == "scaled" else tau
                rec = status(ta, idx, t_a) == "recorded"
                for label, mask in [("type pairs", rec), ("one neuron per type", rec & single)]:
                    st = status(tb, idx[mask], t_b).value_counts().reindex(STATUSES, fill_value=0)
                    n = int(mask.sum())
                    rows.append({"threshold": mode, "direction": f"{a} → {b}", "tau": tau, "tau_first": t_a,
                                 "tau_second": t_b, "unit": label, "recorded_in_first": n,
                                 **{f"{s_}_in_second": int(st[s_]) for s_ in STATUSES},
                                 "discrepancy_share": (n - st["recorded"]) / n if n else float("nan"),
                                 "absent_share": st["absent"] / n if n else float("nan")})
    return pd.DataFrame(rows)


def scale(tab_x, tab_y, groups_both) -> float:
    """Median ratio of type-pair synapse totals, second copy over first, over type pairs
    with synapses in both; NaN if there are none. A proxy for how many more (> 1) synapses
    the second copy records per connection: it also moves with how many neuron pairs a
    type pair has in each copy."""
    both = tab_x.index.intersection(tab_y.index)
    both = both[both.get_level_values(0).isin(groups_both) & both.get_level_values(1).isin(groups_both)]
    if len(both) == 0:
        return float("nan")
    return float((tab_y.loc[both, "total"] / tab_x.loc[both, "total"]).median())


def _singletons(neurons, key, side) -> set:
    k = pd.Series(key.to_numpy())[(neurons["side"] == side).to_numpy()].dropna()
    c = k.value_counts()
    return set(c.index[c == 1])


def fafb_left_right(neurons, edgeroi, taus=(1, 5, 10)):
    """fafb left vs right on the CX roster, types matched by cell_type."""
    ids = cx_roster(neurons)
    ps = region_pair_counts(edgeroi, ids)
    n = neurons[neurons["neuron_id"].isin(ids)].reset_index(drop=True)
    key = n["cell_type"]
    tl, tr = type_pair_table(n, ps, key, "left"), type_pair_table(n, ps, key, "right")
    groups = set(key[n["side"] == "left"].dropna()) & set(key[n["side"] == "right"].dropna())
    single = _singletons(n, key, "left") & _singletons(n, key, "right")
    return (compare(tl, tr, groups, single, taus, ("fafb left", "fafb right")),
            {"groups": len(groups), "singleton_types": len(single), "scale_second_over_first": scale(tl, tr, groups)})


def fafb_vs_other(fafb_neurons, fafb_edgeroi, other_neurons, other_edgeroi, fafb_col, other_col, name,
                  side="right", taus=(1, 5, 10)):
    """fafb CX roster (right) against another copy (right), types matched through
    `fafb_col` / `other_col` with compound values merged (src.eda_e.match_groups).
    The other copy's population is every neuron whose group occurs in the fafb roster."""
    ids = cx_roster(fafb_neurons)
    fn = fafb_neurons[fafb_neurons["neuron_id"].isin(ids)].reset_index(drop=True)
    groups = match_groups(fn[fafb_col], other_neurons[other_col])
    kf = group_key(fn[fafb_col], groups)
    ko_all = group_key(other_neurons[other_col], groups)
    keep = ko_all.isin(set(kf.dropna())).to_numpy()
    on, ko = other_neurons[keep].reset_index(drop=True), ko_all[keep].reset_index(drop=True)
    tf = type_pair_table(fn, region_pair_counts(fafb_edgeroi, ids), kf, side)
    to = type_pair_table(on, region_pair_counts(other_edgeroi, on["neuron_id"]), ko, side)
    both = set(kf[fn["side"] == side].dropna()) & set(ko[on["side"] == side].dropna())
    single = _singletons(fn, kf, side) & _singletons(on, ko, side)
    sc = scale(tf, to, both)
    return (compare(tf, to, both, single, taus, (f"fafb {side}", f"{name} {side}"), scale_y=sc),
            {"groups": len(both), "singleton_types": len(single), "other_neurons": int((on["side"] == side).sum()),
             "scale_second_over_first": sc})


def fig_discrepancy(results: dict, tau: int = 10):
    """Stacked bars, one per direction: share of type pairs recorded at τ in the first copy
    that are recorded, below τ or absent in the second. `results`: {label: compare() table};
    between-animal rows use the scale-matched threshold."""
    import matplotlib.pyplot as plt
    from src.eda_plots import INK_2, MUTED, SERIES, finish, style
    style()
    rows = []
    for label, r in results.items():
        mode = "scaled" if (r["threshold"] == "scaled").any() else "same"
        sub = r[(r["tau"] == tau) & (r["unit"] == "type pairs") & (r["threshold"] == mode)]
        for _, x in sub.iterrows():
            n = x["recorded_in_first"]
            rows.append((f"{x['direction']}  (τ {x['tau_first']} → {x['tau_second']}; {n:,})",
                         [x[f"{s}_in_second"] / n * 100 for s in STATUSES]))
    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.55 * len(rows)))
    colors = [SERIES[0], SERIES[3], SERIES[1]]
    for y, (name, shares) in enumerate(rows):
        left = 0
        for s, v, c in zip(STATUSES, shares, colors):
            ax.barh(y, v, left=left, color=c, edgecolor="white", linewidth=2, height=0.6,
                    label={"below_tau": "below τ"}.get(s, s) if y == 0 else None)
            if v >= 4:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Type pairs recorded in the first copy, % (status in the second copy)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    return finish(fig, f"S15_discrepancy_fafb-malecns-hemibrain_cx_min{tau}",
                  "Few central-complex connections vanish in another copy; many fall below the threshold",
                  f"Central complex (fafb roster cell_class CX; other copies: neurons of the matched types), right side "
                  f"between animals, synapses summed over PB, FB, EB, NO. Recorded = some neuron pair of the type pair "
                  f"reaches τ (fafb τ = {tau}; hemibrain and malecns at τ times their median type-pair total ratio to fafb, "
                  "a proxy for synapses per connection). "
                  "Datasets fafb 783, malecns v1.0, hemibrain v1.2 (public export). fafb vs malecns includes sex "
                  "differences and is never read as missed connections.")
