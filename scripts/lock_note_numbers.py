"""Numbers cited by the lock note, from the corrected tables. No model is fitted here.

Writes reports/lock_note_numbers.json. c estimate: recurrence of recorded type pairs
across copies (src/discrepancy.py). If a recorded connection were recorded in the other
copy independently with probability c, and the copies shared their wiring at type level,
the share recorded in the other copy would estimate c. Biological differences push it
down, correlated misses push it up. Intervals resample source type groups.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eda_shared.corrections import fafb_neuropil_counts, region_from_counts  # noqa: E402
from src import fetch_fafb, fetch_hemibrain, fetch_malecns  # noqa: E402
from src.discrepancy import scale, status, type_pair_table  # noqa: E402
from src.eda_e import flags, group_key, match_groups  # noqa: E402
from src.pairs import cx_roster, region_pair_counts  # noqa: E402

B, SEED = 2000, 0
fn, fer = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(fn)
cx = fn[fn["neuron_id"].isin(ids)].reset_index(drop=True)
ps = region_pair_counts(fer, ids)
n_pairs = len(ids) * (len(ids) - 1)


def recurrence(tx, ty, groups, tau_x, tau_y):
    """Per source group: type pairs recorded in x, and how many are recorded in y."""
    idx = tx.index[tx.index.get_level_values(0).isin(groups) & tx.index.get_level_values(1).isin(groups)]
    idx = idx[status(tx, idx, tau_x) == "recorded"]
    hit = status(ty, idx, tau_y) == "recorded"
    d = pd.DataFrame({"src": idx.get_level_values(0), "hit": hit.to_numpy()})
    return d.groupby("src")["hit"].agg(["sum", "size"])


def boot(tables, seed=SEED):
    """Pooled share and a percentile interval, resampling source groups (same draw in each direction)."""
    srcs = sorted(set().union(*[set(t.index) for t in tables]))
    a = np.array([[t["sum"].get(s, 0) for s in srcs] for t in tables]).sum(0)
    b = np.array([[t["size"].get(s, 0) for s in srcs] for t in tables]).sum(0)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(srcs), size=(B, len(srcs)))
    r = a[draw].sum(1) / b[draw].sum(1)
    return {"share": float(a.sum() / b.sum()), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "type_pairs": int(b.sum()), "source_groups": len(srcs)}


def lr(tau):
    key = cx["cell_type"]
    tl, tr = type_pair_table(cx, ps, key, "left"), type_pair_table(cx, ps, key, "right")
    g = set(key[cx["side"] == "left"].dropna()) & set(key[cx["side"] == "right"].dropna())
    return boot([recurrence(tl, tr, g, tau, tau), recurrence(tr, tl, g, tau, tau)])


def mirror_pairs(tau):
    """The neuron-pair check: types with one neuron per side give every neuron a known mirror."""
    typed = cx[cx["cell_type"].notna() & cx["side"].isin(["left", "right"])]
    n = typed.groupby(["cell_type", "side"], observed=True).size().unstack(fill_value=0)
    one = typed[typed["cell_type"].isin(n.index[(n["left"] == 1) & (n["right"] == 1)])]
    mirror = {a: b for _, g in one.groupby("cell_type", observed=True) for a, b in [tuple(g["neuron_id"]), tuple(g["neuron_id"])[::-1]]}
    kept = ps[ps >= tau]
    kept = kept[kept.index.get_level_values(0).isin(mirror) & kept.index.get_level_values(1).isin(mirror)]
    other = pd.MultiIndex.from_arrays([kept.index.get_level_values(0).map(mirror), kept.index.get_level_values(1).map(mirror)])
    hit = ps.reindex(other).fillna(0).to_numpy() >= tau
    src = one.set_index("neuron_id")["cell_type"].astype(str).reindex(kept.index.get_level_values(0)).to_numpy()
    return boot([pd.DataFrame({"src": src, "hit": hit}).groupby("src")["hit"].agg(["sum", "size"])]) | {"types": len(mirror) // 2}


def between(other_n, other_er, fafb_col, other_col, tau=10):
    groups = match_groups(cx[fafb_col], other_n[other_col])
    kf = group_key(cx[fafb_col], groups)
    ko = group_key(other_n[other_col], groups)
    keep = ko.isin(set(kf.dropna())).to_numpy()
    on, k = other_n[keep].reset_index(drop=True), ko[keep].reset_index(drop=True)
    tf = type_pair_table(cx, ps, kf, "right")
    to = type_pair_table(on, region_pair_counts(other_er, on["neuron_id"]), k, "right")
    both = set(kf[cx["side"] == "right"].dropna()) & set(k[on["side"] == "right"].dropna())
    sc = scale(tf, to, both)
    t2 = max(1, round(tau * sc))
    return {"scale": sc, "tau_other": t2, "same_tau": boot([recurrence(tf, to, both, tau, tau), recurrence(to, tf, both, tau, tau)]),
            "scaled_tau": boot([recurrence(tf, to, both, tau, t2), recurrence(to, tf, both, t2, tau)])}


fl = flags(cx)
pre = pd.read_feather(fetch_fafb.raw_file("neuropil_pre"))
post = pd.read_feather(fetch_fafb.raw_file("neuropil_post"))
reg = region_from_counts(ids.to_numpy(), fafb_neuropil_counts(pre[pre["pre_pt_root_id"].isin(ids)], post[post["post_pt_root_id"].isin(ids)]))
out_any = set(ps.index.get_level_values(0))
in_any = set(ps.index.get_level_values(1))
base = fl["region"].astype(str).str.replace(r"_[LR]$", "", regex=True)
hn, her = fetch_hemibrain.load_neurons(), fetch_hemibrain.load_edgeroi()
mn, _, mer, _ = fetch_malecns.load_brain()
groups = cx["cell_type"].where(cx["cell_type"].notna() & cx["side"].isin(["left", "right"]))

stats = {
    "roster": {"neurons": len(cx), "types": int(cx["cell_type"].nunique()), "ordered_pairs": n_pairs,
               "is_optic": int(fl["is_optic"].sum()), "region_unknown": int((cx["region"] == "unknown").sum()),
               "region_in_cx_neuropils": int(base.isin(["PB", "FB", "EB", "NO"]).sum()),
               "region_ties": int(reg["tied"].sum()), "untyped": int(cx["cell_type"].isna().sum()),
               "side": cx["side"].astype(str).value_counts().to_dict()},
    "labels": {str(t): {"positives": int((ps >= t).sum()), "prevalence": float((ps >= t).sum() / n_pairs),
                        "below_tau": int(((ps >= 1) & (ps < t)).sum())} for t in [1, 5, 10, 31]},
    "unrecorded": int(n_pairs - len(ps)),
    "missing": {"no_output_in_cx": int((~pd.Index(ids).isin(out_any)).sum()), "no_input_in_cx": int((~pd.Index(ids).isin(in_any)).sum()),
                "neither": int(((~pd.Index(ids).isin(out_any)) & (~pd.Index(ids).isin(in_any))).sum())},
    "nt": {"before": cx["nt_type"].value_counts().loc[lambda s: s > 0].to_dict(),
           "after": cx["nt_type_corrected"].value_counts().loc[lambda s: s > 0].to_dict(),
           "changed": int((cx["nt_type"].astype(str) != cx["nt_type_corrected"].astype(str)).sum()),
           "source_known_nt": int((cx["nt_correction_source"] == "known_nt").sum())},
    "cell_sub_class_missing": int(cx["cell_sub_class"].isna().sum()),
    "homologue_groups": int(cx["cell_type"].nunique() + cx["cell_type"].isna().sum()),
    "c_left_right": {str(t): lr(t) for t in [5, 10, 31]},
    "c_mirror_neuron_pairs": {str(t): mirror_pairs(t) for t in [5, 10, 31]},
    "c_hemibrain": between(hn, her, "hemibrain_type", "cell_type"),
    "c_malecns_never_for_claims": between(mn, mer, "cell_type", "flywire_type"),
}
(ROOT / "reports" / "lock_note_numbers.json").write_text(json.dumps(stats, indent=1, default=int))
print(json.dumps(stats, indent=1, default=int))
