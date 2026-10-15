"""Region rule for the D15 discussion: what happens to neurons whose most-synapse label is
outside every named neuropil (MaleCNS "unspecified" labels, FlyWire "None"), under two
counting measures. Evidence for workstream E; writes reports/region_rules.json.

MaleCNS (Traced neurons; syn-partners at confidence 0.5; all partners, Traced or not):
  connections = upstream + downstream (one count per synaptic connection on each side; the repo's rule)
  sites       = post sites (one row per post site, so equal to upstream) + unique T-bars
                (body_pre, x_pre, y_pre, z_pre); a T-bar's neuropil is the primary_post of its first row,
                an approximation (its partners can lie in different neuropils)
Rules: "literal" (any label can win; missing labels ignored) and "repo" (labels containing
"unspecified" never win, as src/fetch_malecns.py). Ties: alphabetically first, in both.
FAFB: FlyWire per-neuron neuropil counts (the repo's source); "literal" lets "None"/"UNASGD" win.
Also checks whether FlyWire's per-neuron output counts are connections or release sites: each
neuron's output count against its outgoing synapses to proofread partners (a subset of its connections).
Reads the 6.8 GB syn-partners file batch by batch: about 7 minutes and 15 GB of memory.
Usage: python scripts/region_rules.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import fetch_fafb, fetch_malecns                     # noqa: E402
from src.eda_e import MALECNS_NERVE_CORD, OPTIC, base_neuropil  # noqa: E402

OUT = REPO / "reports" / "region_rules.json"
UNSPEC = "unspecified"
res = {}


def top_region(counts: pd.DataFrame, exclude) -> pd.Series:
    """counts: neuron_id, label, count -> label with most count (ties: alphabetical)."""
    c = counts[~exclude(counts["label"])] if exclude is not None else counts
    c = c.groupby(["neuron_id", "label"], observed=True)["count"].sum().reset_index()
    c = c.sort_values(["neuron_id", "count", "label"], ascending=[True, False, True])
    return c.drop_duplicates("neuron_id").set_index("neuron_id")["label"]


def is_unspec_malecns(s):
    return s.str.contains(UNSPEC, regex=False)


def kind(region: pd.Series) -> pd.Series:
    """brain_named / optic_named / nerve_cord / unspecified label / unknown (for MaleCNS).
    Any other name counts as central; every label that can win here is a MaleCNS neuropil that
    canonical_neuropil accepts, so this matches in_brain from add_provisional_flags."""
    b = base_neuropil(region.fillna("unknown")).astype(str)
    out = np.where(region.isna() | (region == "unknown"), "unknown",
          np.where(region.fillna("").str.contains(UNSPEC, regex=False), "unspecified:" + region.fillna(""),
          np.where(b.isin(MALECNS_NERVE_CORD) | region.isin(MALECNS_NERVE_CORD), "nerve_cord",
          np.where(b.isin(OPTIC), "optic", "central"))))
    return pd.Series(out, index=region.index)


# ---------------- MaleCNS ----------------
neurons = fetch_malecns.load_neurons()[["neuron_id", "region"]]
ids = neurons["neuron_id"].to_numpy("int64")
keep = pa.array(ids)
reader = ipc.open_file(pa.memory_map(str(fetch_malecns.raw_file("syn_partners"))))
down, up, tb = [], [], []
maxc = 0
for i in range(reader.num_record_batches):
    b = pa.Table.from_batches([reader.get_batch(i)]).select(["body_pre", "body_post", "x_pre", "y_pre", "z_pre", "primary_post"])
    lab = pc.cast(b["primary_post"], pa.string())
    b = b.set_column(5, "primary_post", lab)
    mp = pc.is_in(b["body_pre"], value_set=keep)
    mq = pc.is_in(b["body_post"], value_set=keep)
    s = b.filter(mp)
    if s.num_rows:
        down.append(s.group_by(["body_pre", "primary_post"]).aggregate([([], "count_all")]).to_pandas())
        x, y, z = (s[c].to_numpy().astype(np.int64) for c in ("x_pre", "y_pre", "z_pre"))
        maxc = max(maxc, int(max(x.max(), y.max(), z.max())))
        key = (x << 42) | (y << 21) | z
        t = pd.DataFrame({"neuron_id": s["body_pre"].to_numpy(), "key": key,
                          "label": s["primary_post"].to_pandas()}).drop_duplicates(["neuron_id", "key"])
        tb.append(t)
    s = b.filter(mq)
    if s.num_rows:
        up.append(s.group_by(["body_post", "primary_post"]).aggregate([([], "count_all")]).to_pandas())
    if i % 500 == 0:
        print("batch", i, flush=True)
assert maxc < 2 ** 21, maxc                              # packing also needs x, y, z >= 0 (all coordinates are positive)


def tidy(parts, col):
    d = pd.concat(parts).rename(columns={col: "neuron_id", "primary_post": "label", "count_all": "count"})
    return d.groupby(["neuron_id", "label"], dropna=False)["count"].sum().reset_index()


down, up = tidy(down, "body_pre"), tidy(up, "body_post")
tb = pd.concat(tb).drop_duplicates(["neuron_id", "key"])
tbars = tb.groupby(["neuron_id", "label"], dropna=False).size().rename("count").reset_index()
del tb
measures = {"connections": pd.concat([down, up]), "sites": pd.concat([tbars, up])}
res["malecns_totals"] = {"downstream": int(down["count"].sum()), "upstream_post_sites": int(up["count"].sum()),
                         "tbars": int(tbars["count"].sum())}

for mname, counts in measures.items():
    counts = counts[counts["label"].notna()]                      # missing labels never count
    r1 = top_region(counts, None).reindex(ids)
    r2 = top_region(counts, is_unspec_malecns).reindex(ids)
    if mname == "connections":
        same = (r2.fillna("unknown").to_numpy() == neurons["region"].to_numpy()).mean()
        res["malecns_check_repo_rule_connections_equals_cache_region"] = float(same)
    k1 = kind(r1)
    unspec = k1.str.startswith("unspecified:")
    r2u = kind(r2[unspec])
    named = r2[unspec].dropna()
    res[f"malecns_{mname}"] = {
        "neurons": int(len(ids)),
        "literal_top_label_unspecified_by_label": k1[unspec].str.replace("unspecified:", "").value_counts().to_dict(),
        "literal_unknown": int((k1 == "unknown").sum()),
        "repo_region_of_those_neurons_by_kind": r2u.value_counts().to_dict(),
        "repo_region_of_those_neurons_top12": named.value_counts().head(12).to_dict(),
        "repo_unknown_total": int(r2.isna().sum()),
        # per literal label: where the repo rule puts those neurons, and their top named regions
        "repo_region_by_literal_label": {
            l.replace("unspecified:", ""): {"by_kind": kind(r2[k1 == l]).value_counts().to_dict(),
                                           "top8": r2[k1 == l].dropna().value_counts().head(8).to_dict()}
            for l in k1[unspec].unique()},
    }
    # share of the repo-rule region among the neuron's labelled synapses, for the neurons the literal rule gives an unspecified label
    c = counts[counts["neuron_id"].isin(r1[unspec].index)].groupby(["neuron_id", "label"])["count"].sum().reset_index()
    tot = c.groupby("neuron_id")["count"].sum()
    win = c.merge(named.rename("label").reset_index(), on=["neuron_id", "label"]).set_index("neuron_id")["count"]
    share = (win / tot.reindex(win.index))
    lab1 = r1[unspec]
    res[f"malecns_{mname}"]["repo_region_share_median_by_literal_label"] = {
        l: float(share.reindex(lab1[lab1 == l].index).median()) for l in lab1.unique()}

    # brain subset under three treatments
    def brain(k):
        return int(k.isin(["central", "optic"]).sum())
    k1_csv = k1.replace({"unspecified:Optic-unspecified(L)": "optic", "unspecified:Optic-unspecified(R)": "optic",
                         "unspecified:CentralBrain-unspecified": "central"})
    res[f"malecns_{mname}"]["brain_subset"] = {
        "literal + proposed CSV rows (Optic-/CentralBrain-unspecified in brain)": brain(k1_csv),
        "literal, unspecified labels rejected -> unknown": brain(k1),
        "repo rule": brain(kind(r2)),
    }
    print(mname, json.dumps(res[f"malecns_{mname}"], indent=1, default=str), flush=True)
del measures, down, up, tbars

# ---------------- FAFB ----------------
fn = fetch_fafb.load_neurons()[["neuron_id", "region"]]
fids = fn["neuron_id"].to_numpy("int64")
pre = pd.read_feather(fetch_fafb.raw_file("neuropil_pre")).rename(columns={"pre_pt_root_id": "neuron_id"})
post = pd.read_feather(fetch_fafb.raw_file("neuropil_post")).rename(columns={"post_pt_root_id": "neuron_id"})
both = pd.concat([pre, post]).rename(columns={"neuropil": "label"})
both = both[both["neuron_id"].isin(fids)]
unas = lambda s: s.isin(["None", "UNASGD"])
f1 = top_region(both, None).reindex(fids)
f2 = top_region(both, unas).reindex(fids)
moved = unas(f1.fillna(""))
fk = base_neuropil(f2[moved].fillna("unknown")).astype(str)
res["fafb"] = {
    "neurons": int(len(fids)),
    "labels_outside_neuropils": sorted(set(both["label"][unas(both["label"])])),
    "literal_top_label_unassigned": int(moved.sum()),
    "repo_region_of_those_by_kind": pd.Series(np.where(f2[moved].isna(), "unknown", np.where(fk.isin(OPTIC), "optic", "central"))).value_counts().to_dict(),
    "repo_region_of_those_top12": f2[moved].dropna().value_counts().head(12).to_dict(),
    "repo_rule_check_equals_cache_region": float((f2.fillna("unknown").to_numpy() == fn["region"].to_numpy()).mean()),
}
# Are FlyWire's per-neuron output counts connections or release sites? Compare each neuron's
# total output count with its outgoing synapses to proofread partners (a subset of all its connections).
conn = pd.read_feather(fetch_fafb.raw_file("connections"), columns=["pre_pt_root_id", "syn_count"])
out_proof = conn.groupby("pre_pt_root_id")["syn_count"].sum()
pre_tot = pre.groupby("neuron_id")["count"].sum()
j = pd.concat([pre_tot.rename("pre_count"), out_proof.rename("out_to_proofread")], axis=1, join="inner")
j = j[j.index.isin(fids)]
ratio = j["pre_count"] / j["out_to_proofread"]
# Each synapse has one pre and one post end: if both files count connections, their totals over
# all root ids are equal; counting release sites on the output side would make the pre total smaller.
res["fafb_pre_post_totals_all_ids"] = {"pre": int(pre["count"].sum()), "post": int(post["count"].sum())}
res["fafb_output_count_check"] = {
    "neurons": int(len(j)), "pre_count_total": int(j["pre_count"].sum()), "out_to_proofread_total": int(j["out_to_proofread"].sum()),
    "ratio_quantiles": ratio.quantile([0.01, 0.1, 0.5, 0.9]).round(3).to_dict(),
    "share_pre_count_at_least_out_to_proofread": float((ratio >= 1).mean()),
}
print(json.dumps({k: res[k] for k in ("fafb", "fafb_output_count_check")}, indent=1, default=str))
OUT.write_text(json.dumps(res, indent=1, default=str))
