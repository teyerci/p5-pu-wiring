"""Step 0 on the corrected tables (Shared Data Corrections): writes reports/step0.md and .json.

Builds any missing table. --force rebuilds the corrected tables first. A number with a
reference value in the document is checked against it, and the script stops on a mismatch.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.feather as pf  # noqa: E402

from eda_shared.corrections import kenyon_cell_types, nt_correction_summary, region_from_counts  # noqa: E402
from src import fetch_fafb, fetch_hemibrain, fetch_malecns  # noqa: E402
from src.eda_e import flags  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
force = "--force" in sys.argv

fn = fetch_fafb.load_neurons(force_refresh=force)
fe = fetch_fafb.load_edges()
fetch_hemibrain.load_neurons(force_refresh=force)
mn, me, _, brain = fetch_malecns.load_brain(force_refresh=force)
mall = fetch_malecns.load_neurons()

# neuron set
ann_ids = pd.read_csv(fetch_fafb.raw_file("annotations"), sep="\t", usecols=["root_id"], dtype={"root_id": "int64"})["root_id"]
proof = np.load(fetch_fafb.raw_file("proofread_ids")).astype("int64")   # uint64 beside int64 would compare as float64
neuron_set = {"total": len(fn), "annotation_rows": len(ann_ids), "kept": int(ann_ids.isin(proof).sum()),
              "dropped": int((~ann_ids.isin(proof)).sum()), "added": int((~pd.Series(proof).isin(ann_ids)).sum())}

# FAFB region
ids = fn["neuron_id"].to_numpy()
reg = region_from_counts(ids, fetch_fafb._neuropil_counts(ids))
assert reg.set_index("neuron_id")["region"].reindex(ids).astype(str).tolist() == fn["region"].astype(str).tolist()

# self-connections, from the raw tables the edges are built from
conn = fetch_fafb._connections()
self_fafb = int(conn.loc[conn["pre_pt_root_id"] == conn["post_pt_root_id"], ["pre_pt_root_id"]].drop_duplicates().shape[0])
w = pf.read_table(fetch_malecns.raw_file("weights"), columns=["body_pre", "body_post"])
same = w.filter(pc.equal(w["body_pre"], w["body_post"]))
self_malecns = int(pc.sum(pc.is_in(same["body_pre"], value_set=pa.array(fetch_malecns.traced_ids()))).as_py() or 0)


def central(neurons):
    fl = flags(neurons)
    return int(((~fl["is_optic"]) & (fl["region"] != "unknown")).sum())


nt = pd.concat([nt_correction_summary(fn), nt_correction_summary(mall)])
kc = kenyon_cell_types(pd.concat([fn[["dataset", "cell_type"]], mall[["dataset", "cell_type"]]]))
kc_f = fn[fn["cell_type"].astype("string").str.startswith("KC").fillna(False)]
cx = fn[fn["cell_class"] == "CX"]

stats = {
    "fafb_neurons": neuron_set,
    "fafb_super_class": fn["super_class"].value_counts().loc[lambda s: s > 0].to_dict(),
    "fafb_edges": {"tau1": len(fe), "ge5": int((fe["syn_count"] >= 5).sum())},
    "fafb_region": {"per_region": fn["region"].value_counts().to_dict(), "ties": int(reg["tied"].sum()),
                    "unknown": int((fn["region"] == "unknown").sum())},
    "malecns_brain": brain,
    "malecns_super_class_brain": mn["super_class"].value_counts().loc[lambda s: s > 0].to_dict(),
    "malecns_region": {"per_region": mn["region"].value_counts().to_dict(), "unknown_in_all_traced": int((mall["region"] == "unknown").sum())},
    "self_connections": {"fafb": self_fafb, "malecns": self_malecns},
    "central_population": {"fafb": central(fn), "malecns": central(mn)},
    "transmitter": nt.astype({"dataset": str, "nt_correction_source": str}).to_dict("records"),
    "kenyon_types": kc.to_dict("records"),
    "fafb_kenyon_corrected_ach": int((kc_f["nt_type_corrected"] == "acetylcholine").sum()),
    "fafb_kenyon_total": len(kc_f),
    "fafb_dopamine": {"before": int((fn["nt_type"] == "dopamine").sum()), "after": int((fn["nt_type_corrected"] == "dopamine").sum())},
    "cx_roster": {"neurons": len(cx), "nt_before": cx["nt_type"].value_counts().loc[lambda s: s > 0].to_dict(),
                  "nt_after": cx["nt_type_corrected"].value_counts().loc[lambda s: s > 0].to_dict(),
                  "nt_changed": int((cx["nt_type"].astype(str) != cx["nt_type_corrected"].astype(str)).sum())},
}

known = {
    "total": (neuron_set["total"], 139255), "kept": (neuron_set["kept"], 139241), "dropped": (neuron_set["dropped"], 7),
    "added": (neuron_set["added"], 14), "edges>=5": (stats["fafb_edges"]["ge5"], 2700513),
    "self fafb": (self_fafb, 0), "self malecns": (self_malecns, 101),
    "known_nt rows": (int(nt[(nt.dataset == "fafb") & (nt.nt_correction_source == "known_nt")]["rows"].sum()), 77169),
    "known_nt changed": (int(nt[(nt.dataset == "fafb") & (nt.nt_correction_source == "known_nt")]["changed"].sum()), 18513),
    "kenyon": (stats["fafb_kenyon_corrected_ach"], 5177),
    "dopamine before": (stats["fafb_dopamine"]["before"], 5909), "dopamine after": (stats["fafb_dopamine"]["after"], 638),
}
ref_sc = {"optic": 77537, "central": 32383, "sensory": 16904, "visual_projection": 8038, "ascending": 1750, "descending": 1303,
          "sensory_ascending": 612, "visual_centrifugal": 524, "motor": 110, "endocrine": 80, "unknown": 14}
known |= {f"super_class {k}": (stats["fafb_super_class"].get(k, 0), v) for k, v in ref_sc.items()}
bad = {k: v for k, v in known.items() if v[0] != v[1]}
print("reference checks:", len(known), "mismatches:", bad)

(ROOT / "reports" / "step0.json").write_text(json.dumps(stats, indent=1, default=int))
lines = ["# Step 0 on the corrected tables (eda_shared v0.2.0)", "",
         f"Built by `scripts/step0.py`. {len(known) - len(bad)} of {len(known)} reference values in Shared Data Corrections match."
         + (f" Mismatches: {bad}." if bad else ""), "",
         "| Quantity | Value |", "|---|---|",
         f"| FAFB neurons | {neuron_set['total']:,} = {neuron_set['kept']:,} kept + {neuron_set['added']} added; {neuron_set['dropped']} dropped |",
         "| FAFB superclasses | " + "; ".join(f"{k} {v:,}" for k, v in stats["fafb_super_class"].items()) + " |",
         f"| FAFB edges | τ = 1: {len(fe):,}; ≥ 5: {stats['fafb_edges']['ge5']:,} |",
         f"| FAFB region | ties {stats['fafb_region']['ties']}; unknown {stats['fafb_region']['unknown']}; per region in `step0.json` |",
         f"| MaleCNS brain part | kept {brain['neurons_kept']:,}; dropped {brain['neurons_dropped']:,}; ascending kept {brain['ascending_kept']:,}; sensory_ascending kept {brain['sensory_ascending_kept']:,}; region ties {brain['region_ties']} |",
         f"| MaleCNS brain edges | τ = 1: {brain['edges_tau1']:,}; ≥ 5: {brain['edges_ge5']:,}; synapses: brain {brain['edgeroi_syn_brain']:,}, nerve cord {brain['edgeroi_syn_nerve_cord']:,}, no location {brain['edgeroi_syn_no_location']:,} |",
         f"| Self-connections dropped | FAFB {self_fafb}; MaleCNS {self_malecns} |",
         f"| Central population | FAFB {stats['central_population']['fafb']:,}; MaleCNS {stats['central_population']['malecns']:,} |",
         "| Transmitter correction | " + "; ".join(f"{r.dataset} {r.nt_correction_source}: {r.rows:,} rows, {r.changed:,} changed" for r in nt.itertuples()) + " |",
         f"| Kenyon cells (FAFB) | {stats['fafb_kenyon_corrected_ach']:,} of {stats['fafb_kenyon_total']:,} acetylcholine; dopamine {stats['fafb_dopamine']['before']:,} → {stats['fafb_dopamine']['after']:,} |",
         f"| CX roster (P5) | {len(cx):,} neurons; transmitter changed for {stats['cx_roster']['nt_changed']} |",
         "", "Kenyon-cell types found (check that the list holds only Kenyon cells): "
         + ", ".join(f"{r.dataset}:{r.cell_type} ({r.neurons})" for r in kc.itertuples()) + "."]
(ROOT / "reports" / "step0.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if bad:
    sys.exit(1)
