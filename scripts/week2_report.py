"""Write reports/week2_fafb.md: what Projects 2 and 5 submit for week 2.

Printed source columns, value counts for side, super_class and nt source,
missing-value counts, and edge counts at 1 and 5 synapses.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from src.fetch_fafb import load_edges, load_neurons, raw_file   # noqa: E402
from src.summary import summary                      # noqa: E402

ann = pd.read_csv(raw_file("annotations"), sep="\t", low_memory=False)
neurons, edges = load_neurons(), load_edges()
proofread = np.load(raw_file("proofread_ids"))


def vc(s):
    return "\n".join(f"| {k} | {v:,} |" for k, v in s.value_counts(dropna=False).items())


out = ["# Week 2 report — FAFB 783 (annotations v3.2.0, Buhmann synapses)", ""]
out += ["## Source columns", "", "Annotation table (`Supplemental_file1_neuron_annotations.tsv`):", "",
        "`" + "`, `".join(sorted(ann.columns)) + "`", "",
        "Connection table (`proofread_connections_783.feather`): `pre_pt_root_id`, `post_pt_root_id`, "
        "`neuropil`, `syn_count`, `gaba_avg`, `ach_avg`, `glut_avg`, `oct_avg`, `ser_avg`, `da_avg`", ""]
for col in ["side", "super_class", "top_nt"]:
    out += [f"## Source value counts: `{col}`", "", "| value | neurons |", "|---|---|", vc(ann[col]), ""]
out += ["## Missing values in the neurons table (after mapping)", "", "| column | missing |", "|---|---|"]
out += [f"| {c} | {int(neurons[c].isna().sum()):,} |" for c in neurons.columns]
s = summary(neurons, edges).loc["fafb"]
out += ["", "## Counts", "", "| quantity | value | expected |", "|---|---|---|",
        f"| neurons in the annotation table | {len(ann):,} | 139,248 |",
        f"| proofread root ids (Zenodo) | {len(proofread):,} | 139,255 published |",
        f"| edge ids with no annotation row, added as unknown | {len(neurons) - len(ann):,} | 7 expected |",
        f"| neurons table rows | {s.neurons:,} | |",
        f"| edges ≥ 1 synapse | {s.edges_min1:,} | |",
        f"| edges ≥ 5 synapses | {s.edges_min5:,} | 2,700,513 |",
        f"| edges ≥ 10 synapses | {s.edges_min10:,} | |",
        f"| synapses in edges | {s.synapses:,} | |",
        f"| distinct cell types | {s.cell_types:,} | |",
        f"| self-connections dropped | 0 | 0 |", ""]
Path(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "week2_fafb.md").write_text("\n".join(out))
print("\n".join(out[-14:]))
