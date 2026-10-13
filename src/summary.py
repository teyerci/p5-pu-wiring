"""summary(): counts to compare against the published figures."""
from __future__ import annotations

import pandas as pd


def summary(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Neuron count, edge counts at 1/5/10 synapses, total synapses and distinct cell types per dataset."""
    rows = []
    for ds, n in neurons.groupby("dataset", observed=True):
        e = edges[edges["dataset"] == ds]
        rows.append({
            "dataset": ds,
            "neurons": len(n),
            "edges_min1": len(e),
            "edges_min5": int((e["syn_count"] >= 5).sum()),
            "edges_min10": int((e["syn_count"] >= 10).sum()),
            "synapses": int(e["syn_count"].astype("int64").sum()),
            "cell_types": int(n["cell_type"].nunique()),
        })
    return pd.DataFrame(rows).set_index("dataset")
