"""FlyWire ↔ male CNS type mapping and coverage report (course week 3, cross-dataset matching).

The mapping uses only the annotation columns: fafb `cell_type` against malecns
`flywire_type`, never the malecns type names. Compound values ("A,B": one type
matching the union of several) are merged into match groups (src.eda_e.match_groups).
A group is matched when it has neurons in both datasets; otherwise it is specific
to one dataset. This is annotation coverage, not proof that a type is sex-specific:
annotation gaps and the volumes' coverage also leave types unmatched.

malecns is its brain part (src.eda_e.brain_part), as everywhere it is set beside fafb.
"""
from __future__ import annotations

import pandas as pd

from src.eda_e import flags, group_key, match_groups


def mapping_table(fafb, malecns) -> pd.DataFrame:
    """One row per match group: the fafb and malecns type names in it and neuron counts."""
    groups = match_groups(fafb["cell_type"], malecns["flywire_type"])
    kf, km = group_key(fafb["cell_type"], groups), group_key(malecns["flywire_type"], groups)
    f = pd.DataFrame({"group": kf, "type": fafb["cell_type"]}).dropna()
    m = pd.DataFrame({"group": km, "type": malecns["cell_type"].fillna("(untyped)")})[km.notna().to_numpy()]
    tab = pd.DataFrame({
        "fafb_types": f.groupby("group")["type"].agg(lambda t: ",".join(sorted(set(t)))),
        "fafb_neurons": f.groupby("group").size(),
        "malecns_types": m.groupby("group")["type"].agg(lambda t: ",".join(sorted(set(t)))),
        "malecns_neurons": m.groupby("group").size(),
    }).fillna({"fafb_neurons": 0, "malecns_neurons": 0})
    tab[["fafb_neurons", "malecns_neurons"]] = tab[["fafb_neurons", "malecns_neurons"]].astype(int)
    tab["status"] = "matched"
    tab.loc[tab["malecns_neurons"] == 0, "status"] = "fafb only"
    tab.loc[tab["fafb_neurons"] == 0, "status"] = "malecns only"
    return tab.sort_index()


def coverage(fafb, malecns, tab=None) -> tuple[dict, pd.DataFrame]:
    """Overall coverage and coverage by region (base neuropil and region group)."""
    tab = mapping_table(fafb, malecns) if tab is None else tab
    matched = set(tab.index[tab["status"] == "matched"])
    groups = match_groups(fafb["cell_type"], malecns["flywire_type"])
    in_f = group_key(fafb["cell_type"], groups).isin(matched).to_numpy()
    in_m = group_key(malecns["flywire_type"], groups).isin(matched).to_numpy()
    overall = {
        "groups": int(len(tab)), "matched_groups": len(matched),
        "fafb_only_groups": int((tab["status"] == "fafb only").sum()),
        "malecns_only_groups": int((tab["status"] == "malecns only").sum()),
        "fafb_types": int(fafb["cell_type"].nunique()),
        "fafb_types_matched": int(fafb.loc[in_f, "cell_type"].nunique()),
        "malecns_types": int(malecns["cell_type"].nunique()),
        "malecns_types_matched": int(malecns.loc[in_m, "cell_type"].nunique()),
        "fafb_neurons": int(len(fafb)), "fafb_neurons_matched": int(in_f.sum()),
        "malecns_neurons": int(len(malecns)), "malecns_neurons_matched": int(in_m.sum()),
        "malecns_neurons_without_flywire_type": int(malecns["flywire_type"].isna().sum()),
    }
    rows = []
    for ds, n, hit in [("fafb", fafb, in_f), ("malecns", malecns, in_m)]:
        fl = flags(n)
        d = pd.DataFrame({"region_group": fl["region_group"].to_numpy(), "neuropil": fl["neuropil"].to_numpy(),
                          "optic": fl["is_optic"].to_numpy(), "typed": n["cell_type"].notna().to_numpy(), "matched": hit})
        for level in ["region_group", "neuropil"]:
            g = d.groupby(level)
            rows.append(pd.DataFrame({"dataset": ds, "level": level, "neurons": g.size(), "typed": g["typed"].sum(),
                                      "matched": g["matched"].sum()}).rename_axis("region").reset_index())
    by_region = pd.concat(rows, ignore_index=True)
    by_region["matched_share"] = by_region["matched"] / by_region["neurons"]
    return overall, by_region
