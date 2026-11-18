"""Lock note addendum A7: edges per neuron on cell types present in both datasets, descriptive.

Edge = ordered roster pair with >= tau synapses summed over the CX neuropils (`region_pair_counts`, as the lock
note numbers use). FAFB: `cx_roster`, tau 10. MaleCNS: the brain neurons (`load_brain`, as `between` in
lock_note_numbers.py), tau 10 (same) and tau 18 (scaled: 1.75 x 10, read from reports/lock_note_numbers.json
"c_malecns_never_for_claims"). Type matching is that of lock_note_numbers.py: `match_groups` on FAFB cell_type
and MaleCNS flywire_type (exact names; a compound "A,B" joins its names). Both populations are cut to the shared
types, and partners are counted inside that population, so the FAFB value is not E7's (region-defined CX, tau 5,
all partners). Type sizes differ between the datasets, so a type-weighted summary is given beside the
neuron-weighted one. Writes reports/week9/edges_per_neuron.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import fetch_fafb, fetch_malecns  # noqa: E402
from src.edges_per_neuron import compare  # noqa: E402
from src.eda_e import group_key, match_groups  # noqa: E402
from src.pairs import cx_roster, region_pair_counts  # noqa: E402

TAU = 10


def main():
    fn, fer = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
    ids = cx_roster(fn)
    cx = fn[fn["neuron_id"].isin(ids)].reset_index(drop=True)
    mn, _, mer, _ = fetch_malecns.load_brain()
    groups = match_groups(cx["cell_type"], mn["flywire_type"])
    kf = pd.Series(group_key(cx["cell_type"], groups).to_numpy(), index=cx["neuron_id"].to_numpy())
    km = pd.Series(group_key(mn["flywire_type"], groups).to_numpy(), index=mn["neuron_id"].to_numpy())
    tau_m = json.loads((ROOT / "reports" / "lock_note_numbers.json").read_text())["c_malecns_never_for_claims"]["tau_other"]
    out = compare(ids, kf, region_pair_counts(fer, ids), mn["neuron_id"], km, region_pair_counts(mer, mn["neuron_id"]),
                  TAU, {"same_tau": TAU, "scaled_tau": tau_m})
    path = ROOT / "reports" / "week9" / "edges_per_neuron.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("shared types", out["shared_types"], "neurons FAFB", out["neurons_fafb"], "MaleCNS", out["neurons_malecns"],
          "| FAFB tau", TAU, "mean", round(out["fafb"]["mean"], 2), "median", out["fafb"]["median"])
    for name, s in out["settings"].items():
        print(name, "MaleCNS tau", s["tau_malecns"], "mean", round(s["mean"], 2), "median", s["median"], "ratio", round(s["ratio_mean"], 2),
              "| type-weighted FAFB", round(s["type_weighted"]["mean_fafb"], 2), "MaleCNS", round(s["type_weighted"]["mean_malecns"], 2),
              "median type ratio", round(s["type_weighted"]["median_type_ratio"], 2))


if __name__ == "__main__":
    main()
