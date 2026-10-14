"""Is the held-out-type offset a train/test feature shift on the target side? (diagnostic)

Training features are built on a graph without any held-out source's edges into the roster
(src.pair_features.split_features), so every training target has lost those inputs, while
held-out pairs' target features (in_deg_j, frac_in_*, spatial) use the full graph. This
script reruns the benchmark with held-out target features taken from the training graph
(`target_graph="train"`) and compares with the full-graph runs of reports/benchmark_relative.json
(λ = 1e-4, same seeds and splits; the runs are deterministic).
q = 1 and q = 0.7 synapse dropout, seeds 0-2. Writes reports/benchmark_target_graph.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                       # noqa: E402
from src.benchmark import run_condition          # noqa: E402
from src.pairs import cx_roster                  # noqa: E402

LAM = 1e-4
neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
OUT = ROOT / "reports" / "benchmark_target_graph.json"
results = []
for seed in range(3):
    for q in (1.0, 0.7):
        results.append(run_condition(neurons, er, ids, q, "synapse", seed=seed, lam=LAM, target_graph="train"))
        OUT.write_text(json.dumps(results, indent=1, default=float))
        print(seed, q, flush=True)

full = [r for r in json.loads((ROOT / "reports" / "benchmark_relative.json").read_text())
        if r["lambda"] == LAM and r["mode"] == "synapse"]
rows = []
for graph, runs in (("full", full), ("train", results)):
    for r in runs:
        base = next(b for b in runs if b["seed"] == r["seed"] and b["q"] == 1.0)
        bl = base["models"]["naive"]["mean_p"] / base["models"]["naive"]["mean_y"]
        for name in ("naive", "pu_true_c"):
            m = r["models"][name]
            rows.append({"target_graph": graph, "q": r["q"], "model": name, "seed": r["seed"], "offset": bl,
                         "level": m["mean_p"] / m["mean_y"], "relative_level": m["mean_p"] / m["mean_y"] / bl,
                         "auc_y": m["auc_y"], "brier_y": m["brier_y"],
                         "top_bin": f"{m['reliability'][-1]['mean_p']:.4f}/{m['reliability'][-1]['rate_y']:.4f}"})
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
