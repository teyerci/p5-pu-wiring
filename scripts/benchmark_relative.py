"""Benchmark levels relative to a same-split baseline (H2 without the held-out-type offset).

Held-out source types carry a level offset even without thinning (+10% to +25% for the
naive model, depending on λ), which every thinned run shares because the split depends only
on the seed. For each seed this script runs the no-thinning condition and the thinned ones
on the same split, and reports each model's mean P(y=1) relative to the no-thinning naive
model of that seed. Under SCAR the naive ratio should be about c and the PU ratio about 1.
q = 0.7, synapse and connection dropout, λ ∈ {1e-2, 1e-4}, seeds 0-2.
Writes reports/benchmark_relative.json and prints the summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                       # noqa: E402
from src.benchmark import run_condition          # noqa: E402
from src.pairs import cx_roster                  # noqa: E402

neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
OUT = ROOT / "reports" / "benchmark_relative.json"
results = []
for lam in (1e-2, 1e-4):
    for seed in range(3):
        for q, mode in [(1.0, "synapse"), (0.7, "synapse"), (0.7, "connection")]:
            r = run_condition(neurons, er, ids, q, mode, seed=seed, lam=lam)
            r["lambda"] = lam
            results.append(r)
            OUT.write_text(json.dumps(results, indent=1, default=float))
            print(lam, seed, q, mode, flush=True)

rows = []
for r in results:
    base = next(b for b in results if b["lambda"] == r["lambda"] and b["seed"] == r["seed"] and b["q"] == 1.0)
    b = base["models"]["naive"]["mean_p"] / base["models"]["naive"]["mean_y"]
    for name in ("naive", "pu_true_c"):
        m = r["models"][name]
        rows.append({"lambda": r["lambda"], "seed": r["seed"], "condition": "none" if r["q"] == 1 else f"{r['mode']} q={r['q']}",
                     "model": name, "c_train": r["realised_c_train"], "level": m["mean_p"] / m["mean_y"],
                     "relative_level": (m["mean_p"] / m["mean_y"]) / b, "brier_y": m["brier_y"], "auc_y": m["auc_y"]})
df = pd.DataFrame(rows)
summary = df.groupby(["lambda", "condition", "model"])[["c_train", "level", "relative_level", "brier_y", "auc_y"]].agg(["mean", "std"])
print(summary.round(4).to_string())
