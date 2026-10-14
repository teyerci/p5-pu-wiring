"""Is the held-out-type offset caused by the source one-hots? (diagnostic)

Held-out source types are never seen as sources in training, so their `type_i` one-hot is all
zeros at test time. This script refits the benchmark with that column removed and compares
the no-thinning offset (naive mean P(y=1) / true rate on held-out sources) and the relative
levels under q = 0.7 synapse dropout.
λ = 1e-4 fixed (the offset was +37% there with all features), seeds 0-2.
Writes reports/benchmark_type_onehot.json and prints the summary.
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

EXCLUDES = [(), ("type_i",)]
neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
OUT = ROOT / "reports" / "benchmark_type_onehot.json"
results = []
for exclude in EXCLUDES:
    for seed in range(3):
        for q in (1.0, 0.7):
            results.append(run_condition(neurons, er, ids, q, "synapse", seed=seed, lam=1e-4, exclude=exclude))
            OUT.write_text(json.dumps(results, indent=1, default=float))
            print(exclude, seed, q, flush=True)

rows = []
for r in results:
    base = next(b for b in results if b["seed"] == r["seed"] and b["q"] == 1.0 and b["excluded"] == r["excluded"])
    bl = base["models"]["naive"]["mean_p"] / base["models"]["naive"]["mean_y"]
    for name in ("naive", "pu_true_c"):
        m = r["models"][name]
        rows.append({"excluded": "+".join(r["excluded"]) or "none", "q": r["q"], "model": name, "seed": r["seed"],
                     "offset": bl, "level": m["mean_p"] / m["mean_y"], "relative_level": m["mean_p"] / m["mean_y"] / bl,
                     "auc_y": m["auc_y"], "brier_y": m["brier_y"]})
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
print(df.groupby(["excluded", "q", "model"], sort=False)[["offset", "level", "relative_level", "auc_y"]].mean().round(4))
