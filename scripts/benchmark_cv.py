"""Benchmark with λ chosen by inner cross-validation on training pairs (src.benchmark.select_lambda),
relative to a same-split no-thinning baseline: q = 0.7 synapse and connection dropout, seeds 0-2.
Writes reports/benchmark_cv.json and prints relative levels, Brier and AUC."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                       # noqa: E402
from src.benchmark import run_condition          # noqa: E402
from src.pairs import cx_roster                  # noqa: E402

neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
OUT = ROOT / "reports" / "benchmark_cv.json"
results = []
for seed in range(3):
    for q, mode in [(1.0, "synapse"), (0.7, "synapse"), (0.7, "connection")]:
        results.append(run_condition(neurons, er, ids, q, mode, seed=seed, lam="cv"))
        OUT.write_text(json.dumps(results, indent=1, default=float))
        print(seed, q, mode, "λ", results[-1]["lambda"], flush=True)

rows = []
for r in results:
    base = next(b for b in results if b["seed"] == r["seed"] and b["q"] == 1.0)
    bl = base["models"]["naive"]["mean_p"] / base["models"]["naive"]["mean_y"]
    for name in ("naive", "pu_true_c"):
        m = r["models"][name]
        rows.append({"condition": "none" if r["q"] == 1 else r["mode"], "model": name, "seed": r["seed"],
                     "lambda": r["lambda"], "c_train": r["realised_c_train"], "offset": bl,
                     "relative_level": m["mean_p"] / m["mean_y"] / bl, "brier_y": m["brier_y"], "auc_y": m["auc_y"],
                     "top_bin": f"{m['reliability'][-1]['mean_p']:.4f}/{m['reliability'][-1]['rate_y']:.4f}"})
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
