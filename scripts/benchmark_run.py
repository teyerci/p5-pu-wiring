"""Dropout benchmark over seeds: fafb central-complex roster, τ = 10, q ∈ {1, 0.9, 0.7},
synapse and connection dropout, seeds 0-4. Writes reports/benchmark.json (one record per
condition and seed) and prints a summary (mean and sd over seeds)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                       # noqa: E402
from src.benchmark import run_condition          # noqa: E402
from src.pairs import cx_roster                  # noqa: E402

SEEDS = range(5)
CONDITIONS = [(1.0, "synapse"), (0.9, "synapse"), (0.7, "synapse"), (0.9, "connection"), (0.7, "connection")]
neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
results = []
for q, mode in CONDITIONS:
    for seed in (SEEDS if q < 1 else [0]):
        r = run_condition(neurons, er, ids, q, mode, seed=seed)
        results.append(r)
        print(q, mode, seed, {k: round(v["brier_y"], 6) for k, v in r["models"].items()}, flush=True)
        (ROOT / "reports" / "benchmark.json").write_text(json.dumps(results, indent=1, default=float))
