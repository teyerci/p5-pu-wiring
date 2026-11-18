"""Week 9 runs (lock note addendum A1, A2, A5, A8): FAFB, τ = 10, seeds 10-14 x 5 homologue-group folds.

Conditions: q = 1 (no thinning; also the real-data description), synapse dropout q = 0.7 (H2
primary), connection dropout q = 0.7 and synapse dropout q = 0.5 (secondary). One JSON file per
condition under reports/week9/, rewritten after every run, so an interrupted job keeps what it has;
a run already in the file is skipped. The folds of every seed go to reports/week9/splits.csv for
week 11. Naive, naive/c and PU are fitted (A8).

Usage: python scripts/week9_naive.py [condition ...]   (default: all; names as in CONDITIONS)
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                                   # noqa: E402
from src.benchmark import fold_of_groups, homologue_groups   # noqa: E402
from src.pairs import cx_roster                              # noqa: E402
from src.week9 import benchmark_fold                         # noqa: E402

SEEDS, N_FOLDS, TAU, PER_POSITIVE = range(10, 15), 5, 10, 20     # fresh seeds: the pilot used 0-2
CONDITIONS = {"q1": (1.0, "synapse"), "synapse_q0.7": (0.7, "synapse"),
              "connection_q0.7": (0.7, "connection"), "synapse_q0.5": (0.5, "synapse")}
OUT = ROOT / "reports" / "week9"
OUT.mkdir(parents=True, exist_ok=True)

names = sys.argv[1:] or list(CONDITIONS)
unknown = set(names) - set(CONDITIONS)
if unknown:
    raise SystemExit(f"unknown condition(s): {sorted(unknown)}; choose from {list(CONDITIONS)}")

neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
roster = neurons[neurons["neuron_id"].isin(ids)]
groups = homologue_groups(roster)
pd.concat([pd.DataFrame({"neuron_id": roster["neuron_id"].to_numpy(), "homologue_group": groups.to_numpy(),
                         "seed": seed, "fold": fold_of_groups(groups, N_FOLDS, seed).to_numpy()}) for seed in SEEDS]
          ).to_csv(OUT / "splits.csv", index=False)
print("seeds", list(SEEDS), "folds", N_FOLDS, "tau", TAU, "unlabelled per positive", PER_POSITIVE, flush=True)

for name in names:
    q, mode = CONDITIONS[name]
    path = OUT / f"runs_{name}.json"
    results = json.loads(path.read_text()) if path.exists() else []
    done = {(r["seed"], r["fold"]) for r in results}
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            if (seed, fold) in done:
                continue
            t = time.time()
            r = benchmark_fold(neurons, er, ids, q, mode, seed=seed, fold=fold, tau=TAU, per_positive=PER_POSITIVE, n_folds=N_FOLDS, verbose=True)
            r["seconds"] = round(time.time() - t, 1)
            results.append(r)
            path.write_text(json.dumps(results, indent=1, default=float))
            print(name, "seed", seed, "fold", fold, "λ", {k: f"{v:g}" for k, v in r["lambda"].items()}, "brier naive", round(r["models"]["naive"]["brier_y"], 6),
                  "pu", round(r["models"]["pu"]["brier_y"], 6),
                  f"{r['seconds']} s", flush=True)
