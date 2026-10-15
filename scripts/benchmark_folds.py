"""Benchmark over homologue-group folds (worksheet D18): every roster neuron is a held-out
source exactly once per seed, so levels and calibration can be pooled over the whole roster
and checked per fold.

Per seed and fold: no thinning and q = 0.7 dropout (synapse by default), λ chosen by inner CV on the
training pairs (`lam="cv"`; "cv3" for 3 inner folds with the 1-SE rule; "cv3h" the same with inner folds of
homologue groups; a number fixes it), and
held-out target features on the training graph (`target_graph="train"`, see
scripts/benchmark_target_graph.py), so train and test features share one distribution, and
held-out numerics limited to the training range (`clip=True`, see reports/benchmark_folds.md;
pass "noclip" as the third argument to turn it off).
Pooled level = Σ_folds weight · mean P(y=1) / Σ_folds weight · true rate, with the
weight of a fold's evaluation set. Writes reports/benchmark_folds_<lam>[_noclip].json and
prints the summary. Usage: python scripts/benchmark_folds.py [lam] [seeds] [clip|noclip] [synapse|connection] [tag] [q=Q]
(seeds: a count n for seeds 0..n-1, or a range "a-b"; the dropout mode of the thinned runs, where "connection" is the
SCAR control and adds _connection to the file name; q=Q sets the thinned q, default 0.7, and adds _q<Q> otherwise)
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

Q_ARGS = [a for a in sys.argv[1:] if a.startswith("q=")]
Q = float(Q_ARGS[-1][2:]) if Q_ARGS else 0.7
if not 0 < Q < 1:
    raise SystemExit("q=Q must lie in (0, 1); the unthinned q = 1 runs are always included")
sys.argv = [a for a in sys.argv if not a.startswith("q=")]
LAM = "cv" if len(sys.argv) < 2 else sys.argv[1] if sys.argv[1] in ("cv", "cv3", "cv3h") else float(sys.argv[1])
INNER = "homologue" if LAM == "cv3h" else "source"         # cv3h: cv3 with inner folds by homologue group
S_ARG = sys.argv[2] if len(sys.argv) > 2 else "1"
SEEDS = range(int(S_ARG.split("-")[0]), int(S_ARG.split("-")[1]) + 1) if "-" in S_ARG else range(int(S_ARG))
CLIP = not (len(sys.argv) > 3 and sys.argv[3] == "noclip")
MODE = sys.argv[4] if len(sys.argv) > 4 else "synapse"
TAG = "_" + sys.argv[5] if len(sys.argv) > 5 else ""     # optional suffix, so a rerun keeps earlier files
N_FOLDS = 5
neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
OUT = ROOT / "reports" / f"benchmark_folds_{LAM if isinstance(LAM, str) else format(LAM, 'g')}{'' if CLIP else '_noclip'}{'' if MODE == 'synapse' else '_' + MODE}{'' if Q == 0.7 else f'_q{Q:g}'}{TAG}.json"
results = []
for seed in SEEDS:
    for fold in range(N_FOLDS):
        for q in (1.0, Q):
            results.append(run_condition(neurons, er, ids, q, MODE, seed=seed, lam="cv3" if LAM == "cv3h" else LAM, fold=fold,
                                         n_folds=N_FOLDS, target_graph="train", clip=CLIP, inner_groups=INNER))
            OUT.write_text(json.dumps(results, indent=1, default=float))
            print(seed, fold, q, "λ", results[-1]["lambda"], flush=True)

rows = []
for r in results:
    for name in ("naive", "pu_true_c"):
        m = r["models"][name]
        w = sum(b["weight"] for b in m["reliability"])
        rows.append({"seed": r["seed"], "fold": r["fold"], "q": r["q"], "model": name, "lambda": r["lambda"],
                     "c_train": r["realised_c_train"], "weight": w, "sum_p": w * m["mean_p"], "sum_y": w * m["mean_y"],
                     "level": m["mean_p"] / m["mean_y"], "auc_y": m["auc_y"]})
df = pd.DataFrame(rows)
print(df.drop(columns=["sum_p", "sum_y"]).round(4).to_string(index=False))
pooled = df.groupby(["seed", "q", "model"]).agg(sum_p=("sum_p", "sum"), sum_y=("sum_y", "sum"),
                                                level_min=("level", "min"), level_max=("level", "max"))
pooled["pooled_level"] = pooled["sum_p"] / pooled["sum_y"]
print(pooled[["pooled_level", "level_min", "level_max"]].round(4))
