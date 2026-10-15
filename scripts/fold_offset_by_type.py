"""Which held-out types carry the fold-level offset? (diagnostic)

The homologue-group fold benchmark (scripts/benchmark_folds.py) shows a no-thinning offset that
varies strongly by fold even with target features from the training graph. This script reruns
the five folds of seed 0 without thinning at a fixed λ = 1e-4 (to separate the fold from the CV
choice of λ), keeps the per-pair scores, and splits each fold's excess predicted connections,
Σ weight · (P(y=1) − y), by the source's and the target's cell type.
With the argument "clip", held-out numerics are limited to the training range (`design(clip=True)`).
Writes reports/fold_offset_by_type.json (or ..._clip.json) and prints the summary.
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

LAM, N_FOLDS, SEED, TOP = 1e-4, 5, 0, 8
CLIP = len(sys.argv) > 1 and sys.argv[1] == "clip"
OUT = ROOT / "reports" / ("fold_offset_by_type_clip.json" if CLIP else "fold_offset_by_type.json")
neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
ctype = neurons.set_index("neuron_id")["cell_type"].astype(object).fillna("untyped")
out = []
for fold in range(N_FOLDS):
    r = run_condition(neurons, er, ids, 1.0, "synapse", seed=SEED, lam=LAM, fold=fold, n_folds=N_FOLDS,
                      target_graph="train", keep_scores=True, clip=CLIP)
    sc = r.pop("scores")
    sc["excess"] = sc["weight"] * (sc["f_naive"] - sc["y"])
    sc["pre_type"], sc["post_type"] = ctype.reindex(sc["pre_id"]).to_numpy(), ctype.reindex(sc["post_id"]).to_numpy()
    total_y = float((sc["weight"] * sc["y"]).sum())

    def by(col):
        g = sc.groupby(col).agg(excess=("excess", "sum"), true=("y", lambda v: float((v * sc.loc[v.index, "weight"]).sum())),
                                pred=("f_naive", lambda v: float((v * sc.loc[v.index, "weight"]).sum())))
        g["excess_share_of_true_total"] = g["excess"] / total_y
        return g.sort_values("excess", ascending=False)

    src, tgt = by("pre_type"), by("post_type")
    row = {"fold": fold, "test_sources": r["test_sources"], "source_types": int(sc["pre_type"].nunique()),
           "offset": r["models"]["naive"]["mean_p"] / r["models"]["naive"]["mean_y"], "auc_y": r["models"]["naive"]["auc_y"],
           "true_connections_weighted": total_y, "excess_weighted": float(sc["excess"].sum()),
           "top_source_types": src.head(TOP).reset_index().to_dict("records"),
           "bottom_source_types": src.tail(3).reset_index().to_dict("records"),
           "top_target_types": tgt.head(TOP).reset_index().to_dict("records")}
    out.append(row)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(fold, round(row["offset"], 3), "sources", row["test_sources"], flush=True)
    print(src.head(TOP).round(2).to_string(), flush=True)
