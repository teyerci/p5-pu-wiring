"""Summary of a homologue-group fold benchmark JSON (scripts/benchmark_folds.py).

Per seed, pooled over its folds (each fold's evaluation set weighted by its total weight):
offset without thinning, levels at q = 0.7 relative to the pooled no-thinning naive level,
Brier score, and a pooled reliability table (the folds' bins merged, then regrouped into
10 bins of equal weight by predicted mean). Paired over (seed, fold): the Brier difference
naive − PU at the true c at q = 0.7 (> 0 favours PU), with a sign test, and the H3 Spearman;
the naive model's shrinkage (thinned / unthinned mean prediction) by true synapse stratum; and how
often each q = 0.7 model has a lower Brier than the same fold's model fitted without thinning.
Usage: python scripts/benchmark_folds_summary.py reports/benchmark_folds_0.0001.json
Writes <json stem>_summary.csv and <json stem>_paired.json next to the input.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

path = Path(sys.argv[1])
runs = json.loads(path.read_text())
if len({(r["mode"], r.get("clip", False)) for r in runs}) != 1:
    raise ValueError("one dropout mode and clip setting per file")
rows = []
for r in runs:
    for name in ("naive", "pu_true_c"):
        m = r["models"][name]
        w = sum(b["weight"] for b in m["reliability"])
        rows.append({"seed": r["seed"], "fold": r["fold"], "q": r["q"], "model": name, "lambda": r["lambda"],
                     "c_train": r["realised_c_train"], "w": w, "sum_p": w * m["mean_p"], "sum_y": w * m["mean_y"],
                     "sum_brier": w * m["brier_y"], "auc_y": m["auc_y"], "level": m["mean_p"] / m["mean_y"],
                     "spearman_h3": r.get("h3_naive_vs_pu_true_c", {}).get("spearman", np.nan)})
df = pd.DataFrame(rows)

per_seed = []
for seed, g in df.groupby("seed"):
    base = g[(g.q == 1.0) & (g.model == "naive")]
    off = base.sum_p.sum() / base.sum_y.sum()
    rec = {"seed": seed, "mode": runs[0]["mode"], "clip": runs[0].get("clip", False),
           "folds": int(base.fold.nunique()), "offset_pooled": off,
           "offset_fold_min": base.level.min(), "offset_fold_max": base.level.max()}
    for name in ("naive", "pu_true_c"):
        t = g[(g.q == 0.7) & (g.model == name)]
        rec[f"{name}_level"] = t.sum_p.sum() / t.sum_y.sum()
        rec[f"{name}_relative"] = rec[f"{name}_level"] / off
        rec[f"{name}_brier"] = t.sum_brier.sum() / t.w.sum()
        rec[f"{name}_auc_mean"] = t.auc_y.mean()
    rec["c_train_mean"] = g[g.q == 0.7].c_train.mean()
    per_seed.append(rec)
per_seed = pd.DataFrame(per_seed)

thin = df[df.q == 0.7].pivot_table(index=["seed", "fold"], columns="model", values=["sum_brier", "w", "level", "spearman_h3"])
d = (thin["sum_brier"]["naive"] / thin["w"]["naive"]) - (thin["sum_brier"]["pu_true_c"] / thin["w"]["pu_true_c"])
k = int((d > 0).sum())
paired = {"runs": int(len(d)), "pu_lower_brier": k, "brier_diff_mean": float(d.mean()), "brier_diff_sd": float(d.std(ddof=1)),
          "sign_test_p": float(stats.binomtest(k, len(d), 0.5).pvalue),
          "pu_level_closer": int(((thin["level"]["pu_true_c"] - 1).abs() < (thin["level"]["naive"] - 1).abs()).sum()),
          "h3_spearman_min": float(thin["spearman_h3"]["naive"].min()) if thin["spearman_h3"]["naive"].notna().any() else None}


def pooled_reliability(seed, name, bins=10):
    parts = [b for r in runs if r["seed"] == seed and r["q"] == 0.7 for b in r["models"][name]["reliability"]]
    t = pd.DataFrame(parts).sort_values("mean_p")
    cum = (t["weight"].cumsum() - t["weight"]) / t["weight"].sum()
    t["bin"] = np.minimum((cum * bins + 1e-12).astype(int), bins - 1)
    return t.groupby("bin").apply(lambda x: pd.Series({"weight": x.weight.sum(),
                                                       "mean_p": np.average(x.mean_p, weights=x.weight),
                                                       "rate_y": np.average(x.rate_y, weights=x.weight)}), include_groups=False)


# Shrinkage of the naive model by true stratum: Σ weight · mean f at q = 0.7 over Σ weight · mean f
# without thinning, same seed and fold, pooled over all (seed, fold); c(x) constant under SCAR.
by_run = {(r["seed"], r["fold"], r["q"]): r["models"]["naive"]["by_true_stratum"] for r in runs}
shrink = {}
zero = {"weight": 0.0, "mean_f": 0.0}                     # a stratum empty in a run adds nothing
for st in dict.fromkeys(st for v in by_run.values() for st in v):      # union, first-seen order
    num = sum(v.get(st, zero)["weight"] * v.get(st, zero)["mean_f"] for (s, f, q), v in by_run.items() if q == 0.7)
    den = sum(by_run[(s, f, 1.0)].get(st, zero)["weight"] * by_run[(s, f, 1.0)].get(st, zero)["mean_f"]
              for (s, f, q) in by_run if q == 0.7)
    shrink[st] = num / den
paired["naive_shrinkage_by_true_stratum"] = shrink

# Brier at q = 0.7 against the same fold's model fitted without thinning (all labels).
b1 = {(r["seed"], r["fold"]): r["models"]["naive"]["brier_y"] for r in runs if r["q"] == 1.0}
for name in ("naive", "pu_true_c"):
    lower = [r["models"][name]["brier_y"] < b1[(r["seed"], r["fold"])] for r in runs if r["q"] == 0.7]
    paired[f"{name}_lower_brier_than_unthinned"] = int(sum(lower))
    paired[f"{name}_vs_unthinned_sign_test_p"] = float(stats.binomtest(int(sum(lower)), len(lower), 0.5).pvalue)
paired["brier_unthinned_mean"] = float(np.mean(list(b1.values())))

pd.set_option("display.width", 200)
print(per_seed.round(4).to_string(index=False))
print(json.dumps(paired, indent=1))
for name in ("naive", "pu_true_c"):
    print(name, "pooled reliability, top 3 bins, seed 0:")
    print(pooled_reliability(per_seed.seed.iloc[0], name).tail(3).round(4).to_string())
per_seed.to_csv(path.with_name(path.stem + "_summary.csv"), index=False)
(path.with_name(path.stem + "_paired.json")).write_text(json.dumps(paired, indent=1))
