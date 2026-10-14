"""Paired comparison of naive vs PU at the true c over seeds (H2), from reports/benchmark.json.

Per condition and seed both models are fitted on the same thinned copy and scored on the
same pairs, so differences are paired. Reports the mean and sd of the Brier difference
(naive − PU; > 0 means PU is better), how many seeds favour PU, a paired t-test and an
exact sign test, and the same for the error in the mean predicted level. Writes
reports/benchmark_paired.csv. Five seeds give little power; read p-values as screening.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
runs = json.load(open(ROOT / "reports" / "benchmark.json"))
rows = []
for r in runs:
    if r["q"] == 1:
        continue
    n, p = r["models"]["naive"], r["models"]["pu_true_c"]
    rows.append({"condition": f"{r['mode']} q={r['q']}", "seed": r["seed"],
                 "brier_diff": n["brier_y"] - p["brier_y"],
                 "level_err_naive": abs(n["mean_p"] - n["mean_y"]), "level_err_pu": abs(p["mean_p"] - p["mean_y"])})
df = pd.DataFrame(rows)
out = []
for cond, g in df.groupby("condition", sort=False):
    d = g["brier_diff"].to_numpy()
    k = int((d > 0).sum())
    out.append({"condition": cond, "seeds": len(d), "brier_diff_mean": d.mean(), "brier_diff_sd": d.std(ddof=1),
                "seeds_pu_better": k, "t_test_p": stats.ttest_1samp(d, 0).pvalue,
                "sign_test_p": stats.binomtest(k, len(d), 0.5).pvalue,
                "level_err_naive": g["level_err_naive"].mean(), "level_err_pu": g["level_err_pu"].mean(),
                "seeds_pu_closer_level": int((g["level_err_pu"] < g["level_err_naive"]).sum())})
res = pd.DataFrame(out)
res.to_csv(ROOT / "reports" / "benchmark_paired.csv", index=False)
print(res.round(6).to_string(index=False))
